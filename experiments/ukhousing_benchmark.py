"""Explicit-model benchmarks on a checksum-pinned reconstruction of UKHousing."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import faiss
import gpytorch
import matplotlib
import numpy as np
import torch
from scipy.special import ndtr
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwgp import SVGP, GeneralizedPatchworkGP, ProSparGP, fit
from pwgp.config import HyperparameterBounds, NumericalConfig, OptimizerConfig, ProSparConfig
from pwgp.data.download import file_checksum
from pwgp.initialization import (
    hierarchical_initialization,
    hierarchical_supports,
    initialize_stitches,
    observed_supports,
    patch_neighbor_graph,
)
from pwgp.models.standard_patchwork import StandardPatchworkGP
from pwgp.models.vnngp import VNNGP, strict_gpytorch
from pwgp.run_records import record_failures


@dataclass(frozen=True)
class UKConfig:
    model: str = "gpw"
    epochs: int = 100
    warmup_epochs: int = 20
    inducing_points: int = 1024
    patches: int = 8
    stitches_per_edge: int = 8
    q_rank: int = 32
    patch_neighbors: int = 2
    vnngp_neighbors: int = 32
    vnngp_batch_size: int = 512
    exact_maximum_points: int = 512
    exact_stitches_per_edge: int = 2
    exact_partition: str = "adaptive"
    exact_grid_shape: tuple[int, int] | None = None
    exact_tree_patches: int | None = None
    exact_training_points: str = "all"
    exact_gpw_reference: dict[str, int] | None = None
    batch_size: int = 1024
    prediction_batch_size: int = 256
    lengthscale: float = 0.1
    outputscale: float = 1.0
    noise_std: float = 0.1
    learning_rate_q: float = 0.002
    learning_rate_structure: float = 0.001
    learning_rate_hyperparameters: float = 0.001
    exact_learning_rate: float = 0.01
    weight_epsilon: float = 0.01
    seed: int = 31
    snapshot_every: int = 10
    threads: int = 2
    device: str = "cuda:0"
    prospar: ProSparConfig | None = None


def prepare_split(x, y, *, seed=31, train_fraction=0.64, validation_fraction=0.16):
    if x.shape != (len(y), 2) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected finite UKHousing x[N,2], y[N]")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 - train_fraction):
        raise ValueError("Invalid training/validation fractions")
    indices = np.random.default_rng(seed).permutation(len(y))
    nt, nv = int(train_fraction * len(y)), int(validation_fraction * len(y))
    train, valid, test = indices[:nt], indices[nt:nt + nv], indices[nt + nv:]
    center, scale = float(y[train].mean()), float(y[train].std())
    if not scale > 0:
        raise ValueError("Training target variance must be positive")
    lower, span = np.array([-5.7, 50.0]), np.array([7.5, 5.5])
    normalized = (x - lower) / span
    if (normalized <= 0).any() or (normalized >= 1).any():
        raise ValueError("Coordinates outside declared open UKHousing region")
    return {"x": torch.from_numpy(normalized), "y": torch.from_numpy((y - center) / scale),
            "raw_x": x, "raw_y": y, "train": train, "validation": valid, "test": test,
            "target_mean": center, "target_std": scale, "seed": seed,
            "input_transform": "Fixed ROI: (longitude+5.7)/7.5, (latitude-50)/5.5",
            "target_transform": "Natural log GBP, standardized from training targets only"}


def scores(y, mean, variance):
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("Invalid Gaussian predictions")
    error, sd = y - mean, np.sqrt(variance)
    z = error / sd
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error))),
            "nlpd": float(np.mean(0.5 * (np.log(2 * np.pi * variance) + z**2))),
            "crps": float(np.mean(sd * (z * (2 * ndtr(z) - 1)
                                         + 2 * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
                                         - 1 / np.sqrt(np.pi)))),
            "coverage_95": float(np.mean(np.abs(z) <= 1.959963984540054)),
            "mean_interval_width": float(np.mean(2 * 1.959963984540054 * sd))}


def initial_gpw_points(x, reference, split):
    """Use the comparison's exact shared initialization or construct it explicitly."""
    if reference is None or set(reference) != {"inducing_points", "patches", "seed", "threads"}:
        raise ValueError("Matched Patchwork requires an explicit GPW initialization reference; use select_runs")
    if reference["inducing_points"] % reference["patches"]:
        raise ValueError("Reference GPW inducing budget must divide evenly across patches")
    if "gpw_initialization" in split:
        if split["gpw_initialization_reference"] != reference:
            raise ValueError("Shared initialization differs from the requested GPW reference")
        initial = split["gpw_initialization"]
        if not torch.equal(x[initial["indices"]], initial["supports"]):
            raise ValueError("Shared GPW locations do not match their source training rows")
        return initial
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(reference["threads"])
    try:
        return hierarchical_initialization(x, reference["patches"],
            reference["inducing_points"] // reference["patches"], seed=reference["seed"])
    finally:
        torch.set_num_threads(previous_threads)


def build_model(config, split):
    x, y = split["x"][split["train"]], split["y"][split["train"]]
    device = torch.device(config.device)
    numerical = NumericalConfig(weight_epsilon=config.weight_epsilon)
    options = {"lengthscale": config.lengthscale, "outputscale": config.outputscale,
               "noise_std": config.noise_std}
    if config.model == "svgp":
        supports = observed_supports(x, config.inducing_points, seed=config.seed)
        return SVGP(supports.to(device), numerical=numerical, **options)
    if config.model == "gpw":
        if config.inducing_points % config.patches:
            raise ValueError("GPW inducing budget must divide evenly across patches")
        initial = initial_gpw_points(x, {key: getattr(config, key)
            for key in ("inducing_points", "patches", "seed", "threads")}, split)
        supports = initial["supports"]
        edges = patch_neighbor_graph(supports, neighbors=config.patch_neighbors)
        stitches = initialize_stitches(supports, edges, config.stitches_per_edge)
        model = GeneralizedPatchworkGP(supports.to(device), stitches.to(device), edges,
                                       q_rank=config.q_rank, numerical=numerical, **options)
        model.register_buffer("initial_support_locations", supports.to(device))
        model.register_buffer("initial_cluster_centers", initial["centers"].to(device))
        model.register_buffer("source_initial_indices", torch.as_tensor(
            split["train"][initial["indices"].numpy()], device=device))
        return model
    if config.model == "prospar":
        if config.prospar is None:
            raise ValueError("ProSpar requires explicit temperatures, exponent, inducing_nuggets and prior_jitter")
        if config.inducing_points % config.patches:
            raise ValueError("ProSpar runner inducing budget must divide evenly across experts")
        supports = hierarchical_supports(x, config.patches, config.inducing_points // config.patches,
                                        seed=config.seed)
        return ProSparGP(supports.to(device), numerical=numerical, **options, **asdict(config.prospar))
    if config.model == "patchwork":
        centers = None
        if config.exact_training_points == "gpw_initial":
            initial = initial_gpw_points(x, config.exact_gpw_reference, split)
            rows = initial["indices"].flatten()
            x, y = x[rows], y[rows]
            if config.exact_partition == "gpw_voronoi":
                centers = initial["centers"].cpu().numpy()
        elif config.exact_training_points == "all":
            rows = torch.arange(len(x))
            if config.exact_partition == "gpw_voronoi":
                raise ValueError("GPW Voronoi regions require exact_training_points='gpw_initial'")
        else:
            raise ValueError(f"Unknown exact_training_points: {config.exact_training_points}")
        model = StandardPatchworkGP(x.to(device), y.to(device),
            maximum_points=config.exact_maximum_points, stitches_per_edge=config.exact_stitches_per_edge,
            partition="voronoi" if config.exact_partition == "gpw_voronoi" else config.exact_partition,
            grid_shape=config.exact_grid_shape, centers=centers, tree_patches=config.exact_tree_patches,
            seed=config.seed, jitter=numerical.prior_jitter, **options)
        model.register_buffer("source_training_indices", torch.as_tensor(split["train"][rows.numpy()], device=device))
        if config.exact_training_points == "gpw_initial":
            model.register_buffer("reference_supports", initial["supports"].to(device))
            model.register_buffer("initial_cluster_centers", initial["centers"].to(device))
        return model
    if config.model == "vnngp":
        return VNNGP(x.to(device), neighbors=config.vnngp_neighbors,
                     batch_size=config.vnngp_batch_size, jitter=numerical.prior_jitter, **options)
    raise ValueError(f"Unknown model: {config.model}")


@torch.no_grad()
def predict(model, x, *, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    state = model.conditioned_state() if isinstance(model, StandardPatchworkGP) else None
    means, variances = [], []
    for batch in x.split(batch_size):
        options = {"state": state} if state is not None else {}
        mean, variance = model.predict(batch.to(device), observation_noise=True, **options)
        means.append(mean.cpu())
        variances.append(variance.cpu())
    return torch.cat(means).numpy(), torch.cat(variances).numpy()


def progress_figure(path, model, split, losses, config):
    validation = split["validation"][:2048]
    mean, variance = predict(model, split["x"][validation], batch_size=config.prediction_batch_size)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(range(1, len(losses) + 1), losses)
    axes[0].set(title=f"{config.model}: training", xlabel="Epoch", ylabel="Negative objective / observation")
    for ax, values, title in [(axes[1], mean, "Validation mean (standardized log price)"),
                               (axes[2], np.sqrt(variance), "Validation observation SD")]:
        im = ax.scatter(*split["raw_x"][validation].T, c=values, s=4, cmap="viridis")
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
        fig.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(path / "progress.png", dpi=130)
    plt.close(fig)


def train(model, split, config, callback):
    x, y = split["x"][split["train"]], split["y"][split["train"]]
    if config.model in ("gpw", "svgp", "prospar"):
        optimizer = OptimizerConfig(
            learning_rate_q=config.learning_rate_q,
            learning_rate_supports=config.learning_rate_structure,
            learning_rate_stitches=config.learning_rate_structure,
            learning_rate_hyperparameters=config.learning_rate_hyperparameters,
            warmup_fraction=config.warmup_epochs / config.epochs,
        )
        return fit(model, x, y, epochs=config.epochs, batch_size=config.batch_size,
                   seed=config.seed, optimizer=optimizer,
                   on_epoch=lambda epoch, current, history: callback(epoch, history.losses)).losses
    losses = []
    if config.model == "patchwork":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.exact_learning_rate)
        for epoch in (bar := tqdm(range(config.epochs), desc="train:standard-patchwork", unit="epoch")):
            model.train()
            optimizer.zero_grad()
            loss = model.loss()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite exact Patchwork likelihood")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite exact Patchwork gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
            bar.set_postfix(loss=losses[-1])
            if callback(epoch + 1, losses) is True:
                break
        return losses
    q_parameters = list(model.variational_parameters())
    hyperparameters = list(model.hyperparameters())
    optimizer = torch.optim.Adam([{"params": q_parameters, "lr": config.learning_rate_q},
                                  {"params": hyperparameters, "lr": 0.0}])
    objective = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=len(x))
    y = y.to(config.device)
    for epoch in (bar := tqdm(range(config.epochs), desc="train:vnngp", unit="epoch")):
        model.train()
        if epoch == config.warmup_epochs:
            optimizer.param_groups[1]["lr"] = config.learning_rate_hyperparameters
        accumulated = seen = 0
        for _ in range(model.variational_strategy._total_training_batches):
            optimizer.zero_grad()
            with strict_gpytorch():
                output = model(None)
                indices = model.variational_strategy.current_training_indices.to(y.device)
                loss = -objective(output, y[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite VNNGP objective")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite VNNGP gradient")
            optimizer.step()
            accumulated += float(loss.detach()) * len(indices)
            seen += len(indices)
        losses.append(accumulated / seen)
        bar.set_postfix(loss=losses[-1])
        if callback(epoch + 1, losses) is True:
            break
    return losses


def run_ukhousing(config: UKConfig, split, *, directory, on_snapshot=None):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    with record_failures(output, status_key="state"):
        (output / "config.json").write_text(
            json.dumps(asdict(config), indent=2, allow_nan=False), encoding="utf-8"
        )
        (output / "status.json").write_text(json.dumps({"state": "preparing"}), encoding="utf-8")
        return _execute_ukhousing(config, split, output=output, on_snapshot=on_snapshot)


def _execute_ukhousing(config, split, *, output, on_snapshot):
    if config.epochs < 1:
        raise ValueError(f"{config.model}: epochs={config.epochs} must be positive")
    if config.model != "patchwork" and not 0 <= config.warmup_epochs < config.epochs:
        raise ValueError(f"{config.model}: require 0 <= warmup_epochs={config.warmup_epochs} "
                         f"< epochs={config.epochs}")
    if config.snapshot_every < 1:
        raise ValueError("snapshot_every must be positive")
    torch.set_num_threads(config.threads)
    faiss.omp_set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected CUDA device is unavailable; no CPU fallback")
    environment = {"python": platform.python_version(), "torch": torch.__version__,
                   "gpytorch": gpytorch.__version__, "numpy": np.__version__,
                   "faiss": importlib.metadata.version("faiss-cpu"),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "kernel": "ARD RBF", "likelihood": "Gaussian",
                   "dtype": "float64", "bounds": asdict(HyperparameterBounds()),
                   "numerical": asdict(NumericalConfig(weight_epsilon=config.weight_epsilon)),
                   "fusion_covariance": "Cov(local_functions | U, Delta_S=0)" if config.model == "gpw" else None,
                   "standard_patchwork_objective": "joint_negative_log_likelihood(Y, Delta_S=0)"
                   if config.model == "patchwork" else None,
                   "source_sha256": {str(path): file_checksum(path) for path in Path("src/pwgp").rglob("*.py")},
                   "runner_sha256": file_checksum(Path(__file__))}
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (output / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model = build_model(config, split)
    metadata = {"model_class": f"{type(model).__module__}.{type(model).__name__}",
                "package_version": importlib.metadata.version("generalized-patchwork-gp")}
    if config.model == "prospar":
        metadata.update({"paper": "https://arxiv.org/abs/2311.08752v1",
                         "equations": "Derived from (6)-(13); printed (14)/(16) not used literally",
                         "configuration": asdict(config.prospar)})
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_seconds = time.perf_counter() - started
    torch.save(model.state_dict(), output / "initial_model.pt")

    def snapshot(epoch, losses):
        (output / "history.json").write_text(json.dumps({"losses": losses}), encoding="utf-8")
        if epoch % config.snapshot_every == 0 or epoch == config.epochs:
            progress_figure(output, model, split, losses, config)
            torch.save(model.state_dict(), output / "checkpoint.pt")
            if on_snapshot is not None:
                on_snapshot(output, epoch)

    started = time.perf_counter()
    losses = train(model, split, config, snapshot)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    torch.save(model.state_dict(), output / "model.pt")
    if config.model == "prospar":
        learned = {"lengthscales": model.hyp.lengthscales().detach().cpu().tolist(),
                   "outputscales": model.hyp.outputscales().detach().cpu().tolist(),
                   "noise_std": float(model.hyp.noise_std().detach()),
                   "temperatures": model.hyp.temperatures().detach().cpu().tolist(),
                   "exponent": float(model.hyp.exponent().detach()),
                   "inducing_nuggets": [d.detach().cpu().tolist() for d in model.hyp.nuggets()],
                   "prior_jitter": model.prior_jitter}
        (output / "learned_parameters.json").write_text(
            json.dumps(learned, indent=2, allow_nan=False), encoding="utf-8")
    rows = {}
    prediction_seconds = 0.0
    for label in ("validation", "test"):
        indices = split[label]
        started = time.perf_counter()
        mean, variance = predict(model, split["x"][indices], batch_size=config.prediction_batch_size)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prediction_seconds += time.perf_counter() - started
        standardized = scores(split["y"][indices].numpy(), mean, variance)
        raw_mean = mean * split["target_std"] + split["target_mean"]
        raw_variance = variance * split["target_std"]**2
        original = scores(split["raw_y"][indices], raw_mean, raw_variance)
        rows[label] = {"standardized_log_price": standardized, "log_gbp": original}
        np.savez_compressed(output / f"{label}_predictions.npz", indices=indices,
                            coordinates=split["raw_x"][indices], mean=raw_mean,
                            observation_variance=raw_variance, truth=split["raw_y"][indices])
        if label == "test":
            fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
            for position, (ax, values, title) in enumerate(zip(axes, [split["raw_y"][indices], raw_mean,
                                               np.abs(split["raw_y"][indices] - raw_mean),
                                               np.sqrt(raw_variance)],
                                         ["Observed log price", "Predicted log price", "Absolute error", "Observation SD"], strict=True)):
                limits = {"vmin": float(split["raw_y"][indices].min()),
                          "vmax": float(split["raw_y"][indices].max())} if position < 2 else {}
                im = ax.scatter(*split["raw_x"][indices].T, c=values, s=2, cmap="viridis", **limits)
                ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
                fig.colorbar(im, ax=ax, shrink=0.7)
            fig.savefig(output / "maps.png", dpi=150)
            plt.close(fig)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = {**rows["test"]["standardized_log_price"],
               "initialization_seconds": initialization_seconds, "training_seconds": training_seconds,
               "prediction_seconds": prediction_seconds, "final_loss": losses[-1],
               "test_points": len(split["test"]),
               "peak_cuda_megabytes": torch.cuda.max_memory_allocated(device) / 2**20
               if device.type == "cuda" else None,
               "patches": model.patches if config.model in ("gpw", "patchwork", "prospar") else None,
               "inducing_points": len(split["train"]) if config.model == "vnngp"
               else config.inducing_points if config.model in ("gpw", "svgp", "prospar") else 0}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "all_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "status.json").write_text(json.dumps({"state": "complete"}), encoding="utf-8")
    del snapshot
    model = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    return output


def archive_run(directory):
    """Save one completed run with its shared split and source/data provenance."""
    directory = Path(directory)
    if json.loads((directory / "status.json").read_text())["state"] != "complete":
        raise ValueError("Only completed UKHousing runs can be archived")
    root = directory.parent
    shared = [root / name for name in ("source_snapshot.zip", "split.npz",
              "preprocessing.json", "data_manifest.json", "comparison.csv", "source_provenance.json")]
    for path in shared:
        if not path.is_file():
            raise FileNotFoundError(path)
    target = root.parent / f"{root.name}_{directory.name}.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in shared + sorted(path for path in directory.rglob("*") if path.is_file()):
            bundle.write(path, path.relative_to(root))
    return target
