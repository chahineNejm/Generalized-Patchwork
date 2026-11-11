"""Shared OISST comparison with explicit configuration and per-run artifacts."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))

import faiss
import gpytorch
import matplotlib
import numpy as np
import torch
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.exploratory_analysis import (
    analyze_run,
    explore_dataset,
    plot_hyperparameters,
    plot_positions,
    plot_training,
)
from experiments.ukhousing_benchmark import UKConfig, build_model, scores, train
from pwgp.config import HyperparameterBounds, NumericalConfig, OptimizerConfig
from pwgp.data import DatasetManager
from pwgp.data.download import file_checksum
from pwgp.models.standard_patchwork import StandardPatchworkGP
from pwgp.run_records import record_failures


@dataclass(frozen=True)
class OISSTConfig:
    day: str
    bounds: tuple[float, float, float, float]
    seed: int
    train_fraction: float
    validation_fraction: float
    save_inducing_positions: bool


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare_split(dataset, config):
    x, y = dataset.x, dataset.y
    west, east, south, north = config.bounds
    if not -180 <= west < east <= 0 or not -90 <= south < north <= 90:
        raise ValueError("This OISST protocol requires a western-hemisphere ROI with west < east")
    if not 0 < config.train_fraction < 1 or not 0 < config.validation_fraction < 1 - config.train_fraction:
        raise ValueError("Train/validation fractions must be positive and leave a nonempty test fraction")
    if x.shape != (len(y), 2) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OISST split requires finite x[N,2], y[N]")
    normalized = (x - [west % 360, south]) / [east - west, north - south]
    if (normalized <= 0).any() or (normalized >= 1).any():
        raise ValueError("OISST cell centers must lie strictly inside the declared ROI")
    indices = np.random.default_rng(config.seed).permutation(len(y))
    nt, nv = int(config.train_fraction * len(y)), int(config.validation_fraction * len(y))
    groups = {"train": indices[:nt], "validation": indices[nt:nt + nv], "test": indices[nt + nv:]}
    if any(len(rows) == 0 for rows in groups.values()):
        raise ValueError("Every OISST split must be nonempty")
    center, scale = float(y[groups["train"]].mean()), float(y[groups["train"]].std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training target SD must be finite and positive")
    return {**groups, "x": torch.from_numpy(normalized.astype(np.float64)),
            "y": torch.from_numpy((y.astype(np.float64) - center) / scale),
            "raw_x": x, "raw_y": y, "target_mean": center, "target_std": scale,
            "input_transform": "Longitude in [0,360); affine normalization using declared ROI",
            "target_transform": "Celsius standardized using training targets only"}


def acquire(config, *, data_root):
    manager = DatasetManager(data_root)
    with tqdm(total=3, desc="prepare:OISST", unit="stage") as progress:
        paths = manager.download("oisst", day=date.fromisoformat(config.day))
        progress.update()
        if len(paths) != 1:
            raise ValueError("OISST acquisition must produce exactly one daily file")
        dataset = manager.load("oisst", path=paths[0], bounds=config.bounds)
        progress.update()
        split = prepare_split(dataset, config)
        progress.update()
    manifest = {"acquisition": manager.manifest("oisst"), "day": config.day,
                "bounds": config.bounds, "sha256": file_checksum(paths[0]),
                "cleaning": dataset.metadata, "observations": len(dataset.y)}
    return split, manifest


@torch.no_grad()
def predict(model, x, *, batch_size):
    if batch_size < 1 or len(x) == 0:
        raise ValueError("Prediction requires positive batch size and nonempty inputs")
    device = next(model.parameters()).device
    model.eval()
    state = model.conditioned_state() if isinstance(model, StandardPatchworkGP) else None
    means, variances = [], []
    for batch in tqdm(x.split(batch_size), desc="predict", unit="batch", leave=False):
        options = {"state": state} if state is not None else {}
        mean, variance = model.predict(batch.to(device), observation_noise=True, **options)
        means.append(mean.cpu())
        variances.append(variance.cpu())
    return torch.cat(means).numpy(), torch.cat(variances).numpy()


def supports(model, kind):
    if kind == "patchwork":
        return None
    if kind == "vnngp":
        return model.variational_strategy.inducing_points.detach().cpu().numpy()[None]
    z = model.supports()
    z = torch.stack(z) if kind == "prospar" else z
    result = z.detach().cpu().numpy()
    return result[None] if result.ndim == 2 else result


def map_figure(coordinates, fields, *, destination):
    columns = min(len(fields), 4)
    rows = (len(fields) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows),
                             constrained_layout=True, squeeze=False)
    for ax, (title, values) in zip(axes.flat, fields.items(), strict=False):
        artist = ax.scatter(*coordinates.T, c=values, s=3, cmap="viridis")
        ax.set(title=title, xlabel="Longitude (0-360)", ylabel="Latitude")
        fig.colorbar(artist, ax=ax, shrink=0.7)
    for ax in list(axes.flat)[len(fields):]:
        ax.set_visible(False)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def learned_parameters(model, kind):
    if kind == "vnngp":
        return {"lengthscales": model.covar_module.base_kernel.lengthscale.detach().cpu().tolist(),
                "outputscales": [float(model.covar_module.outputscale.detach())],
                "noise_std": float(model.likelihood.noise.detach().sqrt().squeeze())}
    hyp = model.hyp
    learned = {"lengthscales": hyp.lengthscales().detach().cpu().tolist(),
               "outputscales": hyp.outputscales().detach().cpu().tolist(),
               "noise_std": float(hyp.noise_std().detach())}
    if kind == "prospar":
        learned.update(temperatures=hyp.temperatures().detach().cpu().tolist(),
                       exponent=float(hyp.exponent().detach()),
                       inducing_nuggets=[d.detach().cpu().tolist() for d in hyp.nuggets()],
                       prior_jitter=model.prior_jitter)
    return learned


def run_model(config, split, *, directory, save_inducing_positions, on_snapshot=None):
    output = Path(directory)
    output.mkdir(exist_ok=False)
    with record_failures(output, status_key="state"):
        write_json(output / "config.json", asdict(config))
        if not 0 <= config.warmup_epochs < config.epochs or config.snapshot_every < 1:
            raise ValueError("Require 0 <= warmup < epochs and positive snapshot interval")
        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested GPU is unavailable; no CPU fallback")
        torch.set_num_threads(config.threads)
        faiss.omp_set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        write_json(output / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "gpytorch": gpytorch.__version__,
            "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
            "faiss": importlib.metadata.version("faiss-cpu"), "dtype": "float64",
            "device": str(device), "threads": config.threads,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "bounds": asdict(HyperparameterBounds()),
            "numerical": asdict(NumericalConfig(weight_epsilon=config.weight_epsilon)),
            "active_prior_jitter": config.prospar.prior_jitter if config.model == "prospar"
            and config.prospar is not None else NumericalConfig().prior_jitter,
            "optimizer": asdict(OptimizerConfig(
                learning_rate_q=config.learning_rate_q,
                learning_rate_supports=config.learning_rate_structure,
                learning_rate_stitches=config.learning_rate_structure,
                learning_rate_hyperparameters=config.learning_rate_hyperparameters,
                warmup_fraction=config.warmup_epochs / config.epochs)),
            "source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[1])): file_checksum(p)
                              for p in sorted((Path(__file__).resolve().parents[1] / "src/pwgp").rglob("*.py"))},
        })
        visuals = output / "visuals"
        visuals.mkdir()
        started = time.perf_counter()
        model = build_model(config, split)
        initialization_seconds = time.perf_counter() - started
        write_json(output / "model_metadata.json", {
            "model_class": f"{type(model).__module__}.{type(model).__name__}",
            "package_version": importlib.metadata.version("generalized-patchwork-gp"),
            "kernel": "ARD RBF", "likelihood": "Gaussian",
            "inducing_positions_saved": save_inducing_positions and config.model != "patchwork",
            "reference": "arXiv:2311.08752v1 (6)-(13), approved derived objective/prediction"
            if config.model == "prospar" else None,
        })
        torch.save(model.state_dict(), output / "initial_model.pt")
        positions, position_epochs, parameter_history = [], [], []

        def snapshot(epoch, losses):
            write_json(output / "history.json", {"losses": losses})
            parameter_history.append({"epoch": epoch, **learned_parameters(model, config.model)})
            write_json(output / "hyperparameter_history.json", parameter_history)
            write_json(output / "status.json", {"state": "training", "epoch": epoch})
            if epoch == 0 or epoch % config.snapshot_every == 0 or epoch == config.epochs:
                if save_inducing_positions and config.model != "patchwork":
                    positions.append(supports(model, config.model).copy())
                    position_epochs.append(epoch)
                    np.savez_compressed(output / "inducing_positions.npz", epochs=position_epochs,
                                        supports=np.stack(positions))
                    plot_positions(positions[0], positions[-1], destination=visuals / "inducing_positions.png")
                if config.model == "gpw":
                    np.savez_compressed(output / f"stitches_epoch_{epoch}.npz",
                                        stitches=model.stitches().detach().cpu().numpy())
                plot_training(losses, destination=visuals / "training.png")
                plot_hyperparameters(parameter_history, destination=visuals / "hyperparameters.png")
                indices = split["validation"]
                mean, variance = predict(model, split["x"][indices], batch_size=config.prediction_batch_size)
                map_figure(split["raw_x"][indices],
                           {"Validation mean (standardized)": mean, "Observation SD": np.sqrt(variance)},
                           destination=visuals / "progress.png")
                torch.save(model.state_dict(), output / "checkpoint.pt")
                if on_snapshot is not None:
                    on_snapshot(output, epoch)

        snapshot(0, [])
        started = time.perf_counter()
        losses = train(model, split, config, snapshot)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - started
        torch.save(model.state_dict(), output / "model.pt")
        write_json(output / "learned_parameters.json", learned_parameters(model, config.model))
        if config.model in ("gpw", "prospar"):
            weights = []
            with torch.no_grad():
                for batch in tqdm(split["x"][split["test"]].split(config.prediction_batch_size),
                                  desc="weights", unit="batch", leave=False):
                    query = batch.to(device)
                    value = model.fusion_weights(query) if config.model == "gpw" else model.conditional_terms(query)["weights"]
                    weights.append(value.cpu())
            weights = torch.cat(weights).numpy()
            np.savez_compressed(output / "fusion_weights.npz", indices=split["test"], weights=weights)
            map_figure(split["raw_x"][split["test"]],
                       {f"Expert {j + 1} weight": weights[:, j] for j in range(weights.shape[1])},
                       destination=visuals / "weights.png")
        all_scores, prediction_seconds = {}, 0.0
        for label in tqdm(("validation", "test"), desc="score", unit="split"):
            indices = split[label]
            started = time.perf_counter()
            mean, variance = predict(model, split["x"][indices], batch_size=config.prediction_batch_size)
            prediction_seconds += time.perf_counter() - started
            standardized = scores(split["y"][indices].numpy(), mean, variance)
            raw_mean = mean * split["target_std"] + split["target_mean"]
            raw_variance = variance * split["target_std"] ** 2
            all_scores[label] = {"standardized": standardized,
                                 "celsius": scores(split["raw_y"][indices], raw_mean, raw_variance)}
            np.savez_compressed(output / f"{label}_predictions.npz", indices=indices,
                                coordinates=split["raw_x"][indices], mean=raw_mean,
                                observation_variance=raw_variance, truth=split["raw_y"][indices])
            if label == "test":
                map_figure(split["raw_x"][indices], {
                    "Observed (C)": split["raw_y"][indices], "Predicted (C)": raw_mean,
                    "Absolute error (C)": np.abs(raw_mean - split["raw_y"][indices]),
                    "Observation SD (C)": np.sqrt(raw_variance)}, destination=visuals / "maps.png")
        metrics = {**all_scores["test"]["celsius"], "final_loss": losses[-1],
                   "initialization_seconds": initialization_seconds, "training_seconds": training_seconds,
                   "prediction_seconds": prediction_seconds, "test_points": len(split["test"]),
                   "peak_cuda_megabytes": torch.cuda.max_memory_allocated(device) / 2**20
                   if device.type == "cuda" else None}
        write_json(output / "all_scores.json", all_scores)
        write_json(output / "metrics.json", metrics)
        analyze_run(output)
        write_json(output / "status.json", {"state": "complete"})
        del snapshot
        model = None
        gc.collect()
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return output


def run_comparison(data_config: OISSTConfig, runs: list[tuple[str, UKConfig]], *,
                   directory, data_root, source_provenance, source_zip,
                   on_snapshot=None, on_complete=None):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    with record_failures(output, status_key="state"):
        if not runs or len({name for name, _ in runs}) != len(runs):
            raise ValueError("Comparison requires nonempty, uniquely named runs")
        if any(config.seed != data_config.seed for _, config in runs):
            raise ValueError("All runs must use the declared shared data seed")
        write_json(output / "data_config.json", asdict(data_config))
        write_json(output / "runs.json", [{"name": name, **asdict(config)} for name, config in runs])
        write_json(output / "source_provenance.json", source_provenance)
        (output / "source_snapshot.zip").write_bytes(source_zip)
        split, manifest = acquire(data_config, data_root=data_root)
        write_json(output / "data_manifest.json", manifest)
        write_json(output / "preprocessing.json", {key: split[key] for key in
                   ("target_mean", "target_std", "input_transform", "target_transform")})
        np.savez_compressed(output / "split.npz", **{key: split[key] for key in ("train", "validation", "test")})
        explore_dataset(split, directory=output / "exploratory")
        summaries = []
        for index, (name, config) in enumerate(tqdm(runs, desc="compare:OISST", unit="model")):
            result = run_model(config, split, directory=output / f"{index:02d}_{config.model}",
                               save_inducing_positions=data_config.save_inducing_positions,
                               on_snapshot=on_snapshot)
            summaries.append({"model": name, **json.loads((result / "metrics.json").read_text())})
            write_json(output / "comparison.json", summaries)
            archive = output / f"{index:02d}_{config.model}.zip"
            with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
                for path in sorted(output.iterdir()):
                    if path.is_file() and (path.suffix in (".json", ".npz") or path.name == "source_snapshot.zip"):
                        bundle.write(path, path.name)
                for path in sorted(result.rglob("*")):
                    if path.is_file():
                        bundle.write(path, path.relative_to(output))
            if on_complete is not None:
                on_complete(result, archive, summaries[-1])
        write_json(output / "status.json", {"state": "complete"})
    return output
