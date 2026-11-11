"""Official Heaton MODIS split with explicit variational-model run records."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path("results/.matplotlib").resolve())
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import matplotlib
import numpy as np
import pyreadr
import torch
from scipy.special import ndtr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwgp import SVGP, GeneralizedPatchworkGP, ProSparGP, fit
from pwgp.config import HyperparameterBounds, NumericalConfig, OptimizerConfig, ProSparConfig
from pwgp.data.download import download_file, file_checksum
from pwgp.initialization import (
    hierarchical_supports,
    initialize_stitches,
    observed_supports,
    patch_neighbor_graph,
)
from pwgp.run_records import record_failures
from pwgp.training import predict_batched

REVISION = "32a64d27fad5b3acf95c73b8ca9d9cc035219769"
SOURCE = (
    "https://raw.githubusercontent.com/finnlindgren/heatoncomparison/"
    f"{REVISION}/Data/AllSatelliteTemps.RData"
)
SHA256 = "876bb415454e39d164f7cedd066a73d9ee787267bf7f7b2a8caecf061f8f30c5"


@dataclass(frozen=True)
class HeatonConfig:
    model: str = "patchwork"
    inducing_points: int = 480
    patches: int = 8
    stitches_per_edge: int = 8
    neighbors: int = 2
    q_rank: int = 32
    epochs: int = 100
    warmup_epochs: int = 20
    batch_size: int = 2048
    prediction_batch_size: int = 2048
    lengthscale: float = 0.1
    outputscale: float = 1.0
    noise_std: float = 0.1
    learning_rate_q: float = 2e-3
    learning_rate_supports: float = 1e-3
    learning_rate_stitches: float = 1e-3
    learning_rate_hyperparameters: float = 1e-3
    weight_epsilon: float = 1e-2
    seed: int = 31
    threads: int = 2
    snapshot_every: int = 10
    device: str = "cuda:0"
    prospar: ProSparConfig | None = None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_heaton(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = pyreadr.read_r(str(path))["all.sat.temps"]
    if list(frame.columns) != ["Lon", "Lat", "MaskTemp", "TrueTemp"]:
        raise ValueError("Unexpected official MODIS schema")
    coordinates = frame[["Lon", "Lat"]].to_numpy(dtype=np.float64)
    truth = frame["TrueTemp"].to_numpy(dtype=np.float64)
    masked = frame["MaskTemp"].to_numpy(dtype=np.float64)
    training = np.isfinite(masked)
    testing = ~training & np.isfinite(truth)
    if (len(frame), int(training.sum()), int(testing.sum())) != (150000, 105569, 42740):
        raise ValueError("Official MODIS training/test counts do not match")
    if not np.isfinite(coordinates).all() or not np.array_equal(masked[training], truth[training]):
        raise ValueError("Invalid official training coordinates or targets")
    return coordinates, truth, training, testing


def prepare_data(coordinates, truth, training):
    origin = coordinates[training].min(0)
    span = np.ptp(coordinates[training], axis=0)
    if np.any(span <= 0):
        raise ValueError("Each coordinate must have nonzero training range")
    x = (coordinates - origin) / span
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design[training], truth[training], rcond=None)[0]
    trend = design @ coefficients
    residual = truth[training] - trend[training]
    scale = float(residual.std())
    if not scale > 0:
        raise ValueError("Training residual scale must be positive")
    preprocessing = {
        "coordinate_origin": origin.tolist(),
        "coordinate_span": span.tolist(),
        "mean_design": ["intercept", "normalized_lon", "normalized_lat"],
        "training_ols_coefficients": coefficients.tolist(),
        "training_residual_std": scale,
        "fit_on": "official_training_rows_only",
    }
    return torch.from_numpy(x), torch.from_numpy(residual / scale), trend, scale, preprocessing


def scores(truth: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> dict[str, float]:
    if not np.isfinite(mean).all() or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise FloatingPointError("Predictions must be finite with positive observation variance")
    sd = np.sqrt(variance)
    error = mean - truth
    z = -error / sd
    density = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    crps = sd * (z * (2 * ndtr(z) - 1) + 2 * density - 1 / np.sqrt(np.pi))
    halfwidth = 1.959963984540054 * sd
    outside = np.maximum(np.abs(error) - halfwidth, 0)
    return {
        "rmse_celsius": float(np.sqrt(np.mean(error**2))),
        "mae_celsius": float(np.mean(np.abs(error))),
        "nlpd": float(np.mean(0.5 * (np.log(2 * np.pi * variance) + error**2 / variance))),
        "crps_celsius": float(crps.mean()),
        "coverage_95": float(np.mean(np.abs(error) <= halfwidth)),
        "mean_interval_width_celsius": float(np.mean(2 * halfwidth)),
        "interval_score_95_celsius": float(np.mean(2 * halfwidth + 40 * outside)),
    }


def draw_field(axis, coordinates, values, *, title, cmap, vmin=None, vmax=None):
    lon = np.unique(coordinates[:, 0])
    lat = np.unique(coordinates[:, 1])
    raster = np.full((len(lat), len(lon)), np.nan)
    raster[np.searchsorted(lat, coordinates[:, 1]), np.searchsorted(lon, coordinates[:, 0])] = (
        values
    )
    artist = axis.imshow(
        raster,
        origin="lower",
        extent=(lon.min(), lon.max(), lat.min(), lat.max()),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
    return artist


def save_progress(path, config, model, history, positions, initial_supports, initial_stitches):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(
        np.arange(1, len(history.losses) + 1),
        history.losses,
        color="#176B87",
        marker=".",
        markersize=3,
    )
    axes[0].axvline(config.warmup_epochs, color="#9B2226", linestyle="--")
    axes[0].set(
        title=f"{config.model.upper()} optimization", xlabel="Epoch", ylabel="Negative objective / N"
    )
    groups = initial_supports.shape[0]
    colors = plt.colormaps["tab10"](np.arange(groups) % 10)
    trajectory = np.stack(positions)
    for patch in range(groups):
        axes[1].scatter(*initial_supports[patch].T, s=9, color=colors[patch], alpha=0.2)
        axes[1].plot(
            trajectory[:, patch, :, 0],
            trajectory[:, patch, :, 1],
            color=colors[patch],
            alpha=0.2,
            linewidth=0.5,
        )
        axes[1].scatter(
            *positions[-1][patch].T, s=10, color=colors[patch], label=f"Patch {patch + 1}"
        )
    if config.model == "patchwork":
        stitches = model.stitches().detach().cpu().numpy().reshape(-1, 2)
        axes[1].scatter(*initial_stitches.reshape(-1, 2).T, marker="x", c="#888888", s=14)
        axes[1].scatter(*stitches.T, marker="x", c="#151515", s=20, label="Stitches")
        axes[1].legend(fontsize=7, ncol=3, loc="upper right")
    axes[1].set(
        title=f"Inducing positions after epoch {len(history.losses)}",
        xlabel="Normalized longitude",
        ylabel="Normalized latitude",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    fig.savefig(path / "progress.png", dpi=140)
    plt.close(fig)


def save_maps(path, config, coordinates, truth, training, testing, mean, variance, latent_variance):
    observed = np.where(training, truth, np.nan)
    error = np.where(testing, mean - truth, np.nan)
    low, high = float(np.nanmin(truth)), float(np.nanmax(truth))
    limit = float(np.nanmax(np.abs(error)))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    fields = [
        (truth, "Observed reference (including held-out values)", "viridis", low, high),
        (observed, "Official training observations", "viridis", low, high),
        (mean, f"{config.model.upper()} predictive mean", "viridis", low, high),
        (error, "Held-out signed error (C)", "RdBu_r", -limit, limit),
        (np.sqrt(variance), "Predictive observation standard deviation (C)", "magma", None, None),
        (np.sqrt(latent_variance), "Latent standard deviation (C)", "magma", None, None),
    ]
    for axis, (values, title, cmap, vmin, vmax) in zip(axes.flat, fields, strict=True):
        artist = draw_field(axis, coordinates, values, title=title, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(artist, ax=axis, shrink=0.85)
    fig.savefig(path / "maps.png", dpi=160)
    plt.close(fig)


def run_heaton(
    config: HeatonConfig,
    *,
    data_root="data/heaton",
    results_root="results/heaton",
    run_directory: str | None = None,
    capture_git: bool = True,
    on_snapshot: Callable[[Path, int], None] | None = None,
) -> Path:
    if config.model not in {"patchwork", "svgp", "prospar"}:
        raise ValueError("model must be patchwork, svgp or prospar")
    if config.model == "prospar" and config.prospar is None:
        raise ValueError("ProSpar requires explicit temperatures, exponent, inducing_nuggets and prior_jitter")
    if not 0 <= config.warmup_epochs < config.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs)")
    if config.inducing_points % config.patches or config.snapshot_every < 1:
        raise ValueError("Inducing count must divide patches, and snapshot_every must be positive")
    numerical = NumericalConfig(weight_epsilon=config.weight_epsilon)
    bounds = HyperparameterBounds()
    optimizer = OptimizerConfig(
        learning_rate_q=config.learning_rate_q,
        learning_rate_supports=config.learning_rate_supports,
        learning_rate_stitches=config.learning_rate_stitches,
        learning_rate_hyperparameters=config.learning_rate_hyperparameters,
        warmup_fraction=config.warmup_epochs / config.epochs,
        alternate_patchwork_updates=True,
        gradient_clip=10.0,
        weight_decay=0.0,
    )
    settings = {
        **asdict(config),
        "numerical": asdict(numerical),
        "bounds": asdict(bounds),
        "optimizer": asdict(optimizer),
        "kernel": "ARD_RBF",
        "dtype": "float64",
        "device": config.device,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = (
        Path(results_root) / f"{config.model}_{digest}_{timestamp}"
        if run_directory is None
        else Path(run_directory)
    )
    output.mkdir(parents=True, exist_ok=False)
    with record_failures(output, status_key="status"):
        write_json(output / "config.json", settings)
        write_json(output / "status.json", {"status": "preparing", "epoch": 0})
        return _execute_heaton(config, output, data_root, numerical, bounds, optimizer,
                               settings, capture_git, on_snapshot)


def _execute_heaton(config, output, data_root, numerical, bounds, optimizer,
                    settings, capture_git, on_snapshot):
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(config.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device is unavailable; no CPU fallback")
        torch.cuda.set_device(device)
    raw = Path(data_root) / "raw" / "AllSatelliteTemps.RData"
    download_file(SOURCE, raw, checksum=SHA256)
    coordinates, truth, training, testing = load_heaton(raw)
    x, y, trend, scale, preprocessing = prepare_data(coordinates, truth, training)
    x_train = x[training]
    np.savez_compressed(
        output / "split.npz",
        training_rows=np.flatnonzero(training),
        test_rows=np.flatnonzero(testing),
    )
    manifest = {
        "source": SOURCE,
        "source_revision": REVISION,
        "sha256": file_checksum(raw),
        "training_count": int(training.sum()),
        "test_count": int(testing.sum()),
        "preprocessing": preprocessing,
    }
    write_json(output / "dataset.json", manifest)
    write_json(Path(data_root) / "manifest.json", manifest)
    package = importlib.metadata.version("generalized-patchwork-gp")
    environment = {
        "package_version": package,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cholesky_solver": "two_triangular_solves",
        "fusion_covariance": "Cov(local_functions | U, Delta_S=0)" if config.model == "patchwork" else None,
        "latent_variance": "w^T S w + alpha^T R alpha" if config.model == "patchwork"
        else "conditional_residual + inducing_posterior",
        "training_objective": "ELBO",
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory
        if device.type == "cuda"
        else None,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "execution_mode": "repository" if capture_git else "standalone_snapshot",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if capture_git
        else None,
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True)
        if capture_git
        else None,
        "source_sha256": {str(p): file_checksum(p) for p in sorted(Path("src").rglob("*.py"))},
        "runner_sha256": file_checksum(Path(__file__)),
    }
    write_json(output / "environment.json", environment)
    options = {
        "lengthscale": config.lengthscale,
        "outputscale": config.outputscale,
        "noise_std": config.noise_std,
        "numerical": numerical,
        "bounds": bounds,
    }
    if config.model == "patchwork":
        supports = hierarchical_supports(
            x_train, config.patches, config.inducing_points // config.patches, seed=config.seed
        ).to(device)
        edges = patch_neighbor_graph(supports, neighbors=config.neighbors)
        stitches = initialize_stitches(supports, edges, config.stitches_per_edge)
        model = GeneralizedPatchworkGP(supports, stitches, edges, q_rank=config.q_rank, **options)
    elif config.model == "prospar":
        supports = hierarchical_supports(
            x_train, config.patches, config.inducing_points // config.patches, seed=config.seed
        ).to(device)
        model = ProSparGP(supports, **options, **asdict(config.prospar))
    else:
        supports = observed_supports(x_train, config.inducing_points, seed=config.seed).to(device)
        model = SVGP(supports, **options)
    write_json(output / "model_metadata.json", {
        "model_class": f"{type(model).__module__}.{type(model).__name__}",
        "package_version": package,
        "prospar": asdict(config.prospar) if config.model == "prospar" else None,
        "paper": "https://arxiv.org/abs/2311.08752v1" if config.model == "prospar" else None,
        "equations": "Derived from (6)-(13); printed (14)/(16) not used literally"
        if config.model == "prospar" else None,
    })
    initial_supports = (
        (torch.stack(model.supports()) if config.model == "prospar" else model.supports())
        .detach()
        .cpu()
        .numpy()
        .reshape(-1, config.inducing_points // config.patches, 2)
    )
    if config.model == "svgp":
        initial_supports = model.supports().detach().cpu().numpy()[None]
    initial_stitches = (
        model.stitches().detach().cpu().numpy()
        if config.model == "patchwork"
        else np.empty((0, 0, 2))
    )
    positions = [initial_supports.copy()]
    stitch_positions = [initial_stitches.copy()]
    snapshot_epochs = [0]
    torch.save(model.state_dict(), output / "initial_model.pt")
    np.savez_compressed(
        output / "initial_geometry.npz",
        supports=initial_supports,
        stitches=initial_stitches,
        edges=np.asarray(model.edges if config.model == "patchwork" else [], dtype=int),
    )
    print(
        json.dumps(
            {"output": str(output.resolve()), "settings": settings, "dataset": manifest}, indent=2
        ),
        flush=True,
    )

    def snapshot(epoch, current_model, history):
        write_json(
            output / "status.json",
            {
                "status": "training",
                "epoch": epoch,
                "epochs": config.epochs,
                "loss": history.losses[-1],
                "elapsed_seconds": history.elapsed_seconds,
            },
        )
        write_json(output / "history.json", asdict(history))
        if epoch == 1 or epoch % config.snapshot_every == 0 or epoch == config.epochs:
            positions.append(
                (torch.stack(current_model.supports()) if config.model == "prospar"
                 else current_model.supports()).detach().cpu().numpy().reshape(initial_supports.shape)
            )
            stitch_positions.append(
                current_model.stitches().detach().cpu().numpy()
                if config.model == "patchwork"
                else initial_stitches.copy()
            )
            snapshot_epochs.append(epoch)
            np.savez_compressed(
                output / "geometry_history.npz",
                epochs=snapshot_epochs,
                supports=np.stack(positions),
                stitches=np.stack(stitch_positions),
            )
            save_progress(
                output,
                config,
                current_model,
                history,
                positions,
                initial_supports,
                initial_stitches,
            )
            torch.save(
                {"epoch": epoch, "state_dict": current_model.state_dict()}, output / "checkpoint.pt"
            )
            if on_snapshot is not None:
                on_snapshot(output, epoch)
            print(
                json.dumps(
                    {
                        "model": config.model,
                        "epoch": epoch,
                        "loss": history.losses[-1],
                        "elapsed_seconds": history.elapsed_seconds,
                    }
                ),
                flush=True,
            )

    history = fit(
        model,
        x_train,
        y,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed,
        optimizer=optimizer,
        on_epoch=snapshot,
    )
    write_json(output / "history.json", asdict(history))
    started = time.perf_counter()
    latent_mean, latent_variance = predict_batched(
        model, x, batch_size=config.prediction_batch_size, observation_noise=False
    )
    mean = latent_mean.numpy() * scale + trend
    latent_variance = latent_variance.numpy() * scale**2
    noise_variance = float(model.hyp.noise_std().detach()) ** 2 * scale**2
    variance = latent_variance + noise_variance
    prediction_seconds = time.perf_counter() - started
    metrics = scores(truth[testing], mean[testing], variance[testing])
    metrics.update(
        {
            "training_seconds": history.elapsed_seconds,
            "prediction_seconds": prediction_seconds,
            "final_loss": history.losses[-1],
            "test_points": int(testing.sum()),
            "noise_std_celsius": float(np.sqrt(noise_variance)),
            "peak_cuda_megabytes": history.peak_cuda_megabytes,
        }
    )
    write_json(output / "metrics.json", metrics)
    np.savez_compressed(
        output / "predictions.npz",
        coordinates=coordinates,
        mean=mean,
        observation_variance=variance,
        latent_variance=latent_variance,
        truth=truth,
        training_mask=training,
        test_mask=testing,
    )
    learned = {
        "lengthscales": model.hyp.lengthscales().detach().cpu().tolist(),
        "outputscales": model.hyp.outputscales().detach().cpu().tolist(),
        "noise_std": float(model.hyp.noise_std().detach()),
    }
    if config.model == "prospar":
        learned.update({"temperatures": model.hyp.temperatures().detach().cpu().tolist(),
                        "exponent": float(model.hyp.exponent().detach()),
                        "inducing_nuggets": [d.detach().cpu().tolist() for d in model.hyp.nuggets()],
                        "prior_jitter": model.prior_jitter})
    if config.model == "patchwork":
        with torch.no_grad():
            weights = torch.cat(
                [
                    model.fusion_weights(batch.to(device)).cpu()
                    for batch in x.split(config.prediction_batch_size)
                ]
            ).numpy()
        np.savez_compressed(output / "fusion_weights.npz", weights=weights)
        learned["edges"] = model.edges
        learned["weight_min"] = float(weights.min())
        learned["weight_max"] = float(weights.max())
        rows = (config.patches + 3) // 4
        fig, axes = plt.subplots(
            rows, 4, figsize=(16, 3.5 * rows), constrained_layout=True, squeeze=False
        )
        for patch, axis in enumerate(axes.flat):
            if patch >= config.patches:
                axis.set_visible(False)
                continue
            artist = draw_field(
                axis,
                coordinates,
                weights[:, patch],
                title=f"Patch {patch + 1} weight",
                cmap="coolwarm",
                vmin=float(weights.min()),
                vmax=float(weights.max()),
            )
            fig.colorbar(artist, ax=axis, shrink=0.85)
        fig.savefig(output / "weights.png", dpi=140)
        plt.close(fig)
    write_json(output / "learned_parameters.json", learned)
    save_maps(
        output, config, coordinates, truth, training, testing, mean, variance, latent_variance
    )
    torch.save(model.state_dict(), output / "model.pt")
    write_json(
        output / "status.json", {"status": "complete", "epoch": config.epochs, "metrics": metrics}
    )
    print(
        json.dumps(
            {"model": config.model, "metrics": metrics, "output": str(output.resolve())}, indent=2
        ),
        flush=True,
    )
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name, field in HeatonConfig.__dataclass_fields__.items():
        if name == "prospar":
            continue
        parser.add_argument(
            "--" + name.replace("_", "-"), type=type(field.default), default=field.default
        )
    parser.add_argument("--run-directory")
    parser.add_argument("--prospar-config", type=json.loads,
                        help="JSON with explicit temperatures, exponent, inducing_nuggets, prior_jitter")
    arguments = vars(parser.parse_args())
    run_directory = arguments.pop("run_directory")
    prospar = arguments.pop("prospar_config")
    arguments["prospar"] = ProSparConfig(**prospar) if prospar is not None else None
    run_heaton(HeatonConfig(**arguments), run_directory=run_directory)
