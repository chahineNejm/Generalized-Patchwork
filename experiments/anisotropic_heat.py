"""Two joined anisotropic plates: sensitivity to support-cloud orientation."""

from __future__ import annotations

import gc
import json
import platform
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply

from experiments.two_phase_sine import _file_digest, _git
from pwgp import GeneralizedPatchworkGP, fit
from pwgp.config import HyperparameterBounds, NumericalConfig, OptimizerConfig
from pwgp.training import predict_batched


@dataclass(frozen=True)
class HeatConfig:
    side: int = 60
    observations: int = 500
    diffusivity: tuple[float, float] = (0.6, 0.025)
    time: float = 0.015
    pulse_width: float = 0.035
    centers: tuple[tuple[float, float], ...] = ((0.25, 0.5), (0.75, 0.5))
    orientations: tuple[tuple[float, float], ...] = ((0, 90), (90, 0), (45, 135), (0, 0))
    cloud_axes: tuple[float, float] = (0.19, 0.065)
    supports_per_patch: int = 28
    stitches: int = 10
    q_rank: int = 12
    epochs: int = 350
    warmup_epochs: int = 40
    noise_std: float = 0.03
    lengthscale: float = 0.12
    outputscale: float = 1.0
    learning_rate_q: float = 0.008
    learning_rate_supports: float = 0.004
    learning_rate_hyperparameters: float = 0.003
    weight_epsilon: float = 0.01
    snapshot_every: int = 50
    seed: int = 31
    threads: int = 2


def heat_operator(side, diffusivity):
    """Conservative cell-centered fluxes; harmonic interface and insulated edges."""
    if side < 4 or side % 2:
        raise ValueError("side must be even and at least four, with a face at x=0.5")
    if len(diffusivity) != 2 or not np.isfinite(diffusivity).all() or min(diffusivity) <= 0:
        raise ValueError("diffusivity must contain two finite positive diagonal coefficients")
    axis = (np.arange(side) + 0.5) / side
    x, y = np.meshgrid(axis, axis, indexing="ij")
    coordinates = np.column_stack((x.ravel(), y.ravel()))
    conductivity = np.empty((side, side, 2))
    conductivity[:side // 2] = diffusivity
    conductivity[side // 2:] = diffusivity[::-1]
    indices = np.arange(side**2).reshape(side, side)
    rows, cols, values = [], [], []
    for direction in range(2):
        a, b = ((indices[:-1], indices[1:]) if direction == 0
                else (indices[:, :-1], indices[:, 1:]))
        k = conductivity[..., direction]
        ka, kb = ((k[:-1], k[1:]) if direction == 0 else (k[:, :-1], k[:, 1:]))
        conductance = (2 * ka * kb / (ka + kb) * side**2).ravel()
        a, b = a.ravel(), b.ravel()
        rows.extend((a, b, a, b))
        cols.extend((b, a, a, b))
        values.extend((conductance, conductance, -conductance, -conductance))
    operator = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(side**2, side**2)).tocsr()
    return coordinates, operator


def simulate(config):
    """Solve dT/dt=div(K grad T), with equal unit heat capacities and no source."""
    coordinates, operator = heat_operator(config.side, config.diffusivity)
    if config.time <= 0 or config.pulse_width <= 0:
        raise ValueError("time and pulse_width must be positive")
    initial = sum(np.exp(-((coordinates - center)**2).sum(1) / (2 * config.pulse_width**2))
                  for center in config.centers)
    temperature = expm_multiply(config.time * operator, initial)
    np.testing.assert_allclose(temperature.sum(), initial.sum(), rtol=1e-10)
    if not np.isfinite(temperature).all() or temperature.min() < -1e-12:
        raise FloatingPointError("Heat solution must be finite and nonnegative")
    return coordinates, initial, temperature


def support_clouds(config, angles):
    """Rotate the same deterministic filled ellipse independently in each plate."""
    if config.supports_per_patch < 3 or len(angles) != 2:
        raise ValueError("Require at least three supports and two orientation angles")
    i = np.arange(config.supports_per_patch)
    radius, theta = np.sqrt((i + 0.5) / len(i)), i * np.pi * (3 - np.sqrt(5))
    base = np.column_stack((radius * np.cos(theta), radius * np.sin(theta))) * config.cloud_axes
    clouds = []
    for center, angle in zip(config.centers, angles, strict=True):
        a = np.deg2rad(angle)
        rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        clouds.append(base @ rotation.T + center)
    clouds = np.stack(clouds)
    if not np.isfinite(clouds).all() or (clouds <= 0).any() or (clouds >= 1).any():
        raise ValueError("Rotated support clouds must lie strictly inside [0,1]^2")
    return torch.from_numpy(clouds)


def field(ax, values, config, title, *, vmax=None, cmap="inferno"):
    image = ax.imshow(np.asarray(values).reshape(config.side, config.side).T, origin="lower",
                      extent=(0, 1, 0, 1), cmap=cmap, vmin=0, vmax=vmax)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=0.8)
    ax.set(title=title, xlabel="x", ylabel="y", aspect="equal")
    return image


def run(config: HeatConfig, *, directory=None):
    if not 0 <= config.warmup_epochs < config.epochs or config.snapshot_every < 1:
        raise ValueError("Require 0 <= warmup_epochs < epochs and positive snapshot_every")
    if len(config.centers) != 2 or not config.orientations or config.stitches < 1:
        raise ValueError("Require exactly two centers, at least one orientation and positive stitches")
    if config.threads < 1 or not 1 <= config.observations < config.side**2:
        raise ValueError("Require positive threads and a nonempty held-out grid")
    output = Path(directory) if directory else Path("results/anisotropic_heat") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    def write(name, data):
        (output / name).write_text(json.dumps(data, indent=2), encoding="utf-8")
    numerical = NumericalConfig(weight_epsilon=config.weight_epsilon)
    optimizer = OptimizerConfig(learning_rate_q=config.learning_rate_q,
        learning_rate_supports=config.learning_rate_supports,
        learning_rate_hyperparameters=config.learning_rate_hyperparameters,
        warmup_fraction=config.warmup_epochs / config.epochs)
    write("config.json", {**asdict(config), "numerical": asdict(numerical),
                         "optimizer": asdict(optimizer), "bounds": asdict(HyperparameterBounds()),
                         "train_supports": True, "train_stitches": False,
                         "train_hyperparameters": True, "device": "cpu", "dtype": "float64"})
    write("status.json", {"state": "running"})
    try:
        torch.set_num_threads(config.threads)
        xy, initial_heat, temperature = simulate(config)
        rng = np.random.default_rng(config.seed)
        train = rng.choice(len(xy), config.observations, replace=False)
        test = np.setdiff1d(np.arange(len(xy)), train)
        center, scale = float(temperature[train].mean()), float(temperature[train].std())
        if scale <= 0:
            raise ValueError("Training target standard deviation must be positive")
        truth = (temperature - center) / scale
        observations = truth[train] + config.noise_std * rng.normal(size=len(train))
        test_y = truth[test] + rng.normal(scale=config.noise_std, size=len(test))
        np.savez_compressed(output / "data.npz", xy=xy, initial_heat=initial_heat, temperature=temperature,
                            train=train, test=test, train_y=observations, test_y=test_y)
        write("provenance.json", {"model": "pwgp.GeneralizedPatchworkGP", "package_version": version("generalized-patchwork-gp"),
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "scipy": scipy.__version__, "numpy": np.__version__, "device": "cpu",
            "git_head": _git(["rev-parse", "HEAD"]), "git_status": _git(["status", "--short"]),
            "source_hashes": {str(p): _file_digest(p) for p in [Path(__file__), *Path("src/pwgp").rglob("*.py")]},
            "dataset_sha256": _file_digest(output / "data.npz"), "target_mean": center, "target_std": scale,
            "physics": "dimensionless equal heat capacity; insulated edges; perfect thermal contact",
            "solver": "conservative finite volumes, harmonic face diffusivity, scipy expm_multiply",
            "objective": "posterior-adaptive expected Gaussian likelihood minus KL, not a fixed-model ELBO"})
        x, y = torch.from_numpy(xy[train]), torch.from_numpy(observations)
        stitches = torch.stack((torch.full((config.stitches,), 0.5, dtype=torch.float64),
                                torch.linspace(0.05, 0.95, config.stitches, dtype=torch.float64)), -1)[None]
        point_fig, point_axes = plt.subplots(len(config.orientations), 2, figsize=(9, 4 * len(config.orientations)), squeeze=False, constrained_layout=True)
        fit_fig, fit_axes = plt.subplots(len(config.orientations), 4, figsize=(16, 3.8 * len(config.orientations)), squeeze=False, constrained_layout=True)
        rows = []
        for index, angles in enumerate(config.orientations):
            torch.manual_seed(config.seed)
            folder = output / f"orientation_{index}"
            folder.mkdir()
            supports = support_clouds(config, angles)
            model = GeneralizedPatchworkGP(supports, stitches, [(0, 1)], q_rank=config.q_rank,
                train_supports=True, train_stitches=False, train_hyperparameters=True,
                lengthscale=config.lengthscale, outputscale=config.outputscale, noise_std=config.noise_std,
                numerical=numerical)
            def snapshot(epoch, current, history, folder=folder, angles=angles):
                if epoch % config.snapshot_every == 0 or epoch == config.epochs:
                    torch.save(current.state_dict(), folder / "checkpoint.pt")
                    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
                    field(ax, temperature, config, f"{angles}: epoch {epoch}")
                    for patch, color in enumerate(("cyan", "lime")):
                        ax.scatter(*current.supports()[patch].detach().numpy().T, c=color, s=12)
                    fig.savefig(folder / "progress.png", dpi=130)
                    plt.close(fig)
            history = fit(model, x, y, epochs=config.epochs, batch_size=len(x), seed=config.seed,
                          optimizer=optimizer, on_epoch=snapshot)
            mean, variance = predict_batched(model, torch.from_numpy(xy), batch_size=256, observation_noise=False)
            end = model.supports().detach().numpy()
            error = truth[test] - mean.numpy()[test]
            obs_variance = variance.numpy()[test] + float(model.hyp.noise_std().detach())**2
            noisy_error = test_y - mean.numpy()[test]
            metrics = {"angles_degrees": angles, "heldout_rmse_standardized": float(np.sqrt(np.mean(error**2))),
                "heldout_rmse_temperature": float(scale * np.sqrt(np.mean(error**2))),
                "nlpd_standardized": float(np.mean(0.5 * (np.log(2 * np.pi * obs_variance) + noisy_error**2 / obs_variance))),
                "coverage_95": float(np.mean(np.abs(noisy_error) <= 1.96 * np.sqrt(obs_variance))),
                "lengthscales": model.hyp.lengthscales().detach().tolist(),
                "outputscales": model.hyp.outputscales().detach().tolist(),
                "noise_std": float(model.hyp.noise_std().detach()),
                "mean_support_displacement": float(np.linalg.norm(end - supports.numpy(), axis=-1).mean()),
                "training_seconds": history.elapsed_seconds, "final_loss": history.losses[-1]}
            rows.append(metrics)
            torch.save(model.state_dict(), folder / "model.pt")
            (folder / "history.json").write_text(json.dumps(asdict(history), indent=2))
            (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
            np.savez_compressed(folder / "predictions.npz", initial_supports=supports.numpy(), final_supports=end,
                                stitches=stitches.numpy(), mean=mean.numpy(), latent_variance=variance.numpy())
            for column, points in enumerate((supports.numpy(), end)):
                ax = point_axes[index, column]
                field(ax, temperature, config, f"{angles} deg | {'Initial' if column == 0 else 'Final'} supports")
                for patch, color in enumerate(("cyan", "lime")):
                    ax.scatter(*points[patch].T, c=color, edgecolors="black", s=22, linewidths=0.4, label=f"Model {patch + 1}")
                    if column:
                        delta = end[patch] - supports.numpy()[patch]
                        ax.quiver(*supports.numpy()[patch].T, *delta.T, angles="xy", scale_units="xy", scale=1, color=color, alpha=0.5, width=0.003)
                ax.scatter(*stitches[0].numpy().T, c="white", marker="x", s=20)
                ax.legend(fontsize=7, loc="lower right")
            predicted = mean.numpy() * scale + center
            for ax, values, title, vmax in zip(fit_axes[index],
                    (temperature, predicted, np.abs(predicted - temperature), variance.sqrt().numpy() * scale),
                    (f"Truth | start {angles}", "GPW mean", "Absolute error", "Latent SD (full R)"),
                    (temperature.max(), temperature.max(), None, None), strict=True):
                im = field(ax, values, config, title, vmax=vmax)
                fit_fig.colorbar(im, ax=ax, shrink=0.7)
            write("metrics.json", rows)
            point_fig.savefig(output / "points.png", dpi=150)
            fit_fig.savefig(output / "fits.png", dpi=140)
            print(json.dumps(metrics), flush=True)
            del snapshot, model
            gc.collect()
        plt.close(point_fig)
        plt.close(fit_fig)
        write("status.json", {"state": "complete"})
    finally:
        error = sys.exception()
        if error is not None:
            write("status.json", {"state": "failed", "error": str(error), "traceback": traceback.format_exc()})
    print("Results:", output.resolve())
    return output


if __name__ == "__main__":
    run(HeatConfig())
