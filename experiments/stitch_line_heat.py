"""Train GPW on periodic heat with independently trainable stitch locations."""

from __future__ import annotations

import gc
import json
import os
import platform
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skfem
import torch
from torch import nn

from experiments.periodic_heat_preview import simulate
from experiments.stitch_line_preview import LayoutConfig, layout
from experiments.two_phase_sine import _file_digest, _git
from pwgp import GeneralizedPatchworkGP, fit
from pwgp.config import HyperparameterBounds, NumericalConfig, OptimizerConfig
from pwgp.training import predict_batched


class StraightLineParameter(nn.Module):
    """Angle and normal offset; stitches span the box intersection at fixed fractions."""

    def __init__(self, center, angle, count, margin, *, dtype=torch.float64):
        super().__init__()
        if count < 2 or not 0 < margin < 0.5:
            raise ValueError("Require at least two stitches and margin in (0,0.5)")
        angle = torch.as_tensor(np.deg2rad(angle), dtype=dtype)
        center = torch.as_tensor(center, dtype=dtype)
        if center.shape != (2,) or not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(angle)):
            raise ValueError("Line center must have two finite coordinates and angle must be finite")
        normal = torch.stack((angle.sin(), -angle.cos()))
        radius = 0.5 * normal.abs().sum()
        fraction = 0.5 * (1 + ((center - 0.5) * normal).sum() / radius)
        if not bool((fraction > 0) & (fraction < 1)):
            raise ValueError("Initial line must cross the interior of the unit square")
        self.raw = nn.Parameter(torch.stack((angle, torch.logit(fraction))))
        self.register_buffer("fractions", torch.linspace(margin, 1 - margin, count, dtype=dtype))
        self.trainable = True

    def geometry(self):
        angle, offset_raw = self.raw.unbind()
        tangent = torch.stack((angle.cos(), angle.sin()))
        normal = torch.stack((tangent[1], -tangent[0]))
        radius = 0.5 * normal.abs().sum()
        offset = (2 * offset_raw.sigmoid() - 1) * radius
        origin = 0.5 + offset * normal
        # Slab intersection: parallel coordinates impose no bound on distance along the line.
        parallel = tangent == 0
        denominator = torch.where(parallel, torch.ones_like(tangent), tangent)
        a, b = -origin / denominator, (1 - origin) / denominator
        lower = torch.where(parallel, -torch.inf, torch.minimum(a, b)).max()
        upper = torch.where(parallel, torch.inf, torch.maximum(a, b)).min()
        if not bool(torch.isfinite(self.raw).all()) or not bool(upper > lower):
            raise FloatingPointError("Stitch line has non-finite geometry or no interior segment")
        return origin, tangent, lower, upper

    def forward(self):
        origin, tangent, lower, upper = self.geometry()
        points = origin + (lower + self.fractions * (upper - lower))[:, None] * tangent
        if not bool(torch.isfinite(points).all()) or bool(((points <= 0) | (points >= 1)).any()):
            raise FloatingPointError("Line stitches must be finite and strictly inside the unit square")
        return points[None]


@dataclass(frozen=True)
class RunConfig:
    layouts: LayoutConfig = field(default_factory=LayoutConfig)
    observations: int = 500
    epochs: int = 350
    warmup_epochs: int = 40
    learning_rate_q: float = 0.008
    learning_rate_supports: float = 0.004
    learning_rate_line: float = 0.003
    learning_rate_hyperparameters: float = 0.003
    snapshot_every: int = 50
    prediction_batch_size: int = 256
    threads: int = 2
    device: str = "cpu"


def build_model(config, index):
    h = config.layouts.heat
    supports, stitches, _, _ = layout(config.layouts, index)
    model = GeneralizedPatchworkGP(torch.from_numpy(supports), torch.from_numpy(stitches[None]),
        [(0, 1)], q_rank=h.q_rank, lengthscale=h.lengthscale, outputscale=h.outputscale,
        noise_std=h.noise_std, numerical=NumericalConfig(weight_epsilon=h.weight_epsilon))
    torch.testing.assert_close(model.stitches(), torch.from_numpy(stitches[None]), rtol=0, atol=1e-14)
    return model.to(config.device)


def draw_field(ax, xy, values, title, config, *, vmin=None, vmax=None, cmap="inferno"):
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    image = ax.imshow(np.asarray(values)[order].reshape(config.layouts.heat.cells + 1, -1),
        origin="lower", extent=(0, 1, 0, 1), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axvline(0.5, color="white", linestyle="--", linewidth=0.8)
    ax.set(title=title, xlabel="x", ylabel="y", aspect="equal", xlim=(0, 1), ylim=(0, 1))
    return image


def draw_points(ax, supports, stitches):
    for patch, color in enumerate(("cyan", "lime")):
        ax.scatter(*supports[patch].T, c=color, edgecolors="black", linewidths=0.4, s=18,
                   label=f"Model {patch + 1}")
    ax.scatter(*stitches[0].T, c="deepskyblue", marker="x", label="Stitches")


@torch.no_grad()
def kernel_snapshot(model, epoch):
    """Record transformed kernel parameters in model units, without retaining graphs."""
    return {"epoch": epoch, "lengthscales": model.hyp.lengthscales().cpu().tolist(),
            "outputscales": model.hyp.outputscales().cpu().tolist(),
            "noise_std": float(model.hyp.noise_std().cpu())}


def draw_kernel_history(axes, records, warmup_epochs):
    epochs = [row["epoch"] for row in records]
    lengthscales = np.asarray([row["lengthscales"] for row in records])
    outputscales = np.asarray([row["outputscales"] for row in records])
    for patch in range(lengthscales.shape[1]):
        label = f"Kernel {patch + 1}"
        axes[0].plot(epochs, lengthscales[:, patch, 0], label=label)
        axes[1].plot(epochs, lengthscales[:, patch, 1], label=label)
        axes[2].plot(epochs, outputscales[:, patch], label=label)
    axes[3].plot(epochs, [row["noise_std"] for row in records], label="Shared noise", color="black")
    titles = ("ARD lengthscale x", "ARD lengthscale y", "Kernel output variance", "Observation noise SD")
    for ax, title in zip(axes, titles, strict=True):
        ax.set(title=title, xlabel="Epoch", xlim=(0, max(1, epochs[-1])))
        if warmup_epochs:
            ax.axvline(warmup_epochs, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(fontsize=7)


def run(config: RunConfig, *, directory=None, capture_git=True, on_snapshot=None, on_complete=None):
    if not 0 <= config.warmup_epochs < config.epochs or config.snapshot_every < 1:
        raise ValueError("Require 0 <= warmup_epochs < epochs and positive snapshot_every")
    if config.threads < 1 or config.prediction_batch_size < 1:
        raise ValueError("Require positive threads and prediction_batch_size")
    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must explicitly select cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device is unavailable; no CPU fallback")
    output = Path(directory) if directory else Path("results/stitch_line_heat") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)

    def write(path, value):
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")

    optimizer = OptimizerConfig(learning_rate_q=config.learning_rate_q,
        learning_rate_supports=config.learning_rate_supports, learning_rate_stitches=config.learning_rate_line,
        learning_rate_hyperparameters=config.learning_rate_hyperparameters,
        warmup_fraction=config.warmup_epochs / config.epochs)
    h = config.layouts.heat
    write(output / "config.json", {**asdict(config), "optimizer": asdict(optimizer),
        "numerical": asdict(NumericalConfig(weight_epsilon=h.weight_epsilon)),
        "bounds": asdict(HyperparameterBounds()), "dtype": "float64",
        "stitch_parameterization": "independent UnitCubeParameter coordinates; lines initialize only",
        "supports_constrained_to_initial_side_during_training": False})
    write(output / "status.json", {"state": "running"})
    try:
        torch.set_num_threads(config.threads)
        xy, initial_heat, temperature, tensors = simulate(h, (0, 90))
        if not 1 < config.observations < len(xy):
            raise ValueError("Require at least two training observations and a nonempty held-out set")
        rng = np.random.default_rng(h.seed)
        train = rng.choice(len(xy), config.observations, replace=False)
        test = np.setdiff1d(np.arange(len(xy)), train)
        center, scale = float(temperature[train].mean()), float(temperature[train].std())
        if scale <= 0:
            raise ValueError("Training temperature standard deviation must be positive")
        truth = (temperature - center) / scale
        train_y = truth[train] + rng.normal(scale=h.noise_std, size=len(train))
        test_y = truth[test] + rng.normal(scale=h.noise_std, size=len(test))
        np.savez_compressed(output / "data.npz", xy=xy, temperature=temperature, initial_heat=initial_heat,
                            tensors=tensors, train=train, test=test, train_y=train_y, test_y=test_y)
        write(output / "provenance.json", {"model": "pwgp.GeneralizedPatchworkGP", "stitch_class": "UnitCubeParameter",
            "package_version": version("generalized-patchwork-gp"), "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_fem": skfem.__version__,
            "git_head": _git(["rev-parse", "HEAD"]) if capture_git else None,
            "git_status": _git(["status", "--short"]) if capture_git else None,
            "capture_git": capture_git, "device": str(device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "target_mean": center, "target_std": scale,
            "dataset_sha256": _file_digest(output / "data.npz"),
            "source_sha256": {str(p): _file_digest(p) for p in [Path(__file__),
                Path("experiments/periodic_heat_preview.py"), Path("experiments/stitch_line_preview.py"),
                *Path("src/pwgp").rglob("*.py")]},
            "objective": "posterior-adaptive expected log likelihood minus KL; no fixed-model ELBO guarantee"})
        cases = len(config.layouts.line_angles)
        point_fig, point_axes = plt.subplots(cases, 2, figsize=(9, 4 * cases), squeeze=False, constrained_layout=True)
        fit_fig, fit_axes = plt.subplots(cases, 4, figsize=(16, 3.7 * cases), squeeze=False, constrained_layout=True)
        loss_fig, loss_ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        rows = []
        for index, name in enumerate(config.layouts.line_names):
            folder = output / f"layout_{index}"
            folder.mkdir()
            write(folder / "status.json", {"state": "running", "layout": name})
            torch.manual_seed(h.seed)
            model = build_model(config, index)
            initial_supports = model.supports().detach().cpu().numpy().copy()
            initial_stitches = model.stitches().detach().cpu().numpy().copy()
            torch.save(model.state_dict(), folder / "initial_model.pt")
            trajectory = []
            kernel_history = [kernel_snapshot(model, 0)]
            write(folder / "kernel_history.json", kernel_history)

            def snapshot(epoch, current, history, folder=folder, trajectory=trajectory, name=name,
                         kernel_history=kernel_history):
                trajectory.append({"epoch": epoch, "stitches": current.stitches().detach().cpu().tolist()})
                kernel_history.append(kernel_snapshot(current, epoch))
                if epoch % config.snapshot_every == 0 or epoch == config.epochs:
                    write(folder / "history.json", asdict(history))
                    write(folder / "stitch_trajectory.json", trajectory)
                    write(folder / "kernel_history.json", kernel_history)
                    torch.save(current.state_dict(), folder / "checkpoint.pt")
                    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
                    panels = axes.ravel()
                    draw_field(panels[0], xy, temperature, f"{name}: epoch {epoch}", config)
                    draw_points(panels[0], current.supports().detach().cpu().numpy(), current.stitches().detach().cpu().numpy())
                    panels[1].plot(np.arange(1, len(history.losses) + 1), history.losses)
                    panels[1].set(xlabel="Epoch", ylabel="Negative objective / observation")
                    draw_kernel_history(panels[2:], kernel_history, config.warmup_epochs)
                    fig.savefig(folder / "progress.png", dpi=130)
                    plt.close(fig)
                    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
                    draw_kernel_history(axes.ravel(), kernel_history, config.warmup_epochs)
                    fig.suptitle(f"{name}: kernel evolution (standardized targets)")
                    fig.savefig(folder / "kernel_evolution.png", dpi=150)
                    plt.close(fig)
                    if on_snapshot is not None:
                        on_snapshot(folder, epoch)

            history = fit(model, torch.from_numpy(xy[train]), torch.from_numpy(train_y),
                epochs=config.epochs, batch_size=len(train), seed=h.seed, optimizer=optimizer, on_epoch=snapshot)
            mean, variance = predict_batched(model, torch.from_numpy(xy), batch_size=config.prediction_batch_size,
                                             observation_noise=False)
            mean, variance = mean.numpy(), variance.numpy()
            noise = float(model.hyp.noise_std().detach())
            errors = test_y - mean[test]
            observation_variance = variance[test] + noise**2
            supports = model.supports().detach().cpu().numpy()
            stitches = model.stitches().detach().cpu().numpy()
            metrics = {"layout": name, "rmse_standardized": float(np.sqrt(np.mean((mean[test] - truth[test])**2))),
                "rmse_temperature": float(scale * np.sqrt(np.mean((mean[test] - truth[test])**2))),
                "nlpd_standardized": float(np.mean(0.5 * (np.log(2 * np.pi * observation_variance) + errors**2 / observation_variance))),
                "coverage_95": float(np.mean(np.abs(errors) <= 1.959963984540054 * np.sqrt(observation_variance))),
                "initial_line_angle_degrees": config.layouts.line_angles[index],
                "lengthscales": model.hyp.lengthscales().detach().tolist(),
                "outputscales": model.hyp.outputscales().detach().tolist(), "noise_std": noise,
                "final_loss": history.losses[-1], "training_seconds": history.elapsed_seconds,
                "peak_cuda_megabytes": history.peak_cuda_megabytes,
                "mean_stitch_displacement": float(np.linalg.norm(stitches - initial_stitches, axis=-1).mean()),
                "mean_support_displacement": float(np.linalg.norm(supports - initial_supports, axis=-1).mean())}
            rows.append(metrics)
            torch.save(model.state_dict(), folder / "model.pt")
            np.savez_compressed(folder / "predictions.npz", initial_supports=initial_supports,
                final_supports=supports, initial_stitches=initial_stitches, final_stitches=stitches,
                mean=mean, latent_variance=variance)
            write(folder / "metrics.json", metrics)
            write(folder / "status.json", {"state": "complete"})
            for column, (points, line) in enumerate(((initial_supports, initial_stitches), (supports, stitches))):
                ax = point_axes[index, column]
                draw_field(ax, xy, temperature, f"{name}: {'initial' if column == 0 else 'final'}", config)
                draw_points(ax, points, line)
                if column:
                    ax.plot(*initial_stitches[0].T, c="gray", ls="--", lw=1, label="Initial line")
                ax.legend(fontsize=7, loc="lower right")
            prediction = mean * scale + center
            common_min, common_max = min(temperature.min(), prediction.min()), max(temperature.max(), prediction.max())
            for column, (values, title) in enumerate(((temperature, f"Truth: {name}"),
                    (prediction, "GPW mean"), (np.abs(prediction - temperature), "Absolute error"),
                    (np.sqrt(variance) * scale, "Latent SD (full R)"))):
                im = draw_field(fit_axes[index, column], xy, values, title, config,
                    vmin=common_min if column < 2 else 0, vmax=common_max if column < 2 else None)
                fit_fig.colorbar(im, ax=fit_axes[index, column], shrink=0.7)
            loss_ax.plot(np.arange(1, len(history.losses) + 1), history.losses, label=name)
            loss_ax.set(xlabel="Epoch", ylabel="Negative objective / observation", yscale="log")
            loss_ax.legend()
            write(output / "metrics.json", rows)
            point_fig.savefig(output / "points.png", dpi=150, bbox_inches="tight")
            fit_fig.savefig(output / "fits.png", dpi=140, bbox_inches="tight")
            loss_fig.savefig(output / "training.png", dpi=150)
            print(json.dumps(metrics), flush=True)
            del snapshot, model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            if on_complete is not None:
                on_complete(folder, metrics)
        plt.close(point_fig)
        plt.close(fit_fig)
        plt.close(loss_fig)
        write(output / "status.json", {"state": "complete"})
    finally:
        error = sys.exception()
        if error is not None:
            write(output / "status.json", {"state": "failed", "error": str(error), "traceback": traceback.format_exc()})
    print("Results:", output.resolve())
    return output


if __name__ == "__main__":
    run(RunConfig())
