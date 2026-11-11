"""Preview periodic heat, global random supports and an offset movable stitch line."""

from __future__ import annotations

import json
import os
import platform
import sys
import traceback
from dataclasses import asdict, dataclass
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
from scipy.sparse.linalg import factorized
from skfem import Basis, BilinearForm, ElementTriP1, MeshTri, asm
from skfem.models.poisson import mass

from experiments.two_phase_sine import _file_digest, _git
from pwgp import GeneralizedPatchworkGP
from pwgp.config import NumericalConfig


@dataclass(frozen=True)
class PreviewConfig:
    cells: int = 80
    diffusivity: tuple[float, float] = (0.6, 0.025)
    angle_pairs: tuple[tuple[float, float], ...] = ((90, 90), (45, 45), (0, 90))
    periods: tuple[int, int] = (4, 4)
    baseline: float = 1.0
    amplitude: float = 0.8
    time: float = 0.003
    time_steps: int = 40
    supports_per_patch: int = 28
    support_margin: float = 0.02
    stitches: int = 10
    initial_stitch_x: float = 0.56
    stitch_y_margin: float = 0.05
    q_rank: int = 12
    lengthscale: float = 0.12
    outputscale: float = 1.0
    noise_std: float = 0.03
    weight_epsilon: float = 0.01
    seed: int = 31

    def validate(self):
        if self.cells < 4 or self.cells % 2:
            raise ValueError("cells must be even and at least four for an interface at x=0.5")
        if self.time <= 0 or self.time_steps < 1:
            raise ValueError("Require positive time and time_steps")
        if len(self.diffusivity) != 2 or not np.isfinite(self.diffusivity).all() or min(self.diffusivity) <= 0:
            raise ValueError("diffusivity must contain two finite positive principal coefficients")
        if not self.angle_pairs or any(len(a) != 2 or not np.isfinite(a).all() for a in self.angle_pairs):
            raise ValueError("angle_pairs must contain finite pairs of material angles")
        if len(self.periods) != 2 or any(not isinstance(p, int) or p <= 0 for p in self.periods):
            raise ValueError("periods must contain two positive integers")
        if not np.isfinite([self.baseline, self.amplitude, self.time]).all() or not 0 < self.amplitude < self.baseline:
            raise ValueError("Require finite baseline > amplitude > 0 and finite time")
        if self.supports_per_patch < 3 or not 0 < self.support_margin < 0.5:
            raise ValueError("Require at least three supports and support_margin inside (0,0.5)")
        if self.stitches < 2 or not 0 < self.initial_stitch_x < 1 or not 0 < self.stitch_y_margin < 0.5:
            raise ValueError("Require at least two stitches and strictly interior initial stitch coordinates")
        if not 0 <= self.q_rank <= 2 * self.supports_per_patch:
            raise ValueError("q_rank must lie between zero and the total support count")


def tensor_at_angle(diffusivity, angle):
    a = np.deg2rad(angle)
    rotation = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return rotation @ np.diag(diffusivity) @ rotation.T


def assemble_heat(config, angles):
    """P1 FEM with full rotated tensors and natural zero-flux exterior boundaries."""
    config.validate()
    axis = np.linspace(0, 1, config.cells + 1)
    mesh = MeshTri.init_tensor(axis, axis)
    basis = Basis(mesh, ElementTriP1())
    tensors = np.stack([tensor_at_angle(config.diffusivity, a) for a in angles])

    @BilinearForm
    def diffusion(u, v, w):
        right = w.x[0] >= 0.5
        xx = np.where(right, tensors[1, 0, 0], tensors[0, 0, 0])
        xy = np.where(right, tensors[1, 0, 1], tensors[0, 0, 1])
        yy = np.where(right, tensors[1, 1, 1], tensors[0, 1, 1])
        return (xx * u.grad[0] * v.grad[0] + yy * u.grad[1] * v.grad[1]
                + xy * (u.grad[0] * v.grad[1] + u.grad[1] * v.grad[0]))

    return mesh.p.T, asm(mass, basis), asm(diffusion, basis), tensors


def simulate(config, angles):
    """Repeated spatial temperature, then implicit-Euler anisotropic diffusion."""
    xy, capacity, stiffness, tensors = assemble_heat(config, angles)
    phase = 2 * np.pi * xy * config.periods
    initial = config.baseline + 0.5 * config.amplitude * (np.cos(phase[:, 0]) + np.cos(phase[:, 1]))
    step = config.time / config.time_steps
    solve = factorized((capacity + step * stiffness).tocsc())
    temperature = initial.copy()
    for _ in range(config.time_steps):
        temperature = solve(capacity @ temperature)
    volume = np.asarray(capacity.sum(axis=0)).ravel()
    np.testing.assert_allclose(volume @ temperature, volume @ initial, rtol=1e-10)
    if not np.isfinite(temperature).all():
        raise FloatingPointError("Heat solution contains non-finite temperatures")
    return xy, initial, temperature, tensors


def initial_model(config):
    """Independent uniform locations over the entire domain for both submodels."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    supports = torch.from_numpy(rng.uniform(config.support_margin, 1 - config.support_margin,
                                          (2, config.supports_per_patch, 2)))
    stitches = torch.stack((torch.full((config.stitches,), config.initial_stitch_x, dtype=torch.float64),
                           torch.linspace(config.stitch_y_margin, 1 - config.stitch_y_margin,
                                          config.stitches, dtype=torch.float64)), -1)[None]
    torch.manual_seed(config.seed)
    return GeneralizedPatchworkGP(supports, stitches, [(0, 1)], q_rank=config.q_rank,
        train_supports=True, train_stitches=True, train_hyperparameters=True,
        lengthscale=config.lengthscale, outputscale=config.outputscale, noise_std=config.noise_std,
        numerical=NumericalConfig(weight_epsilon=config.weight_epsilon))


def preview(config: PreviewConfig, *, directory=None):
    config.validate()
    output = Path(directory) if directory else Path("results/periodic_heat_preview") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)

    def write(name, data):
        (output / name).write_text(json.dumps(data, indent=2), encoding="utf-8")

    write("config.json", asdict(config))
    write("status.json", {"state": "preparing", "training_epochs": 0})
    try:
        model = initial_model(config)
        supports = model.supports().detach().numpy()
        stitches = model.stitches().detach().numpy()
        torch.save(model.state_dict(), output / "initial_model.pt")
        fig, axes = plt.subplots(1, len(config.angle_pairs), figsize=(6 * len(config.angle_pairs), 5.8),
                                 squeeze=False, constrained_layout=True)
        for i, angles in enumerate(config.angle_pairs):
            xy, initial, temperature, tensors = simulate(config, angles)
            np.savez_compressed(output / f"case_{i}.npz", xy=xy, initial_temperature=initial,
                temperature=temperature, conductivity=tensors, initial_supports=supports, initial_stitches=stitches)
            ax = axes[0, i]
            order = np.lexsort((xy[:, 0], xy[:, 1]))
            image = ax.imshow(temperature[order].reshape(config.cells + 1, config.cells + 1),
                origin="lower", extent=(0, 1, 0, 1), cmap="inferno",
                vmin=config.baseline - config.amplitude, vmax=config.baseline + config.amplitude)
            for patch, color in enumerate(("cyan", "lime")):
                ax.scatter(*supports[patch].T, c=color, s=28, edgecolors="black", linewidths=0.5,
                           label=f"Submodel {patch + 1}: initial supports")
            ax.axvline(0.5, color="white", linestyle="--", linewidth=1, label="True interface x=0.50")
            ax.plot(*stitches[0].T, color="deepskyblue", linestyle=":", marker="x", markersize=7,
                    linewidth=1.2, label=f"Movable stitches x={config.initial_stitch_x:.2f}")
            for patch, angle in enumerate(angles):
                center = np.array([0.25 + patch * 0.5, 0.91])
                delta = 0.07 * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                ax.annotate("", xy=center + delta, xytext=center - delta,
                            arrowprops={"arrowstyle": "<->", "color": "white", "lw": 2})
            ax.set(title=f"Anisotropy {angles[0]} / {angles[1]} degrees", xlabel="x", ylabel="y",
                   xlim=(0, 1), ylim=(0, 1), aspect="equal")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=8)
        fig.colorbar(image, ax=list(axes.flat), shrink=0.7, label="Dimensionless temperature")
        fig.savefig(output / "initial_setup.png", dpi=160)
        plt.close(fig)
        artifacts = [p for p in output.iterdir() if p.is_file() and p.name not in ("status.json", "provenance.json")]
        write("provenance.json", {"model": "pwgp.GeneralizedPatchworkGP",
            "package_version": version("generalized-patchwork-gp"), "python": sys.version,
            "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_fem": skfem.__version__, "device": "cpu", "dtype": "float64",
            "git_head": _git(["rev-parse", "HEAD"]), "git_status": _git(["status", "--short"]),
            "source_hashes": {str(p): _file_digest(p) for p in [Path(__file__), *Path("src/pwgp").rglob("*.py")]},
            "artifact_sha256": {p.name: _file_digest(p) for p in artifacts},
            "train_supports": True, "train_stitches": True, "train_hyperparameters": True,
            "physics": "equal unit heat capacity, perfect contact, insulated exterior; spatially repeating initial temperature",
            "solver": "scikit-fem P1 triangles, consistent mass, backward Euler; full rotated conductivity tensors",
            "training_epochs": 0, "scores": None, "target_scaling": "none: preview only"})
        write("status.json", {"state": "preview_complete", "training_epochs": 0})
    finally:
        error = sys.exception()
        if error is not None:
            write("status.json", {"state": "failed", "error": str(error), "traceback": traceback.format_exc()})
    print("Preview (no training):", output.resolve())
    return output


if __name__ == "__main__":
    preview(PreviewConfig())
