"""Fixed periodic heat, disjoint initial support regions, and four stitch lines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import skfem

from experiments.periodic_heat_preview import PreviewConfig, simulate
from experiments.two_phase_sine import _file_digest


@dataclass(frozen=True)
class LayoutConfig:
    heat: PreviewConfig = field(default_factory=lambda: PreviewConfig(angle_pairs=((0, 90),), time=0.015))
    line_centers: tuple[tuple[float, float], ...] = ((0.5, 0.5), (0.56, 0.5), (0.5, 0.5), (0.5, 0.5))
    line_angles: tuple[float, ...] = (90, 90, 45, 0)
    line_names: tuple[str, ...] = ("Vertical", "Vertical, shifted", "Diagonal", "Horizontal")
    candidate_points: int = 10000
    line_margin: float = 0.05


def layout(config, index):
    """Sample independently of heat, strictly on opposite sides of the chosen line."""
    config.heat.validate()
    count = len(config.line_angles)
    if not len(config.line_centers) == len(config.line_names) == count or not count:
        raise ValueError("Line centers, angles and names must have equal nonzero lengths")
    if not 0 < config.line_margin < 0.5 or config.candidate_points < 2 * config.heat.supports_per_patch:
        raise ValueError("Require an interior line margin and enough candidate points")
    center = np.asarray(config.line_centers[index], dtype=float)
    angle = config.line_angles[index]
    if not np.isfinite([*center, angle]).all() or not ((center > 0) & (center < 1)).all():
        raise ValueError("Line center must be strictly interior and center/angle finite")
    tangent = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    normal = np.array([tangent[1], -tangent[0]])
    rng = np.random.default_rng(config.heat.seed)
    candidates = rng.uniform(config.heat.support_margin, 1 - config.heat.support_margin,
                             (config.candidate_points, 2))
    signed = (candidates - center) @ normal
    count = config.heat.supports_per_patch
    regions = (candidates[signed < 0], candidates[signed > 0])
    if any(len(region) < count for region in regions):
        raise ValueError("Candidate pool has insufficient points in a requested half-domain")
    supports = np.stack([region[:count] for region in regions])
    limits = []
    for coordinate, direction in zip(center, tangent, strict=True):
        if abs(direction) > 1e-14:
            limits.append(sorted((-coordinate / direction, (1 - coordinate) / direction)))
    lower, upper = max(pair[0] for pair in limits), min(pair[1] for pair in limits)
    extent = upper - lower
    positions = np.linspace(lower + config.line_margin * extent,
                            upper - config.line_margin * extent, config.heat.stitches)
    stitches = center + positions[:, None] * tangent
    return supports, stitches, center, normal


def preview(config: LayoutConfig, *, directory=None):
    output = Path(directory) if directory else Path("results/stitch_line_preview") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    xy, initial, temperature, tensors = simulate(config.heat, (0, 90))
    np.savez_compressed(output / "heat.npz", xy=xy, initial_temperature=initial, temperature=temperature, tensors=tensors)
    order = np.lexsort((xy[:, 0], xy[:, 1]))
    fig, axes = plt.subplots(1, len(config.line_angles), figsize=(5 * len(config.line_angles), 6),
                             constrained_layout=True, squeeze=False)
    for index, ax in enumerate(axes.flat):
        supports, stitches, center, normal = layout(config, index)
        np.savez_compressed(output / f"layout_{index}.npz", supports=supports, stitches=stitches,
                            line_center=center, line_normal=normal)
        im = ax.imshow(temperature[order].reshape(config.heat.cells + 1, config.heat.cells + 1),
                       origin="lower", extent=(0, 1, 0, 1), cmap="inferno",
                       vmin=float(temperature.min()), vmax=float(temperature.max()))
        for patch, color in enumerate(("cyan", "lime")):
            ax.scatter(*supports[patch].T, color=color, edgecolors="black", linewidths=0.5,
                       s=29, label=f"Model {patch + 1}: initial supports")
        ax.axvline(0.5, color="white", ls="--", lw=1, label="True material interface")
        ax.plot(*stitches.T, color="deepskyblue", marker="x", ls=":", markersize=7,
                label="Initial stitch line")
        ax.set(title=f"{config.line_names[index]} ({config.line_angles[index]} deg)",
               xlabel="x", ylabel="y", xlim=(0, 1), ylim=(0, 1), aspect="equal")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=8)
    fig.colorbar(im, ax=list(axes.flat), shrink=0.6, label="Temperature")
    fig.suptitle("Same heat in every panel: horizontal pattern left, vertical pattern right\nInitial positions only: no training", fontsize=13)
    fig.savefig(output / "initial_layouts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2))
    (output / "status.json").write_text(json.dumps({"state": "preview_only", "training_epochs": 0}))
    (output / "provenance.json").write_text(json.dumps({
        "scikit_fem": skfem.__version__, "source_sha256": _file_digest(Path(__file__)),
        "heat_source_sha256": _file_digest(Path("experiments/periodic_heat_preview.py")),
        "artifacts_sha256": {p.name: _file_digest(p) for p in output.iterdir() if p.is_file()},
        "heat": "same PDE solution for all layouts; full tensors at 0/90 degrees",
        "training": "none; line-constrained training not implemented by this preview",
        "support_initialization": "uniform rejection sampling conditional on each half-domain, independent of temperature",
    }, indent=2))
    print(output.resolve())
    return output


if __name__ == "__main__":
    preview(LayoutConfig())
