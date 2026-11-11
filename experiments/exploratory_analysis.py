"""Separate dataset summaries and saved-run visual analysis, without fitting."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))

import matplotlib
import numpy as np
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def explore_dataset(split, *, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    x, y = split["raw_x"], split["raw_y"]
    if x.shape != (len(y), 2) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Exploratory analysis requires finite x[N,2] and y[N]")
    summary = {"observations": len(y), "input_min": x.min(0).tolist(),
               "input_max": x.max(0).tolist(), "target_min": float(y.min()),
               "target_max": float(y.max()), "target_mean": float(y.mean()),
               "target_std": float(y.std()),
               "duplicate_locations": len(x) - len(np.unique(x, axis=0)),
               "split_counts": {key: len(split[key]) for key in ("train", "validation", "test")}}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    for kind in tqdm(("targets", "distribution", "split"), desc="explore:data", unit="figure"):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        if kind == "targets":
            artist = ax.scatter(*x.T, c=y, s=3, cmap="viridis")
            fig.colorbar(artist, ax=ax, label="Temperature (C)")
        elif kind == "distribution":
            ax.hist(y, bins=50)
            ax.set(xlabel="Temperature (C)", ylabel="Observations")
        else:
            for label in ("train", "validation", "test"):
                ax.scatter(*x[split[label]].T, s=2, label=label, alpha=0.5)
            ax.legend()
        ax.set_title(kind)
        fig.savefig(directory / f"{kind}.png", dpi=140)
        plt.close(fig)
    return directory


def plot_training(losses, *, destination):
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.set(xlabel="Epoch", ylabel="Negative training objective", title="Training evolution")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_positions(initial, final, *, destination):
    if initial.shape != final.shape or initial.ndim != 3 or initial.shape[-1] != 2:
        raise ValueError("Point plots require matching [groups,points,2] arrays")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, positions, title in zip(axes, (initial, final), ("Initial", "Final"), strict=True):
        for j, points in enumerate(positions):
            ax.scatter(*points.T, s=5, label=f"Group {j + 1}")
        ax.set(title=title, xlim=(0, 1), ylim=(0, 1), xlabel="Normalized longitude",
               ylabel="Normalized latitude")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_hyperparameters(history, *, destination):
    epochs = [row["epoch"] for row in history]
    lengthscales = np.array([row["lengthscales"] for row in history]).reshape(len(history), -1)
    outputscales = np.array([row["outputscales"] for row in history]).reshape(len(history), -1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(epochs, lengthscales)
    axes[1].plot(epochs, outputscales)
    axes[2].plot(epochs, [row["noise_std"] for row in history])
    for ax, title in zip(axes, ("ARD lengthscales", "Output variance", "Noise SD"), strict=True):
        ax.set(title=title, xlabel="Epoch")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def analyze_run(directory):
    """Regenerate figures using only saved outputs."""
    directory = Path(directory)
    visuals = directory / "visuals"
    visuals.mkdir(exist_ok=True)
    history = json.loads((directory / "history.json").read_text())["losses"]
    plot_training(history, destination=visuals / "training.png")
    plot_hyperparameters(json.loads((directory / "hyperparameter_history.json").read_text()),
                         destination=visuals / "hyperparameters.png")
    metrics = json.loads((directory / "metrics.json").read_text())
    names = ["rmse", "nlpd", "coverage_95", "mean_interval_width"]
    fig, axes = plt.subplots(1, len(names), figsize=(12, 3), constrained_layout=True)
    for ax, name in zip(axes, names, strict=True):
        ax.bar([name], [metrics[name]])
        ax.set_title(f"{metrics[name]:.4g}")
    fig.savefig(visuals / "metrics.png", dpi=140)
    plt.close(fig)
    metadata = json.loads((directory / "model_metadata.json").read_text())
    if metadata["inducing_positions_saved"]:
        with np.load(directory / "inducing_positions.npz") as positions:
            plot_positions(positions["supports"][0], positions["supports"][-1],
                           destination=visuals / "inducing_positions.png")
    return visuals
