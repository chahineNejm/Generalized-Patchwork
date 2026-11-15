"""Separate dataset summaries and saved-run visual analysis, without fitting."""

import importlib.metadata
import json
import os
from itertools import pairwise
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))

import matplotlib
import numpy as np
from tqdm.auto import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwgp.data.download import file_checksum
from pwgp.run_records import record_failures


def plot_toy_field(inputs, latent, *, destination, extent):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    artist = ax.scatter(*inputs.T, c=latent, s=3, cmap="viridis")
    ax.set(title="Sampled exponential-GP latent field", xlabel="x1", ylabel="x2",
           xlim=(0, extent), ylim=(0, extent), aspect="equal")
    fig.colorbar(artist, ax=ax, label="Latent function")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def toy_prediction_fields(directory, target_std):
    """Recover both variances in original units from saved observation moments."""
    directory = Path(directory)
    with np.load(directory / "test_predictions.npz", allow_pickle=False) as rows:
        values = {key: rows[key] for key in rows.files}
    parameters = json.loads((directory / "learned_parameters.json").read_text())
    noise_variance = (parameters["noise_std"] * target_std)**2
    latent_variance = values["observation_variance"] - noise_variance
    if not np.isfinite(latent_variance).all() or (latent_variance < 0).any():
        raise ValueError("Saved observation variance is smaller than the learned noise variance")
    return {**values, "latent_variance": latent_variance, "noise_variance": noise_variance}


def plot_toy_predictions(predictions, latent, *, destination, extent, title):
    fields = {"Latent truth": latent, "Predictive mean": predictions["mean"],
              "Latent variance": predictions["latent_variance"],
              "Observation variance": predictions["observation_variance"]}
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    for ax, (name, values) in zip(axes, fields.items(), strict=True):
        artist = ax.scatter(*predictions["inputs"].T, c=values, s=5, cmap="viridis")
        ax.set(title=name, xlabel="x1", ylabel="x2", xlim=(0, extent), ylim=(0, extent), aspect="equal")
        fig.colorbar(artist, ax=ax, shrink=.8)
    fig.suptitle(title)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def draw_toy_points(axes, directory, preprocessing, *, extent, title):
    directory = Path(directory)
    origin, span = np.array(preprocessing["input_origin"]), np.array(preprocessing["input_span"])
    with np.load(directory / "inducing_positions.npz", allow_pickle=False) as history:
        positions = history["supports"][[0, -1]] * span + origin
        epochs = history["epochs"][[0, -1]]
    for ax, supports, epoch, label in zip(axes, positions, epochs, ("Initial", "Final"), strict=True):
        for patch, points in enumerate(supports):
            ax.scatter(*points.T, s=9, color=plt.get_cmap("tab10")(patch % 10), alpha=.8)
        with np.load(directory / f"stitches_epoch_{epoch}.npz", allow_pickle=False) as saved:
            stitches = saved["stitches"].reshape(-1, 2) * span + origin
        if len(stitches):
            ax.scatter(*stitches.T, s=28, marker="+", color="black", linewidths=1.1, label="Stitches")
            ax.legend(loc="upper right", fontsize=8)
        ax.set(title=f"{title}: {label} ({len(stitches)} stitches)", xlabel="x1", ylabel="x2",
               xlim=(0, extent), ylim=(0, extent), aspect="equal")


def plot_toy_points(directory, preprocessing, *, destination, extent, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    draw_toy_points(axes, directory, preprocessing, extent=extent, title=title)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def plot_toy_comparison(directories, split, *, destination, extent, title):
    """Three-mode mean/variance maps with shared scales and paired point layouts."""
    modes = [json.loads((Path(path) / "config.json").read_text())["stitch_mode"] for path in directories]
    saved = [toy_prediction_fields(path, split["target_std"]) for path in directories]
    fields = ("mean", "latent_variance", "observation_variance")
    names = ("Predictive mean", "Latent variance", "Observation variance")
    fig, axes = plt.subplots(len(saved), 3, figsize=(12, 4 * len(saved)), constrained_layout=True, squeeze=False)
    for column, (field, name) in enumerate(zip(fields, names, strict=True)):
        vmin = min(float(row[field].min()) for row in saved)
        vmax = max(float(row[field].max()) for row in saved)
        for index, (mode, rows) in enumerate(zip(modes, saved, strict=True)):
            artist = axes[index, column].scatter(*rows["inputs"].T, c=rows[field], s=5,
                                                 cmap="viridis", vmin=vmin, vmax=vmax)
            axes[index, column].set(title=f"{mode}: {name}", xlabel="x1", ylabel="x2",
                xlim=(0, extent), ylim=(0, extent), aspect="equal")
        fig.colorbar(artist, ax=list(axes[:, column]), shrink=.8)
    fig.suptitle(title + " (held-out locations; shared scales within each column)")
    fig.savefig(Path(destination) / "means_variances.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(len(saved), 2, figsize=(11, 4 * len(saved)), constrained_layout=True, squeeze=False)
    for row, mode, path in zip(axes, modes, directories, strict=True):
        draw_toy_points(row, path, split["preprocessing"], extent=extent, title=mode)
    fig.suptitle(title + "; supports colored by patch, stitches black crosses")
    fig.savefig(Path(destination) / "point_positions.png", dpi=150)
    plt.close(fig)


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


def explore_multidataset(split, *, directory):
    """Full-dimensional summaries with explicitly labeled two-feature projections."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    x, y = split["raw_x"], split["raw_y"]
    if x.ndim != 2 or x.shape[0] != len(y) or x.shape[1] < 2:
        raise ValueError("Exploration requires x[N,D], D>=2, and y[N]")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Exploration requires finite prepared observations")
    summary = {"observations": len(y), "dimensions": x.shape[1], "feature_names": split["feature_names"],
        "input_min": x.min(0).tolist(), "input_max": x.max(0).tolist(),
        "target_name": split["target_name"], "target_mean": float(y.mean()), "target_std": float(y.std()),
        "target_min": float(y.min()), "target_max": float(y.max()),
        "duplicate_locations": len(x) - len(np.unique(x, axis=0)),
        "split_counts": {key: len(split[key]) for key in ("train", "validation", "test")}}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for kind in tqdm(("distribution", "projection"), desc="explore:data", unit="figure"):
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        if kind == "distribution":
            ax.hist(y, bins=50)
            ax.set(xlabel=split["target_name"], ylabel="Observations")
        else:
            artist = ax.scatter(x[:, 0], x[:, 1], c=y, s=2, cmap="viridis")
            fig.colorbar(artist, ax=ax, label=split["target_name"])
            ax.set(xlabel=split["feature_names"][0], ylabel=split["feature_names"][1],
                   title="Two-feature projection" if x.shape[1] > 2 else "Observed targets")
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


def plot_partition(points, labels, vertices, offsets, stitches, *, destination):
    """Show the fixed regions, observed training locations and boundary stitches."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(*points.T, c=labels, cmap="tab10", s=9)
    for lo, hi in pairwise(offsets):
        polygon = vertices[lo:hi]
        closed = np.vstack((polygon, polygon[0]))
        ax.plot(*closed.T, color="black", linewidth=0.8)
    if len(stitches):
        ax.scatter(*stitches.T, marker="x", color="black", s=16, label="Stitches")
        ax.legend()
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized longitude", ylabel="Normalized latitude",
           title=f"{len(points)} observed training locations; {len(offsets) - 1} fixed regions")
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


def analyze_run(directory, *, visuals_directory=None, metrics=None):
    """Regenerate figures using only saved outputs."""
    directory = Path(directory)
    visuals = directory / "visuals" if visuals_directory is None else Path(visuals_directory)
    visuals.mkdir(exist_ok=True)
    history = json.loads((directory / "history.json").read_text())["losses"]
    plot_training(history, destination=visuals / "training.png")
    plot_hyperparameters(json.loads((directory / "hyperparameter_history.json").read_text()),
                         destination=visuals / "hyperparameters.png")
    if metrics is None:
        metrics = json.loads((directory / "metrics.json").read_text())
    names = ["rmse", "nlpd", "coverage_95", "mean_interval_width"]
    fig, axes = plt.subplots(1, len(names), figsize=(12, 3), constrained_layout=True)
    for ax, name in zip(axes, names, strict=True):
        ax.bar([name], [metrics[name]])
        ax.set_title(f"{metrics[name]:.4g}")
    fig.savefig(visuals / "metrics.png", dpi=140)
    plt.close(fig)
    metadata = json.loads((directory / "model_metadata.json").read_text())
    if metadata.get("partition_artifacts_saved") is True:
        with np.load(directory / "training_points.npz", allow_pickle=False) as points, np.load(
                directory / "partition.npz", allow_pickle=False) as partition:
            plot_partition(points["x"], partition["labels"], partition["vertices"], partition["offsets"],
                           partition["stitches"], destination=visuals / "partition.png")
    if metadata["inducing_positions_saved"]:
        with np.load(directory / "inducing_positions.npz") as positions:
            plot_positions(positions["supports"][0], positions["supports"][-1],
                           destination=visuals / "inducing_positions.png")
    return visuals


def recompute_saved_run(directory, *, destination):
    """Recompute scores/plots from saved predictions without creating a model."""
    from experiments.oisst_benchmark import map_figure
    from experiments.ukhousing_benchmark import scores

    source, output = Path(directory), Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    with record_failures(output, status_key="state"):
        (output / "request.json").write_text(json.dumps({"original_run": str(source.resolve()),
            "training_performed": False, "prediction_performed": False}, indent=2))
        if json.loads((source / "status.json").read_text())["state"] != "complete":
            raise ValueError("Reanalysis requires an explicitly selected completed model run")
        preprocessing = json.loads((source.parent / "preprocessing.json").read_text())
        center, scale = preprocessing["target_mean"], preprocessing["target_std"]
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Saved target scaling must be finite with positive SD")
        all_scores = {}
        visuals = output / "visuals"
        visuals.mkdir()
        for label in tqdm(("validation", "test"), desc="reanalyze:scores", unit="split", leave=False):
            with np.load(source / f"{label}_predictions.npz", allow_pickle=False) as saved:
                truth, mean, variance = (saved[key] for key in ("truth", "mean", "observation_variance"))
                if truth.ndim != 1 or len(truth) == 0 or mean.shape != truth.shape or variance.shape != truth.shape:
                    raise ValueError("Saved predictions require matching nonempty truth/mean/variance vectors")
                if not np.isfinite(truth).all():
                    raise ValueError("Saved prediction targets must be finite")
                all_scores[label] = {"celsius": scores(truth, mean, variance),
                    "standardized": scores((truth - center) / scale, (mean - center) / scale, variance / scale**2)}
                if label == "test":
                    map_figure(saved["coordinates"], {"Observed (C)": truth, "Predicted (C)": mean,
                        "Absolute error (C)": np.abs(mean - truth), "Observation SD (C)": np.sqrt(variance)},
                        destination=visuals / "maps.png")
        metrics = {**json.loads((source / "metrics.json").read_text()), **all_scores["test"]["celsius"]}
        for name, value in (("metrics.json", metrics), ("all_scores.json", all_scores)):
            (output / name).write_text(json.dumps(value, indent=2, allow_nan=False))
        inputs = ("validation_predictions.npz", "test_predictions.npz", "history.json",
                  "hyperparameter_history.json", "model_metadata.json", "metrics.json")
        provenance = {"original_run": str(source.resolve()), "package_version":
            importlib.metadata.version("generalized-patchwork-gp"),
            "analysis_source_sha256": file_checksum(Path(__file__)),
            "preprocessing_sha256": file_checksum(source.parent / "preprocessing.json"),
            "inputs_sha256": {name: file_checksum(source / name) for name in inputs},
            "training_performed": False, "prediction_performed": False}
        metadata = json.loads((source / "model_metadata.json").read_text())
        if metadata.get("partition_artifacts_saved") is True:
            provenance["inputs_sha256"].update({name: file_checksum(source / name)
                for name in ("partition.npz", "training_points.npz")})
        if metadata["inducing_positions_saved"]:
            provenance["inputs_sha256"]["inducing_positions.npz"] = file_checksum(source / "inducing_positions.npz")
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False))
        analyze_run(source, visuals_directory=visuals, metrics=metrics)
        (output / "status.json").write_text(json.dumps({"state": "complete"}))
    return output


def recompute_saved_runs(directories, *, destination):
    directories = [Path(path).resolve() for path in directories]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Select a nonempty list of distinct completed run directories")
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=False)
    with record_failures(output, status_key="state"):
        (output / "selected_runs.json").write_text(json.dumps([str(path) for path in directories], indent=2))
        summaries = []
        for index, directory in enumerate(tqdm(directories, desc="reanalyze", unit="run")):
            result = recompute_saved_run(directory, destination=output / f"{index:02d}")
            summaries.append({"original_run": str(directory), "analysis_directory": str(result.resolve()),
                              **json.loads((result / "metrics.json").read_text())})
            (output / "comparison.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
        (output / "status.json").write_text(json.dumps({"state": "complete"}))
    return output
