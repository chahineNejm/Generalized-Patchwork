"""Dimension-independent comparison primitives; dataset policies are explicit."""

import gc
import importlib.metadata
import json
import os
import platform
import time
from dataclasses import asdict
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

from experiments.exploratory_analysis import plot_hyperparameters, plot_training
from experiments.oisst_benchmark import learned_parameters, predict, supports
from experiments.training_control import EarlyStoppingConfig, TrainingLossStopper
from experiments.ukhousing_benchmark import build_model, scores, train
from pwgp.config import HyperparameterBounds, NumericalConfig, OptimizerConfig
from pwgp.run_records import record_failures


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def prepare_split(dataset, *, train_fraction, seed):
    """Explicit min-max input scaling; targets use training-only statistics."""
    dataset = dataset.finite()
    x, y = dataset.x, dataset.y
    if not 0 < train_fraction < 1 or len(y) < 2:
        raise ValueError("A nonempty train/test split requires 0 < train_fraction < 1")
    origin, span = x.min(0), np.ptp(x, axis=0)
    if not np.isfinite(span).all() or (span <= 0).any():
        raise ValueError("Every input feature must have a positive finite range; features are never dropped")
    permutation = np.random.default_rng(seed).permutation(len(y))
    count = int(train_fraction * len(y))
    training, testing = permutation[:count], permutation[count:]
    if not len(training) or not len(testing):
        raise ValueError("Both train and test must contain observations")
    center, scale = float(y[training].mean()), float(y[training].std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training target standard deviation must be positive and finite")
    return {"x": torch.tensor((x - origin) / span, dtype=torch.float64),
            "y": torch.tensor((y - center) / scale, dtype=torch.float64),
            "raw_x": x, "raw_y": y, "train": training, "test": testing,
            "validation": np.empty(0, dtype=np.int64), "target_mean": center,
            "target_std": scale, "target_offset": np.full(len(y), center),
            "feature_names": list(dataset.feature_names), "target_name": dataset.target_name,
            "seed": seed, "score_splits": ["test"],
            "preprocessing": {"input_origin": origin.tolist(), "input_span": span.tolist(),
                "input_statistics_fit_on": "all input rows; no target information",
                "target_mean": center, "target_std": scale, "target_statistics_fit_on": "training only",
                "train_fraction": train_fraction, "seed": seed,
                "removed_nonfinite_at_split": dataset.metadata["removed_nonfinite"]}}


def plot_locations(initial, final, *, destination, title):
    if initial.shape != final.shape or initial.ndim != 3 or initial.shape[-1] < 2:
        raise ValueError("Location plots require matching [groups,points,D] arrays, D>=2")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, locations, label in zip(axes, (initial, final), ("Initial", "Final"), strict=True):
        for points in locations:
            ax.scatter(points[:, 0], points[:, 1], s=5)
        ax.set(title=label, xlabel="Normalized feature 1", ylabel="Normalized feature 2", xlim=(0, 1), ylim=(0, 1))
    fig.suptitle(title + (" (two-feature projection)" if initial.shape[-1] > 2 else ""))
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_predictions(truth, mean, variance, *, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(truth, mean, s=3)
    limits = (float(min(truth.min(), mean.min())), float(max(truth.max(), mean.max())))
    axes[0].plot(limits, limits, color="black", linewidth=1)
    axes[0].set(xlabel="Observed target", ylabel="Predicted target")
    axes[1].scatter(np.sqrt(variance), np.abs(mean - truth), s=3)
    axes[1].set(xlabel="Observation SD", ylabel="Absolute error")
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def run_model(config, split, *, directory, save_inducing_positions, on_snapshot=None, early_stopping=None,
              progress_description=None, progress_every_epochs=1):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    with record_failures(output, status_key="state"):
        if config.epochs < 1 or config.snapshot_every < 1:
            raise ValueError("epochs and snapshot_every must be positive")
        if type(progress_every_epochs) is not int or progress_every_epochs < 1:
            raise ValueError("progress_every_epochs must be a positive integer")
        if config.model != "patchwork" and not 0 <= config.warmup_epochs < config.epochs:
            raise ValueError("Require 0 <= warmup_epochs < epochs; no automatic warm-up adjustment")
        if config.model == "prospar" and config.prospar is None:
            raise ValueError("ProSpar requires explicit temperatures, exponent, nuggets and prior_jitter")
        stop_config = None if early_stopping is None else EarlyStoppingConfig(**early_stopping)
        if stop_config is not None and config.warmup_epochs >= config.epochs:
            raise ValueError("Early stopping requires post-warm-up epochs; no automatic schedule adjustment")
        stopper = None if stop_config is None else TrainingLossStopper(stop_config, warmup_epochs=config.warmup_epochs)
        write_json(output / "training_control.json", {"early_stopping": early_stopping,
            "monitor": "epoch training loss", "monitor_after_epoch": config.warmup_epochs,
            "evaluated_model": "last trained parameters; no restoration",
            "progress_description": progress_description, "progress_every_epochs": progress_every_epochs})
        torch.set_num_threads(config.threads)
        faiss.omp_set_num_threads(config.threads)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        device = torch.device(config.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA device is unavailable; no CPU fallback")
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        write_json(output / "config.json", asdict(config))
        write_json(output / "environment.json", {
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "gpytorch": gpytorch.__version__, "faiss": importlib.metadata.version("faiss-cpu"),
            "device": str(device), "dtype": "float64", "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "bounds": asdict(HyperparameterBounds()),
            "numerical": asdict(NumericalConfig(weight_epsilon=config.weight_epsilon)),
            "active_prior_jitter": config.prospar.prior_jitter if config.model == "prospar"
            else NumericalConfig().prior_jitter,
            "optimizer": {"method": "Adam", "learning_rate": config.exact_learning_rate, "gradient_clip": 10.0}
            if config.model == "patchwork" else asdict(OptimizerConfig(
                learning_rate_q=config.learning_rate_q, learning_rate_supports=config.learning_rate_structure,
                learning_rate_stitches=config.learning_rate_structure,
                learning_rate_hyperparameters=config.learning_rate_hyperparameters,
                warmup_fraction=config.warmup_epochs / config.epochs))})
        visuals = output / "visuals"
        visuals.mkdir()
        started = time.perf_counter()
        model = build_model(config, split)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        initialization_seconds = time.perf_counter() - started
        metadata = {"model_class": f"{type(model).__module__}.{type(model).__name__}",
            "package_version": importlib.metadata.version("generalized-patchwork-gp"),
            "kernel": "ARD RBF", "likelihood": "Gaussian", "dimensions": split["x"].shape[1],
            "training_observations_used": len(model.train_x) if config.model == "patchwork" else len(split["train"]),
            "training_points_policy": config.exact_training_points if config.model == "patchwork" else "all",
            "partition": config.exact_partition if config.model == "patchwork" else None,
            "inducing_positions_saved": save_inducing_positions and config.model != "patchwork"}
        if config.model == "prospar":
            metadata.update({"actual_inducing_points": config.inducing_points, "paper_inducing_points": 5000,
                "capacity_ratio_to_paper": config.inducing_points / 5000,
                "capacity_note": "Matched budget; not the paper's 200x25 expert configuration",
                "equations": "arXiv:2311.08752v1 (6)-(13), approved derived objective/prediction"})
        if config.model == "gpw":
            metadata.update({"stitch_mode": config.stitch_mode, "q_rank": config.q_rank,
                "active_stitch_count": model.stitches().numel() // model.supports().shape[-1],
                "stitches_trainable": model.stitch.trainable})
        write_json(output / "model_metadata.json", metadata)
        torch.save(model.state_dict(), output / "initial_model.pt")
        if config.model == "gpw":
            from experiments.gpw_ablation import model_geometry, verify_geometry

            identity = verify_geometry(model, split)
            np.savez_compressed(output / "initialization.npz", **model_geometry(model))
            write_json(output / "initialization_identity.json", {"geometry_sha256": identity,
                "data_seed": split["seed"], "model_training_seed": config.seed,
                "includes": ["selected rows", "supports", "centers", "candidate edges", "candidate stitches"]})
        if config.model == "patchwork":
            np.savez_compressed(output / "partition.npz", points=model.train_x.cpu().numpy(),
                targets=model.train_y.cpu().numpy(), indices=model.source_training_indices.cpu().numpy(),
                labels=model.labels.cpu().numpy(), tree=model.partition_tree.cpu().numpy(),
                bounds=model.region_bounds.cpu().numpy(), vertices=model.region_vertices.cpu().numpy(),
                offsets=model.region_offsets.cpu().numpy(), stitches=model.stitch_points.cpu().numpy(),
                coefficients=model.coefficients.cpu().numpy())
            points = model.train_x.detach().cpu().numpy()[None]
            plot_locations(points, points, destination=visuals / "training_points.png", title="Fixed observed training locations")
        position_history, position_epochs, parameters, stitch_history = [], [], [], []

        def snapshot(epoch, losses):
            stopped = stopper.update(epoch, losses[-1]) if stopper is not None and epoch > 0 else False
            write_json(output / "history.json", {"losses": losses})
            parameters.append({"epoch": epoch, **learned_parameters(model, config.model)})
            write_json(output / "hyperparameter_history.json", parameters)
            if epoch == 0 or epoch % config.snapshot_every == 0 or epoch == config.epochs or stopped:
                plot_training(losses, destination=visuals / "training.png")
                plot_hyperparameters(parameters, destination=visuals / "hyperparameters.png")
                if save_inducing_positions and config.model != "patchwork":
                    position_history.append(supports(model, config.model).copy())
                    position_epochs.append(epoch)
                    np.savez_compressed(output / "inducing_positions.npz", epochs=position_epochs,
                                        supports=np.stack(position_history))
                    plot_locations(position_history[0], position_history[-1],
                                   destination=visuals / "inducing_positions.png", title="Inducing locations")
                if config.model == "gpw":
                    stitch_history.append(model.stitches().detach().cpu().numpy())
                    np.savez_compressed(output / f"stitches_epoch_{epoch}.npz", stitches=stitch_history[-1])
                    if model.stitches().numel():
                        plot_locations(stitch_history[0], stitch_history[-1],
                                       destination=visuals / "stitches.png", title="Stitch locations")
                torch.save(model.state_dict(), output / "checkpoint.pt")
                if on_snapshot is not None:
                    on_snapshot(output, epoch)
            return stopped

        snapshot(0, [])
        started = time.perf_counter()
        losses = train(model, split, config, snapshot, progress_description=progress_description,
                       progress_every_epochs=progress_every_epochs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - started
        stop_triggered = stopper is not None and stopper.stopped
        stopped = stop_triggered and len(losses) < config.epochs
        write_json(output / "training_summary.json", {"maximum_epochs": config.epochs,
            "epochs_completed": len(losses), "stopped_early": stopped,
            "reason": "training_loss_patience" if stop_triggered else "maximum_epochs",
            "significant_best_loss": stopper.best_loss if stopper is not None else None,
            "significant_best_epoch": stopper.best_epoch if stopper is not None else None,
            "epochs_without_significant_improvement": stopper.bad_epochs if stopper is not None else None,
            "evaluated_epoch": len(losses), "checkpoint_restored": False})
        torch.save(model.state_dict(), output / "model.pt")
        write_json(output / "learned_parameters.json", learned_parameters(model, config.model))
        all_scores, prediction_seconds = {}, 0.0
        for label in tqdm(split["score_splits"], desc="score", unit="split"):
            indices = split[label]
            started = time.perf_counter()
            mean, variance = predict(model, split["x"][indices], batch_size=config.prediction_batch_size)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            prediction_seconds += time.perf_counter() - started
            raw_mean = mean * split["target_std"] + split["target_offset"][indices]
            raw_variance = variance * split["target_std"]**2
            all_scores[label] = {"standardized": scores(split["y"][indices].numpy(), mean, variance),
                                 "original_target": scores(split["raw_y"][indices], raw_mean, raw_variance)}
            np.savez_compressed(output / f"{label}_predictions.npz", indices=indices,
                inputs=split["raw_x"][indices], truth=split["raw_y"][indices],
                mean=raw_mean, observation_variance=raw_variance)
            if label == "test":
                plot_predictions(split["raw_y"][indices], raw_mean, raw_variance, destination=visuals / "predictions.png")
        metrics = {**{f"{key}_{scale}": value for scale, values in all_scores["test"].items()
                     for key, value in values.items()}, "target_name": split["target_name"], "final_loss": losses[-1],
            "epochs_completed": len(losses), "stopped_early": stopped,
            "initialization_seconds": initialization_seconds, "training_seconds": training_seconds,
            "prediction_seconds": prediction_seconds, "test_points": len(split["test"]),
            "peak_cuda_megabytes": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}
        write_json(output / "all_scores.json", all_scores)
        write_json(output / "metrics.json", metrics)
        write_json(output / "status.json", {"state": "complete"})
        del snapshot
        model = None
        gc.collect()
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return output
