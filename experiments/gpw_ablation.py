"""Explicit factorial plans and matched geometry for GPW ablations."""

import hashlib
import json
from dataclasses import asdict, fields, replace

import numpy as np
import torch

from experiments.default_hyperparameters import multidataset_defaults
from experiments.training_control import EarlyStoppingConfig
from experiments.ukhousing_benchmark import UKConfig, initial_gpw_points
from pwgp.initialization import initialize_stitches, patch_neighbor_graph


def distinct_values(values, name, predicate):
    if (not isinstance(values, list) or not values or any(not predicate(v) for v in values)
            or len(values) != len(set(values))):
        raise ValueError(f"{name} must be a nonempty list of distinct valid values")


def make_plan(request):
    """Vary model/training seeds; never regenerate the approved data splits."""
    from experiments.multidataset_worker import plan_hash

    data, presets, deferred = multidataset_defaults(device=request["device"], input_root=request["input_root"])
    selected, seeds, ranks, modes = (request[k] for k in ("datasets", "seeds", "ranks", "stitch_modes"))
    distinct_values(selected, "datasets", lambda v: isinstance(v, str) and v in data)
    distinct_values(seeds, "seeds", lambda v: type(v) is int and v >= 0)
    budget, patches = request["inducing_points"], request["patches"]
    if (type(budget) is not int or type(patches) is not int or patches < 1
            or budget < patches or budget % patches):
        raise ValueError("Positive integer inducing_points must divide evenly across positive patches")
    distinct_values(ranks, "ranks", lambda v: type(v) is int and 0 <= v <= budget)
    distinct_values(modes, "stitch_modes", lambda v: isinstance(v, str) and v in ("none", "fixed", "learned"))
    epochs, warmup = request["epochs"], request["warmup_epochs"]
    if type(epochs) is not int or type(warmup) is not int or not 0 <= warmup < epochs:
        raise ValueError("Require integer 0 <= warmup_epochs < epochs; no automatic adjustment")
    if type(request["save_inducing_positions"]) is not bool:
        raise TypeError("save_inducing_positions must be an explicit boolean")
    interval = request["progress_every_epochs"]
    if type(interval) is not int or interval < 1:
        raise ValueError("progress_every_epochs must be a positive integer")
    stopping = request["early_stopping"]
    if stopping is not None:
        EarlyStoppingConfig(**stopping)
    overrides, data_overrides = request["model_overrides"], request["data_overrides"]
    for name, mapping in (("model_overrides", overrides), ("data_overrides", data_overrides)):
        if not isinstance(mapping, dict) or set(mapping) - set(selected):
            raise ValueError(f"{name} must contain only selected dataset names")
    controlled = {"model", "seed", "q_rank", "stitch_mode", "kernel_mode", "patches", "inducing_points", "epochs", "warmup_epochs"}
    allowed = {f.name for f in fields(UKConfig) if f.name not in controlled | {"prospar"}
               and not f.name.startswith(("exact_", "vnngp_"))}
    specs, runs, summary = [], [], []
    for name in selected:
        spec = replace(data[name], **data_overrides[name]) if name in data_overrides else data[name]
        if (spec.name, spec.kind, spec.dimension, spec.seed) != (
                data[name].name, data[name].kind, data[name].dimension, data[name].seed):
            raise ValueError("Ablation data overrides cannot change source/schema or the fixed data seed")
        specs.append(asdict(spec))
        base = dict(presets[name])["GPW"]
        local = overrides.get(name, {})
        if set(local) - allowed:
            raise ValueError(f"{name}: override conflicts with factorial controls or unknown fields: {sorted(set(local)-allowed)}")
        base = replace(base, **local, inducing_points=budget, patches=patches,
                       epochs=epochs, warmup_epochs=warmup)
        if type(base.stitches_per_edge) is not int or base.stitches_per_edge < 1:
            raise ValueError("Fixed/learned ablation candidates require positive integer stitches_per_edge")
        for seed in seeds:
            for rank in ranks:
                for mode in modes:
                    config = replace(base, seed=seed, q_rank=rank, stitch_mode=mode)
                    label = f"GPW seed{seed} rank{rank} stitches-{mode}"
                    runs.append({"dataset": name, "name": label, "config": asdict(config)})
                    summary.append({"dataset": name, "seed": seed, "data_seed": spec.seed,
                        "rank": rank, "stitches": mode, "inducing_points": budget, "patches": patches,
                        "epochs": epochs, "warmup_epochs": warmup})
    plan = {"study": {"type": "gpw_stitch_rank", "seeds": seeds, "ranks": ranks, "stitch_modes": modes,
            "data_seed_policy": "preserve existing dataset splits and synthetic realizations",
            "initialization_policy": "identical geometry across variants within each dataset/model seed",
            "kernel_comparison": "deferred; separate learned local kernels unchanged"},
        "data": specs, "runs": runs, "save_inducing_positions": request["save_inducing_positions"],
        "progress_every_epochs": interval,
        "early_stopping": stopping, "excluded_model_types": ["svgp", "prospar", "vnngp", "patchwork"],
        "deferred": deferred}
    return {"plan": plan, "plan_sha256": plan_hash(plan), "summary": summary,
            "total_runs": len(runs), "total_epochs": len(runs)*epochs}


def geometry_digest(arrays):
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        digest.update(json.dumps([key, array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def bind_initialization(split, config):
    """Construct one immutable training-row initialization per model seed."""
    reference = {k: getattr(config, k) for k in ("inducing_points", "patches", "seed", "threads")}
    initial = initial_gpw_points(split["x"][split["train"]], reference,
                               {k: v for k, v in split.items() if not k.startswith("gpw_initialization")})
    edges = patch_neighbor_graph(initial["supports"], neighbors=config.patch_neighbors)
    stitches = initialize_stitches(initial["supports"], edges, config.stitches_per_edge)
    geometry = {"supports": initial["supports"].numpy(), "centers": initial["centers"].numpy(),
        "indices": np.asarray(split["train"])[initial["indices"].numpy()],
        "candidate_edges": np.asarray(edges, dtype=np.int64).reshape(-1, 2),
        "candidate_stitches": stitches.numpy()}
    split["gpw_initialization"] = initial
    split["gpw_initialization_reference"] = reference
    split["gpw_initialization_sha256"] = geometry_digest(geometry)
    return geometry


def model_geometry(model):
    return {"supports": model.initial_support_locations.detach().cpu().numpy(),
        "centers": model.initial_cluster_centers.detach().cpu().numpy(),
        "indices": model.source_initial_indices.detach().cpu().numpy(),
        "candidate_edges": model.candidate_edge_pairs.detach().cpu().numpy(),
        "candidate_stitches": model.candidate_stitch_locations.detach().cpu().numpy()}


def verify_geometry(model, split):
    digest = geometry_digest(model_geometry(model))
    if "gpw_initialization_sha256" in split and digest != split["gpw_initialization_sha256"]:
        raise ValueError("GPW initial supports/rows/centers/graph/stitches differ from matched reference")
    if not torch.allclose(model.initial_support_locations, model.supports().detach(), rtol=0, atol=1e-14):
        raise ValueError("GPW initial constrained supports do not match selected training locations")
    return digest


def summarize_results(rows):
    """Descriptive seed means and sample SDs; one-seed SD is explicitly undefined."""
    groups = {}
    for row in rows:
        key = (row["dataset"], row["q_rank"], row["stitch_mode"])
        groups.setdefault(key, []).append(row)
    result = []
    for (dataset, rank, mode), values in groups.items():
        seeds = [v["seed"] for v in values]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Ablation summary contains repeated seeds in the same condition")
        metrics = {}
        for name in values[0]:
            if name.endswith(("_standardized", "_original_target")) or name in (
                    "training_seconds", "prediction_seconds", "initialization_seconds", "epochs_completed"):
                samples = np.array([v[name] for v in values], dtype=float)
                if not np.isfinite(samples).all():
                    raise ValueError(f"Nonfinite ablation summary metric: {name}")
                metrics[name] = {"mean": float(samples.mean()),
                    "sample_sd": float(samples.std(ddof=1)) if len(samples) > 1 else None}
        result.append({"dataset": dataset, "q_rank": rank, "stitch_mode": mode,
                       "seeds": seeds, "completed_seeds": len(seeds), "metrics": metrics})
    return result


def plot_summary(rows, destination):
    from pathlib import Path

    import matplotlib.pyplot as plt

    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    summary = summarize_results(rows)
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        local = [r for r in summary if r["dataset"] == dataset]
        modes = list(dict.fromkeys(r["stitch_mode"] for r in local))
        for rank in dict.fromkeys(r["q_rank"] for r in local):
            ranked = [r for r in local if r["q_rank"] == rank]
            for ax, metric in zip(axes, ("rmse_standardized", "nlpd_standardized", "coverage_95_standardized"), strict=True):
                x = [modes.index(r["stitch_mode"]) for r in ranked]
                means = [r["metrics"][metric]["mean"] for r in ranked]
                ax.plot(x, means, "o-", label=f"rank {rank}")
                for point, mean, row in zip(x, means, ranked, strict=True):
                    sd = row["metrics"][metric]["sample_sd"]
                    if sd is not None:
                        ax.errorbar(point, mean, yerr=sd, capsize=4, color="black")
                ax.set(xticks=range(len(modes)), xticklabels=modes, title=metric)
                ax.legend()
                ax.grid(alpha=.2)
        axes[2].axhline(.95, color="black", linestyle="--", linewidth=1)
        fig.suptitle(f"{dataset}: seed mean and sample SD (not confidence intervals)")
        fig.savefig(destination / f"{dataset}.png", dpi=140)
        plt.close(fig)
