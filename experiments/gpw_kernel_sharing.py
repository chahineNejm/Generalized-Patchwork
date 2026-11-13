"""Shared-kernel GPW runs matched to the approved local-kernel ablations."""

from dataclasses import asdict, replace

from experiments.gpw_ablation import make_plan as ablation_plan
from experiments.gpw_ablation import summarize_results as ablation_summary


def make_plan(request):
    from experiments.multidataset_worker import plan_hash

    if set(request) & {"ranks", "stitch_modes", "kernel_mode", "kernel_modes"}:
        raise ValueError("Shared-only study uses q_rank and learned stitches; conflicting factorial controls are not accepted")
    result = ablation_plan({**request, "ranks": [request["q_rank"]], "stitch_modes": ["learned"]})
    plan = result["plan"]
    plan["study"] = {"type": "gpw_kernel_sharing", "kernel_mode": "shared", "seeds": request["seeds"],
        "data_seed_policy": "fixed approved data splits and preprocessing; vary model/training seeds",
        "comparison": "local rank-matched learned-stitch ablation results; supplied/verified later, never auto-loaded",
        "initialization_policy": "same initial rows/supports/centers/graph/stitches as local reference per dataset/seed"}
    for row, summary in zip(plan["runs"], result["summary"], strict=True):
        row["config"]["kernel_mode"] = "shared"
        row["name"] += " kernels-shared"
        summary["kernel_mode"] = "shared"
    return {**result, "plan": plan, "plan_sha256": plan_hash(plan)}


def reference_record(config, split):
    return {"local_reference_config": asdict(replace(config, kernel_mode="local")),
        "data_seed": split["seed"], "model_training_seed": config.seed,
        "geometry_sha256": split["gpw_initialization_sha256"],
        "required_checks": ["all configuration fields except kernel_mode", "source/numerical algebra identity",
            "data checksums/preprocessing/split identity", "initial geometry SHA256"],
        "reuse": "No local results loaded or claimed completed by this shared-only run"}


def summarize_results(rows):
    if any(row["kernel_mode"] != "shared" for row in rows):
        raise ValueError("Shared-kernel summary cannot pool different kernel modes")
    return [{**item, "kernel_mode": "shared"} for item in ablation_summary(rows)]
