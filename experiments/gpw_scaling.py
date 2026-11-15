"""GPW and matched Standard Patchwork across patch counts at a fixed total budget."""

import copy


def make_plan(request):
    """One comparison plan per patch count; Patchwork uses that GPW's initial rows."""
    from experiments.gpw_ablation import distinct_values
    from experiments.multidataset_worker import make_plan as comparison_plan
    from experiments.multidataset_worker import plan_hash

    counts, budget, seeds = request["patch_counts"], request["inducing_points"], request["seeds"]
    distinct_values(seeds, "seeds", lambda v: type(v) is int and v >= 0)
    distinct_values(counts, "patch_counts", lambda v: type(v) is int and v >= 2 and v & (v - 1) == 0)
    if type(budget) is not int or any(budget % count for count in counts):
        raise ValueError("inducing_points must be an integer divisible by every patch count")
    stitches = request["stitches_per_boundary"]
    if (not isinstance(stitches, dict) or set(stitches) != {str(c) for c in counts}
            or any(type(v) is not int or v < 1 for v in stitches.values())):
        raise ValueError("stitches_per_boundary must map every patch count (as a string) to a positive integer")
    neighbors = request["patch_neighbors"]
    if type(neighbors) is not int or neighbors < 1:
        raise ValueError("patch_neighbors must be a positive integer")
    interval = request["progress_every_epochs"]
    if type(interval) is not int or interval < 1:
        raise ValueError("progress_every_epochs must be a positive integer")
    datasets = request["datasets"]
    base = {key: value for key, value in request.items()
            if key not in ("study", "seeds", "patch_counts", "inducing_points", "stitches_per_boundary",
                           "patch_neighbors", "progress_every_epochs")}
    if set(base["model_overrides"]) - set(datasets):
        raise ValueError("model_overrides must contain only selected datasets")
    controlled = {"patches", "inducing_points", "stitches_per_edge", "patch_neighbors",
                  "exact_tree_patches", "exact_stitches_per_edge", "exact_partition",
                  "exact_training_points", "exact_gpw_reference", "model", "seed"}
    for name, values in base["model_overrides"].items():
        if set(values) - {"GPW", "Standard Patchwork (PC tree)"}:
            raise ValueError(f"{name}: overrides may name only GPW and Standard Patchwork (PC tree)")
        for label, fields in values.items():
            if set(fields) & controlled:
                raise ValueError(f"{name}/{label}: scaling controls cannot be overridden: {sorted(set(fields) & controlled)}")
    data, runs, summary = None, [], []
    for count in counts:
        per_boundary = stitches[str(count)]
        local_neighbors = min(neighbors, count - 1)
        overrides = {}
        for name in datasets:
            user = base["model_overrides"].get(name, {})
            overrides[name] = {
                "GPW": {**user.get("GPW", {}), "inducing_points": budget, "patches": count,
                        "stitches_per_edge": per_boundary, "patch_neighbors": local_neighbors},
                "Standard Patchwork (PC tree)": {**user.get("Standard Patchwork (PC tree)", {}),
                        "exact_tree_patches": count, "exact_stitches_per_edge": per_boundary}}
        result = comparison_plan({**base, "model_overrides": overrides,
            "models": {name: ["GPW", "Standard Patchwork (PC tree)"] for name in datasets}})
        if data is None:
            data = result["plan"]["data"]
        elif result["plan"]["data"] != data:
            raise ValueError("Patch-count plans must share identical dataset configurations")
        if any(spec["seed"] != 31 for spec in data):
            raise ValueError("The scaling study preserves the approved data seed31")
        for seed in seeds:
            for original, item in zip(result["plan"]["runs"], result["summary"], strict=True):
                row = copy.deepcopy(original)
                config = row["config"]
                config["seed"] = seed
                if config["model"] == "patchwork":
                    config["exact_gpw_reference"]["seed"] = seed
                    reference = {"inducing_points": budget, "patches": count, "seed": seed, "threads": config["threads"]}
                    if config["exact_gpw_reference"] != reference:
                        raise ValueError("Standard Patchwork must use the same-P, same-seed GPW initialization")
                row.update(base_name=original["name"], name=f"{original['name']} P{count} seed{seed}",
                           patch_count=count, stitches_per_boundary=per_boundary)
                runs.append(row)
                summary.append({**item, "model": row["name"], "seed": seed, "patch_count": count,
                                "stitches_per_boundary": per_boundary,
                                "patch_neighbors": local_neighbors if config["model"] == "gpw" else None})
    plan = {**result["plan"], "data": data, "runs": runs, "progress_every_epochs": interval,
        "study": {"type": "gpw_scaling", "seeds": seeds, "patch_counts": counts,
            "data_seed_policy": "fixed seed31 data/preprocessing/splits; vary model initialization/training only", "inducing_points": budget,
            "stitches_per_boundary": stitches, "patch_neighbors": f"min({neighbors}, P-1)",
            "initialization_policy": "Standard Patchwork uses the selected rows of the GPW run with the same P",
            "budget_policy": "fixed total inducing points / selected observed targets across P"}}
    return {**result, "plan": plan, "plan_sha256": plan_hash(plan), "summary": summary,
            "total_runs": len(runs), "total_epochs": sum(row["config"]["epochs"] for row in runs)}
