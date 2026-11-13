"""Explicit benchmark repetitions on unchanged prepared data."""

import copy


def make_plan(request):
    from experiments.gpw_ablation import distinct_values
    from experiments.multidataset_worker import make_plan as comparison_plan
    from experiments.multidataset_worker import plan_hash

    seeds = request["seeds"]
    distinct_values(seeds, "seeds", lambda value: type(value) is int and value >= 0)
    interval = request["progress_every_epochs"]
    if type(interval) is not int or interval < 1:
        raise ValueError("progress_every_epochs must be a positive integer")
    base_request = {key: value for key, value in request.items() if key != "study"}
    result = comparison_plan(base_request)
    plan = result["plan"]
    if any(spec["seed"] != 31 for spec in plan["data"]):
        raise ValueError("Benchmark repeats preserve the approved data seed31")
    rows, summary = [], []
    for spec in plan["data"]:
        originals = [(row, item) for row, item in zip(plan["runs"], result["summary"], strict=True)
                     if row["dataset"] == spec["name"]]
        for seed in seeds:
            for original, item in originals:
                row = copy.deepcopy(original)
                config = row["config"]
                config["seed"] = seed
                if config["exact_gpw_reference"] is not None:
                    config["exact_gpw_reference"]["seed"] = seed
                row.update(base_name=original["name"], name=f"{original['name']} seed{seed}")
                rows.append(row)
                summary.append({**item, "model": row["name"], "seed": seed, "data_seed": spec["seed"]})
    plan.update(study={"type": "benchmark_repeats", "seeds": seeds,
        "data_seed_policy": "fixed seed31 data/preprocessing/splits; vary model initialization/training only",
        "seed31_results": "existing archive; not rerun or implicitly merged"},
        runs=rows, progress_every_epochs=interval)
    return {**result, "plan": plan, "plan_sha256": plan_hash(plan), "summary": summary,
            "total_runs": len(rows), "total_epochs": sum(row["config"]["epochs"] for row in rows)}
