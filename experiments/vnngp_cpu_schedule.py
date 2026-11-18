"""Finite multi-dataset continuation around the unchanged CPU VNNGP trainer."""

import json
import time
from pathlib import Path

import pandas as pd

from experiments import vnngp_cpu_worker as cpu
from experiments.multidataset_data import DatasetSpec, load_prepared, verify_manual_inputs
from experiments.multidataset_worker import plan_hash
from pwgp.run_records import record_failures


def make_schedule(request):
    if "dataset" in request:
        raise ValueError("Use DATASETS or the legacy DATASET selector, not both")
    names = request["datasets"]
    if (not isinstance(names, list) or not names or any(not isinstance(n, str) for n in names)
            or len(set(names)) != len(names)):
        raise ValueError("DATASETS must be a nonempty list of distinct dataset names")
    if set(request["data_overrides"]) - set(names):
        raise ValueError("DATA_OVERRIDES contains unselected datasets")
    parts = [cpu.make_plan({"dataset": name, "seeds": request["seeds"],
        "input_root": request["input_root"], "overrides": request["overrides"],
        "data_overrides": request["data_overrides"].get(name, {})}) for name in names]
    plan = json.loads(json.dumps({"study": "vnngp_cpu_table", "datasets": names,
        "dataset_plans": [part["plan"] for part in parts], "seeds": request["seeds"]}))
    return {"plan": plan, "plan_sha256": plan_hash(plan),
            "total_runs": sum(part["total_runs"] for part in parts),
            "maximum_epochs": sum(part["maximum_epochs"] for part in parts)}


def initialize(directory, plan, provenance, source, progress):
    directory.mkdir(parents=True, exist_ok=False)
    cpu.write_json(directory / "plan.json", plan)
    cpu.write_json(directory / "source_provenance.json", provenance)
    cpu.write_json(directory / "environment.json", cpu.environment())
    cpu.write_json(directory / "progress.json", progress)
    (directory / "source_snapshot.zip").write_bytes(source)


def verify_job(directory, plan, provenance):
    if cpu.read_json(directory / "plan.json") != plan:
        raise ValueError("Resume schedule changed; do not add/remove/reorder datasets or seeds")
    if cpu.read_json(directory / "source_provenance.json") != provenance:
        raise ValueError("Resume requires the original source commit/archive")
    current, original = cpu.environment(), cpu.read_json(directory / "environment.json")
    for key in ("python", "torch", "gpytorch", "numpy", "faiss", "package_version"):
        if current[key] != original[key]:
            raise ValueError(f"Resume dependency mismatch: {key}")
    progress = cpu.read_json(directory / "progress.json")
    if progress["state"] == "complete":
        raise ValueError("All requested datasets/seeds are already complete")
    count = progress["next_dataset_index"]
    if type(count) is not int or not 0 <= count < len(plan["dataset_plans"]):
        raise ValueError("Invalid saved dataset cursor")
    for part in plan["dataset_plans"][:count]:
        root = directory / part["data"]["name"]
        if cpu.read_json(root / "plan.json") != part:
            raise ValueError("Completed dataset plan identity mismatch")
        saved = cpu.read_json(root / "progress.json")
        if saved != {"state": "complete", "next_seed_index": len(part["configs"])}:
            raise ValueError("Completed dataset cursor is inconsistent")
        load_prepared(DatasetSpec(**part["data"]), root / "prepared")
    return progress


def save_summary(directory, plan):
    rows = []
    cursor = cpu.read_json(directory / "progress.json")["next_dataset_index"]
    for index, part in enumerate(plan["dataset_plans"]):
        if index > cursor:
            break
        name = part["data"]["name"]
        root = directory / name
        if not root.exists():
            if index < cursor:
                raise ValueError(f"Missing completed dataset outputs: {name}")
            break
        progress = cpu.read_json(root / "progress.json")
        for values in part["configs"][:progress["next_seed_index"]]:
            rows.append({"dataset": name, "model": "VNNGP", "seed": values["seed"],
                "data_seed": part["data"]["seed"], "neighbors": values["vnngp_neighbors"],
                **cpu.read_json(root / f"seed{values['seed']}" / "metrics.json")})
        if progress["state"] != "complete":
            if index < cursor:
                raise ValueError(f"Incomplete dataset before saved cursor: {name}")
            break
    if rows:
        pd.DataFrame(rows).to_csv(directory / "comparison.csv", index=False)


def execute_schedule(request, source):
    directory, plan = Path(request["directory"]), request["plan"]
    provenance = request["source_provenance"]
    started = time.monotonic()
    if request["action"] == "start":
        initialize(directory, plan, provenance, source, {"state": "paused", "next_dataset_index": 0})
    elif request["action"] == "resume":
        cpu.restore_archive(request["resume_archive"], directory)
    else:
        raise ValueError("Select start or resume explicitly")
    with record_failures(directory, status_key="state"):
        progress = verify_job(directory, plan, provenance)
        pending = plan["dataset_plans"][progress["next_dataset_index"]:]
        verify_manual_inputs([DatasetSpec(**part["data"]) for part in pending])
        hours = {part["settings"]["session_hours"] for part in plan["dataset_plans"]}
        if len(hours) != 1:
            raise ValueError("One session time limit must cover the whole schedule")
        deadline = started + hours.pop() * 3600
        while progress["next_dataset_index"] < len(plan["dataset_plans"]) and time.monotonic() < deadline:
            index = progress["next_dataset_index"]
            part = plan["dataset_plans"][index]
            root = directory / part["data"]["name"]
            print(f"Dataset {index + 1}/{len(plan['datasets'])}: {part['data']['name']}", flush=True)
            if not root.exists():
                initialize(root, part, provenance, source, {"state": "paused", "next_seed_index": 0})
                cpu.prepare_dataset(DatasetSpec(**part["data"]), directory=root / "prepared",
                                    data_root=request["data_root"])

            def seed_saved(*, index=index, root=root, part=part):
                local = cpu.read_json(root / "progress.json")
                progress["next_dataset_index"] = index + (local["state"] == "complete")
                progress["state"] = "complete" if progress["next_dataset_index"] == len(plan["datasets"]) else "paused"
                cpu.write_json(directory / "progress.json", progress)
                cpu.write_json(directory / "status.json", {"state": progress["state"]})
                save_summary(directory, plan)

            local = cpu.run_session(root, part, source_provenance=provenance, deadline=deadline,
                archive_directory=directory, on_seed_saved=seed_saved,
                on_complete=lambda output, archive: cpu.emit("complete", directory=str(output),
                    archive=str(archive), metrics=cpu.read_json(output / "metrics.json")))
            if local["state"] != "complete":
                break
        cpu.write_json(directory / "progress.json", progress)
        cpu.write_json(directory / "status.json", {"state": progress["state"]})
        save_summary(directory, plan)
    archive = cpu.archive_session(directory, request["archive"])
    return {"directory": str(directory), "archive": archive, **progress}
