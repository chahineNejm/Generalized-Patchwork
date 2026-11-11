"""Fresh-process OISST stages using the existing model/data runners unchanged."""

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from pwgp.run_records import record_failures


def emit(kind, **values):
    print("\nGPW_EVENT " + json.dumps({"kind": kind, **values}, allow_nan=False), flush=True)
    if sys.stdin.readline() != "continue\n":
        raise RuntimeError("Notebook did not acknowledge the worker event")


def source_snapshot(expected):
    root = Path(__file__).resolve().parents[1]
    subprocess.run(["git", "-C", str(root), "diff", "--exit-code", "HEAD"], check=True)
    untracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "--others", "--exclude-standard",
        "--", ":(glob)src/**/*.py", ":(glob)experiments/**/*.py"], text=True).strip()
    if untracked:
        raise ValueError(f"Untracked source files prevent verified execution: {untracked}")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    source = subprocess.check_output(["git", "-C", str(root), "archive", "--format=zip", commit,
                                      "pyproject.toml", "README.md", "src", "experiments"])
    if commit != expected["commit"] or hashlib.sha256(source).hexdigest() != expected["source_sha256"]:
        raise ValueError("Source changed since setup; rerun setup and settings before this stage")
    return root, source


def execute(request):
    from experiments.default_hyperparameters import oisst_defaults
    from experiments.oisst_benchmark import (
        OISSTConfig,
        archive_directory,
        completed_run_directories,
        load_prepared,
        prepare_experiment,
        run_prepared_comparison,
    )
    from experiments.run_selection import select_runs
    from experiments.ukhousing_benchmark import UKConfig
    from pwgp.config import ProSparConfig

    root, source = source_snapshot(request["source_provenance"])
    action = request["action"]
    if action == "settings":
        data, runs = oisst_defaults(device=request["device"])
        data = replace(data, **request["data_overrides"])
        runs = select_runs(runs, models=request["models"], overrides=request["overrides"])
        return {"data": asdict(data), "runs": [{"name": name, "config": asdict(config)} for name, config in runs]}
    if action == "prepare":
        directory = prepare_experiment(OISSTConfig(**request["data"]), directory=request["directory"],
            data_root=root / "data", source_provenance=request["source_provenance"], source_zip=source)
    elif action == "load":
        directory = Path(request["directory"])
        load_prepared(directory, data_config=OISSTConfig(**request["data"]))
    elif action == "train":
        runs = []
        for row in request["runs"]:
            values = dict(row["config"])
            if values["prospar"] is not None:
                values["prospar"] = ProSparConfig(**values["prospar"])
            runs.append((row["name"], UKConfig(**values)))
        directory = run_prepared_comparison(OISSTConfig(**request["data"]), runs,
            prepared_directory=request["prepared_directory"], directory=request["directory"],
            source_provenance=request["source_provenance"], source_zip=source,
            on_snapshot=lambda path, epoch: emit("snapshot", directory=str(path), epoch=epoch),
            on_complete=lambda path, archive, metrics: emit("complete", directory=str(path),
                                                           archive=str(archive), metrics=metrics))
    elif action == "analysis":
        from experiments.exploratory_analysis import recompute_saved_runs

        if request["analysis_source"] == "latest_batch":
            paths = completed_run_directories(request["batch_directory"])
        elif request["analysis_source"] == "saved_runs":
            paths = request["saved_runs"]
        else:
            raise ValueError("ANALYSIS_SOURCE must be latest_batch or saved_runs")
        directory = recompute_saved_runs(paths, destination=request["directory"])
    elif action == "archive":
        return {"archive": str(archive_directory(request["directory"], destination=request["destination"]))}
    else:
        raise ValueError(f"Unknown OISST worker action: {action}")
    return {"directory": str(Path(directory).resolve())}


def main(request_path):
    request_path = Path(request_path).resolve()
    with record_failures(request_path.parent, status_key="state"):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = {**execute(request), "worker_pid": os.getpid()}
        (request_path.parent / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        (request_path.parent / "status.json").write_text('{"state": "complete"}', encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
