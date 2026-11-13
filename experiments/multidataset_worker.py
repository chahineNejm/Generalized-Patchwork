"""Bounded, explicitly authorized stages for the Kaggle multi-dataset launcher."""

import hashlib
import json
import os
import shutil
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from experiments.multidataset_benchmark import run_model, write_json
from experiments.multidataset_data import (
    DatasetSpec,
    load_prepared,
    prepare_dataset,
    verify_manual_inputs,
)
from experiments.oisst_worker import emit, source_snapshot
from experiments.training_control import EarlyStoppingConfig
from pwgp.run_records import record_failures


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, allow_nan=False).encode()).hexdigest()


def archive_completed_run(directory):
    directory = Path(directory)
    if json.loads((directory / "status.json").read_text())["state"] != "complete":
        raise ValueError("Only completed runs can be archived")
    root = directory.parent
    shared = [root / name for name in ("source_snapshot.zip", "split.npz", "preprocessing.json",
        "data_manifest.json", "comparison.csv", "source_provenance.json", "data_config.json",
        "metadata.json", "prepared.json")]
    shared.extend(sorted((root / "data_visuals").rglob("*")))
    target = root.parent / f"{root.name}_{directory.name}.zip"
    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in shared + sorted(p for p in directory.rglob("*") if p.is_file()):
            if path.is_dir():
                continue
            bundle.write(path, path.relative_to(root))
    return target


def make_plan(request):
    if "study" in request:
        if request["study"] == "gpw_kernel_sharing":
            from experiments.gpw_kernel_sharing import make_plan as kernel_plan

            return kernel_plan(request)
        if request["study"] == "benchmark_repeats":
            from experiments.benchmark_repeats import make_plan as repeat_plan

            return repeat_plan(request)
        if request["study"] != "gpw_stitch_rank":
            raise ValueError(f"Unknown study: {request['study']}; no substitute comparison")
        from experiments.gpw_ablation import make_plan as ablation_plan

        return ablation_plan(request)
    from experiments.default_hyperparameters import multidataset_defaults
    from experiments.run_selection import select_runs

    data, defaults, deferred = multidataset_defaults(device=request["device"], input_root=request["input_root"])
    selected = request["datasets"]
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("DATASETS must be a nonempty list of distinct names")
    if set(selected) - data.keys():
        raise ValueError(f"Unknown or explicitly deferred DATASETS: {sorted(set(selected) - data.keys())}")
    for key in ("models", "model_overrides", "data_overrides"):
        if set(request[key]) - data.keys():
            raise ValueError(f"{key} contains unknown dataset names")
    excluded = request.get("excluded_model_types", [])
    known_types = {config.model for runs in defaults.values() for _, config in runs}
    if (not isinstance(excluded, list) or any(not isinstance(value, str) for value in excluded)
            or len(set(excluded)) != len(excluded) or set(excluded) - known_types):
        raise ValueError("excluded_model_types must list distinct known model types")
    epochs, warmup = request["epochs"], request["warmup_epochs"]
    if (not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 1
            or not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0):
        raise ValueError("EPOCHS must be a positive integer; WARMUP_EPOCHS a nonnegative integer")
    if not isinstance(request["save_inducing_positions"], bool):
        raise TypeError("save_inducing_positions must be an explicit boolean")
    early_stopping = request.get("early_stopping")
    if early_stopping is not None:
        EarlyStoppingConfig(**early_stopping)
    specs, runs, summary = [], [], []
    for name in selected:
        spec = replace(data[name], **request["data_overrides"][name]) if name in request["data_overrides"] else data[name]
        if (spec.name, spec.kind, spec.dimension) != (data[name].name, data[name].kind, data[name].dimension):
            raise ValueError("Dataset overrides cannot relabel a source or change its verified feature schema")
        specs.append(asdict(spec))
        available = [(label, replace(config, epochs=epochs, warmup_epochs=warmup)) for label, config in defaults[name]]
        models = request["models"][name] if name in request["models"] else [
            label for label, config in available if config.model not in excluded]
        overrides = request["model_overrides"].get(name, {})
        chosen = select_runs(available, models=models, overrides=overrides)
        for label, config in chosen:
            if config.model in excluded:
                raise ValueError(f"{name}/{label}: explicitly selected model conflicts with excluded_model_types")
            if (type(config.epochs) is not int or config.epochs < 1
                    or type(config.warmup_epochs) is not int or config.warmup_epochs < 0):
                raise ValueError(f"{name}/{label}: epochs/warm-up must be positive/nonnegative integers")
            if config.model != "patchwork" and not 0 <= config.warmup_epochs < config.epochs:
                raise ValueError(f"{name}/{label}: require 0 <= warm-up < epochs; no automatic adjustment")
            if config.seed != spec.seed:
                raise ValueError(f"{name}/{label}: model and shared dataset seed must match explicitly")
            if early_stopping is not None and config.warmup_epochs >= config.epochs:
                raise ValueError(f"{name}/{label}: early stopping must have post-warm-up epochs to monitor")
            runs.append({"dataset": name, "name": label, "config": asdict(config)})
            summary.append({"dataset": name, "model": label, "dimensions": spec.dimension,
                "epochs": config.epochs, "warmup_epochs": config.warmup_epochs if config.model != "patchwork" else "not used",
                "inducing_points": "N_train" if config.model == "vnngp" else
                config.inducing_points if config.model != "patchwork" else "none (observed targets)",
                "patches": config.patches if config.model in ("gpw", "prospar") else
                config.exact_tree_patches if config.model == "patchwork" and config.exact_partition == "pc_tree" else None,
                "neighbors": config.vnngp_neighbors if config.model == "vnngp" else None,
                "training_points": "GPW initial subset" if config.model == "patchwork" else "all",
                "prospar_paper_points": 5000 if config.model == "prospar" else None})
    plan = {"data": specs, "runs": runs, "save_inducing_positions": request["save_inducing_positions"],
            "early_stopping": early_stopping, "excluded_model_types": excluded, "deferred": deferred}
    return {"plan": plan, "plan_sha256": plan_hash(plan), "summary": summary,
            "total_runs": len(runs), "total_epochs": sum(row["config"]["epochs"] for row in runs),
            "available": {name: [label for label, _ in values] for name, values in defaults.items()}}


def execute(request):
    _, source = source_snapshot(request["source_provenance"])
    action = request["action"]
    if action == "settings":
        return make_plan(request)
    plan = request["plan"]
    if request["plan_sha256"] != plan_hash(plan):
        raise ValueError("Execution plan changed since settings; rerun settings explicitly")
    specs = [DatasetSpec(**values) for values in plan["data"]]
    study = plan["study"]["type"] if "study" in plan else None
    if study not in (None, "gpw_stitch_rank", "benchmark_repeats", "gpw_kernel_sharing"):
        raise ValueError(f"Unknown execution study: {study}")
    directory = Path(request["directory"])
    if action == "preflight":
        import torch

        verify_manual_inputs(specs)
        for row in plan["runs"]:
            device = torch.device(row["config"]["device"])
            if device.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("Enable the requested Kaggle GPU; no CPU substitution")
                torch.cuda.synchronize(device)
        directory.mkdir(parents=True, exist_ok=False)
        write_json(directory / "plan.json", plan)
        write_json(directory / "source_provenance.json", request["source_provenance"])
        return {"directory": str(directory), "free_disk_gb": shutil.disk_usage(directory).free / 2**30,
                "total_runs": len(plan["runs"]), "total_epochs": sum(r["config"]["epochs"] for r in plan["runs"])}
    if action == "load":
        for spec in specs:
            load_prepared(spec, directory / spec.name)
        return {"directory": str(directory)}
    if action == "prepare":
        verify_manual_inputs(specs)
        directory.mkdir(parents=True, exist_ok=False)
        with record_failures(directory, status_key="state"):
            write_json(directory / "plan.json", plan)
            write_json(directory / "source_provenance.json", request["source_provenance"])
            for spec in tqdm(specs, desc="prepare:datasets", unit="dataset"):
                prepare_dataset(spec, directory=directory / spec.name, data_root=request["data_root"])
            write_json(directory / "status.json", {"state": "complete"})
        return {"directory": str(directory)}
    if action == "train":
        if request["execute_training"] is not True:
            raise RuntimeError("Training is disabled; review the plan and explicitly set RUN_TRAINING=True")
        from experiments.ukhousing_benchmark import UKConfig
        from pwgp.config import ProSparConfig

        prepared = Path(request["prepared_directory"])
        for spec in specs:
            load_prepared(spec, prepared / spec.name)
        directory.mkdir(parents=True, exist_ok=False)
        with record_failures(directory, status_key="state"):
            write_json(directory / "plan.json", plan)
            write_json(directory / "source_provenance.json", request["source_provenance"])
            (directory / "source_snapshot.zip").write_bytes(source)
            summaries = []
            for spec in tqdm(specs, desc="compare:datasets", unit="dataset"):
                split = load_prepared(spec, prepared / spec.name)
                output = directory / spec.name
                output.mkdir(exist_ok=False)
                write_json(output / "source_provenance.json", request["source_provenance"])
                (output / "source_snapshot.zip").write_bytes(source)
                write_json(output / "data_config.json", asdict(spec))
                for name in ("split.npz", "preprocessing.json", "data_manifest.json", "prepared.json", "metadata.json"):
                    (output / name).write_bytes((prepared / spec.name / name).read_bytes())
                shutil.copytree(prepared / spec.name / "visuals", output / "data_visuals")
                if spec.kind == "modis":
                    (output / "source_rows.npy").write_bytes((prepared / spec.name / "source_rows.npy").read_bytes())
                local_rows = []
                initialization_seed = None
                for index, row in enumerate(tqdm([r for r in plan["runs"] if r["dataset"] == spec.name],
                                                  desc=f"compare:{spec.name}", unit="model")):
                    values = dict(row["config"])
                    if values["prospar"] is not None:
                        values["prospar"] = ProSparConfig(**values["prospar"])
                    config = UKConfig(**values)
                    if study in ("gpw_stitch_rank", "gpw_kernel_sharing"):
                        from experiments.gpw_ablation import bind_initialization

                        if config.model != "gpw":
                            raise ValueError("GPW ablation plan must contain only the declared GPW study")
                        if initialization_seed != config.seed:
                            bind_initialization(split, config)
                            initialization_seed = config.seed
                    if study == "gpw_kernel_sharing" and (
                            config.kernel_mode != "shared" or config.stitch_mode != "learned"):
                        raise ValueError("Kernel-sharing study requires shared GPW kernels and learned stitches")
                    result = run_model(config, split, directory=output / f"run_{index:02d}",
                        save_inducing_positions=plan["save_inducing_positions"],
                        early_stopping=plan.get("early_stopping"),
                        progress_description=f"train:{spec.name} / {row['name']}",
                        progress_every_epochs=plan["progress_every_epochs"] if "study" in plan else 1,
                        on_snapshot=lambda path, epoch: emit("snapshot", directory=str(path), epoch=epoch))
                    metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
                    if study == "gpw_kernel_sharing":
                        from experiments.gpw_kernel_sharing import reference_record

                        write_json(result / "kernel_comparison_reference.json", reference_record(config, split))
                    summary = {"dataset": spec.name, "model": row["name"], "directory": str(result), **metrics}
                    if study is not None:
                        summary.update({"seed": config.seed, "data_seed": spec.seed,
                            "q_rank": config.q_rank, "stitch_mode": config.stitch_mode,
                            "inducing_points": config.inducing_points, "patches": config.patches})
                    if study == "benchmark_repeats":
                        summary["base_model"] = row["base_name"]
                    if study == "gpw_kernel_sharing":
                        summary["kernel_mode"] = config.kernel_mode
                    local_rows.append(summary)
                    summaries.append(summary)
                    pd.DataFrame(local_rows).to_csv(output / "comparison.csv", index=False)
                    pd.DataFrame(summaries).to_csv(directory / "comparison.csv", index=False)
                    if study == "gpw_stitch_rank":
                        from experiments.gpw_ablation import summarize_results

                        write_json(directory / "ablation_summary.json", summarize_results(summaries))
                    if study == "gpw_kernel_sharing":
                        from experiments.gpw_kernel_sharing import (
                            summarize_results as kernel_summary,
                        )

                        write_json(directory / "kernel_summary.json", kernel_summary(summaries))
                    if spec.kind == "modis":
                        (result / "source_rows.npy").write_bytes((output / "source_rows.npy").read_bytes())
                    archive = archive_completed_run(result)
                    emit("complete", dataset=spec.name, name=row["name"], directory=str(result),
                         archive=str(archive), metrics=metrics)
                write_json(output / "status.json", {"state": "complete"})
            if study == "gpw_stitch_rank":
                from experiments.gpw_ablation import plot_summary

                plot_summary(summaries, directory / "visuals")
            write_json(directory / "status.json", {"state": "complete"})
        return {"directory": str(directory)}
    if action == "archive":
        if json.loads((directory / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
            raise ValueError("Only a completed collection can be archived by this action")
        destination = Path(request["destination"])
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(directory.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(directory))
        return {"archive": str(destination)}
    raise ValueError(f"Unknown multi-dataset action: {action}")


def main(request_path):
    request_path = Path(request_path)
    with record_failures(request_path.parent, status_key="state"):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        write_json(request_path.parent / "result.json", {**execute(request), "worker_pid": os.getpid()})
        write_json(request_path.parent / "status.json", {"state": "complete"})


if __name__ == "__main__":
    main(sys.argv[1])
