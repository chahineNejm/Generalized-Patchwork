"""Single-dataset CPU sessions with explicitly requested, verified continuation."""

import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import zipfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import faiss
import gpytorch
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from experiments.default_hyperparameters import multidataset_defaults, vnngp_cpu_defaults
from experiments.exploratory_analysis import plot_training
from experiments.multidataset_benchmark import plot_predictions, write_json
from experiments.multidataset_data import DatasetSpec, load_prepared, prepare_dataset
from experiments.multidataset_worker import plan_hash
from experiments.oisst_benchmark import learned_parameters
from experiments.oisst_worker import emit, source_snapshot
from experiments.ukhousing_benchmark import UKConfig, build_model, scores
from experiments.vnngp_cpu_training import make_stopper, train_chunk
from pwgp.config import HyperparameterBounds, NumericalConfig
from pwgp.data.download import file_checksum
from pwgp.run_records import record_failures


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def environment():
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "gpytorch": gpytorch.__version__, "numpy": np.__version__,
            "faiss": importlib.metadata.version("faiss-cpu"),
            "package_version": importlib.metadata.version("generalized-patchwork-gp"),
            "system": platform.platform(), "processor": platform.processor()}


def make_plan(request):
    data, _, _ = multidataset_defaults(device="cpu", input_root=request["input_root"])
    name, seeds = request["dataset"], request["seeds"]
    if name not in data:
        raise ValueError(f"Select exactly one known dataset: {list(data)}")
    if (not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)):
        raise ValueError("SEEDS must contain distinct integers in [0, 2**32)")
    spec = replace(data[name], **request["data_overrides"])
    if (spec.name, spec.kind, spec.dimension) != (data[name].name, data[name].kind, data[name].dimension):
        raise ValueError("Dataset overrides cannot relabel the dataset or its schema")
    base = vnngp_cpu_defaults()
    if set(request["overrides"]) - (set(base) - {"seeds"}):
        raise ValueError("Unknown CPU notebook override")
    base["seeds"] = seeds
    base.update(request["overrides"])
    configs = [asdict(UKConfig(model="vnngp", device="cpu", seed=seed,
        epochs=base["epochs"], warmup_epochs=base["warmup_epochs"],
        vnngp_neighbors=base["neighbors"], vnngp_batch_size=base["batch_size"],
        lengthscale=base["lengthscale"], outputscale=base["outputscale"], noise_std=base["noise_std"],
        learning_rate_q=base["learning_rate_q"], learning_rate_hyperparameters=base["learning_rate_hyperparameters"],
        prediction_batch_size=base["prediction_batch_size"], threads=base["threads"])) for seed in seeds]
    for config in configs:
        if (type(config["epochs"]) is not int or type(config["warmup_epochs"]) is not int
                or not 0 <= config["warmup_epochs"] < config["epochs"]):
            raise ValueError("Require integer epochs and 0 <= warm-up < epochs")
        for key in ("vnngp_neighbors", "vnngp_batch_size", "prediction_batch_size", "threads"):
            if type(config[key]) is not int or config[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("learning_rate_q", "learning_rate_hyperparameters"):
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        bounds = HyperparameterBounds()
        for key, limits in (("lengthscale", bounds.lengthscale), ("outputscale", bounds.outputscale),
                            ("noise_std", bounds.noise_std)):
            if not np.isfinite(config[key]) or not limits[0] < config[key] < limits[1]:
                raise ValueError(f"{key} must be strictly inside {limits}")
        make_stopper(base["early_stopping"], config["warmup_epochs"])
    if not np.isfinite(base["session_hours"]) or not 0 < base["session_hours"] <= 10:
        raise ValueError("session_hours must be positive and at most the approved 10 hours")
    if not np.isfinite(base["checkpoint_seconds"]) or base["checkpoint_seconds"] <= 0:
        raise ValueError("checkpoint_seconds must be finite and positive")
    if type(base["progress_every_epochs"]) is not int or base["progress_every_epochs"] < 1:
        raise ValueError("progress_every_epochs must be a positive integer")
    plan = {"data": asdict(spec), "configs": configs, "settings": base,
            "kernel": "ARD RBF", "inducing_order": "prepared training order (existing benchmark policy)",
            "initial_noise_variance": base["noise_std"]**2, "scheduler": "none; existing constant rates"}
    return {"plan": plan, "plan_sha256": plan_hash(plan), "total_runs": len(configs),
            "maximum_epochs": sum(c["epochs"] for c in configs)}


def archive_session(directory, destination):
    directory, destination = Path(directory), Path(destination)
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.name != "resume_manifest.json")
    write_json(directory / "resume_manifest.json", {
        p.relative_to(directory).as_posix(): file_checksum(p) for p in files})
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, directory / "resume_manifest.json"]:
            archive.write(path, path.relative_to(directory))
    return str(destination)


def restore_archive(archive, directory):
    """Extract only the explicit archive, then verify every file before loading tensors."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(set(names)) != len(names):
            raise ValueError("Duplicate archive entries")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError(f"Unsafe resume archive entry: {name}")
        if bundle.testzip() is not None:
            raise ValueError("Resume ZIP CRC failure")
        bundle.extractall(directory)
    manifest = read_json(directory / "resume_manifest.json")
    files = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if files != set(manifest) | {"resume_manifest.json"}:
        raise ValueError("Resume archive file inventory mismatch")
    for name, digest in manifest.items():
        if file_checksum(directory / name) != digest:
            raise ValueError(f"Resume archive checksum mismatch: {name}")
    return directory


def evaluate(model, split, config, output, deadline):
    """Score the last trained model; pause between prediction batches if necessary."""
    result = {}
    for label in split["score_splits"]:
        target = output / f"{label}_predictions.npz"
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                mean, variance = saved["standardized_mean"], saved["standardized_variance"]
        else:
            means, variances = [], []
            indices = split[label]
            with torch.no_grad():
                for batch in tqdm(split["x"][indices].split(config.prediction_batch_size),
                                  desc=f"predict:{label}/VNNGP/seed{config.seed}", unit="batch"):
                    if time.monotonic() >= deadline:
                        return False
                    mean, variance = model.predict(batch, observation_noise=True)
                    means.append(mean.numpy())
                    variances.append(variance.numpy())
            mean, variance = np.concatenate(means), np.concatenate(variances)
            np.savez_compressed(target, indices=indices, inputs=split["raw_x"][indices],
                truth=split["raw_y"][indices], standardized_mean=mean, standardized_variance=variance,
                mean=mean * split["target_std"] + split["target_offset"][indices],
                observation_variance=variance * split["target_std"]**2)
        indices = split[label]
        raw_mean = mean * split["target_std"] + split["target_offset"][indices]
        raw_variance = variance * split["target_std"]**2
        result[label] = {"standardized": scores(split["y"][indices].numpy(), mean, variance),
                         "original_target": scores(split["raw_y"][indices], raw_mean, raw_variance)}
        if label == "test":
            plot_predictions(split["raw_y"][indices], raw_mean, raw_variance,
                             destination=output / "visuals/predictions.png")
    write_json(output / "all_scores.json", result)
    write_json(output / "metrics.json", {f"{key}_{scale}": value
        for scale, values in result["test"].items() for key, value in values.items()})
    return True


def run_session(directory, plan, *, source_provenance, on_complete=None):
    directory = Path(directory)
    if read_json(directory / "plan.json") != plan:
        raise ValueError("Resume plan changed; start a separate experiment instead")
    if read_json(directory / "source_provenance.json") != source_provenance:
        raise ValueError("Resume requires the original source commit/archive; do not migrate silently")
    saved_environment, current = read_json(directory / "environment.json"), environment()
    for key in ("python", "torch", "gpytorch", "numpy", "faiss", "package_version"):
        if saved_environment[key] != current[key]:
            raise ValueError(f"Resume dependency mismatch: {key}")
    progress = read_json(directory / "progress.json")
    if progress["state"] == "complete":
        raise ValueError("All requested seeds are already complete; no implicit retraining")
    split = load_prepared(DatasetSpec(**plan["data"]), directory / "prepared")
    settings = plan["settings"]
    deadline = time.monotonic() + settings["session_hours"] * 3600
    identity = {"plan_sha256": plan_hash(plan), "source": source_provenance,
                "prepared_sha256": file_checksum(directory / "prepared/prepared.json")}
    sessions_path = directory / "sessions.json"
    sessions = read_json(sessions_path) if sessions_path.exists() else []
    sessions.append({"started_utc": datetime.now(UTC).isoformat(), "environment": current,
                     "starting_seed_index": progress["next_seed_index"]})
    write_json(sessions_path, sessions)
    with record_failures(directory, status_key="state"):
        while progress["next_seed_index"] < len(plan["configs"]):
            config = UKConfig(**plan["configs"][progress["next_seed_index"]])
            torch.set_num_threads(config.threads)
            faiss.omp_set_num_threads(config.threads)
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)
            random.seed(config.seed)
            output = directory / f"seed{config.seed}"
            output.mkdir(exist_ok=True)
            (output / "visuals").mkdir(exist_ok=True)
            write_json(output / "config.json", asdict(config))
            model = build_model(config, split)
            write_json(output / "model_metadata.json", {"class": f"{type(model).__module__}.{type(model).__name__}",
                "inducing_points": len(split["train"]), "locations": "fixed prepared training rows",
                "kernel": plan["kernel"], "bounds": asdict(HyperparameterBounds()),
                "numerical": asdict(NumericalConfig()), "gradient_clip": 10.0,
                "dtype": "float64", "hardware": current, "model_seed": config.seed,
                "data_seed": split["seed"], "early_stopping": settings["early_stopping"]})
            checkpoint = output / "checkpoint.pt"
            if not checkpoint.exists():
                np.save(output / "inducing_positions.npy", split["x"][split["train"]].numpy())
            else:
                saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
                history_path = output / "hyperparameter_history.json"
                if history_path.exists():
                    write_json(history_path, [row for row in read_json(history_path)
                                              if row["epoch"] <= len(saved["losses"])])

            def snapshot(epoch, losses, current_model, *, output=output, snapshot_every=config.snapshot_every):
                write_json(output / "history.json", {"losses": losses})
                write_json(output / "learned_parameters.json", learned_parameters(current_model, "vnngp"))
                history_path = output / "hyperparameter_history.json"
                history = read_json(history_path) if history_path.exists() else []
                history = [row for row in history if row["epoch"] < epoch]
                history.append({"epoch": epoch, **learned_parameters(current_model, "vnngp")})
                write_json(history_path, history)
                if epoch % snapshot_every == 0:
                    plot_training(losses, destination=output / "visuals/training.png")

            complete = train_chunk(model, split["y"][split["train"]], config,
                stopper=make_stopper(settings["early_stopping"], config.warmup_epochs),
                checkpoint=checkpoint, identity=identity, resume=checkpoint.exists(), deadline=deadline,
                checkpoint_seconds=settings["checkpoint_seconds"],
                description=f"train:{plan['data']['name']}/VNNGP K{config.vnngp_neighbors}/seed{config.seed}",
                progress_every_epochs=settings["progress_every_epochs"], on_epoch=snapshot)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            write_json(output / "history.json", {"losses": state["losses"]})
            write_json(output / "learned_parameters.json", learned_parameters(model, "vnngp"))
            plot_training(state["losses"], destination=output / "visuals/training.png")
            write_json(output / "training_summary.json", {"epochs_completed": len(state["losses"]),
                "maximum_epochs": config.epochs, "next_minibatch": state["batch"],
                "training_seconds": state["training_seconds"], "evaluated_model": "last trained state",
                "training_complete": complete, "early_stopping": state["stopper"],
                "reason": "training_loss_patience" if state["stopper"]["stopped"] else
                "maximum_epochs" if complete else "session_time_limit"})
            if not complete or not evaluate(model, split, config, output, deadline):
                progress["state"] = "paused"
                write_json(output / "status.json", {"state": "paused", "phase": "prediction" if complete else "training"})
                break
            metrics = read_json(output / "metrics.json")
            metrics.update(training_seconds=state["training_seconds"], final_loss=state["losses"][-1],
                           epochs_completed=len(state["losses"]), test_points=len(split["test"]))
            write_json(output / "metrics.json", metrics)
            torch.save(model.state_dict(), output / "model.pt")
            write_json(output / "status.json", {"state": "complete"})
            progress["next_seed_index"] += 1
            progress["state"] = "complete" if progress["next_seed_index"] == len(plan["configs"]) else "paused"
            write_json(directory / "progress.json", progress)
            write_json(directory / "status.json", {"state": progress["state"]})
            pd.DataFrame([{"dataset": plan["data"]["name"], "model": "VNNGP", "seed": values["seed"],
                "neighbors": values["vnngp_neighbors"],
                **read_json(directory / f"seed{values['seed']}" / "metrics.json")}
                for values in plan["configs"][:progress["next_seed_index"]]]).to_csv(
                    directory / "comparison.csv", index=False)
            destination = directory.parent / f"{directory.name}_seed{config.seed}.zip"
            archive_session(directory, destination)
            print(f"Completed {plan['data']['name']} / VNNGP / seed{config.seed}", flush=True)
            if on_complete is not None:
                on_complete(output, destination)
            if time.monotonic() >= deadline:
                break
        write_json(directory / "progress.json", progress)
        write_json(directory / "status.json", {"state": progress["state"]})
    return progress


def execute(request):
    _, source = source_snapshot(request["source_provenance"])
    if request["action"] == "settings":
        return make_plan(request)
    if request["execute_training"] is not True:
        raise RuntimeError("Set RUN_TRAINING=True explicitly after reviewing the plan")
    plan = request["plan"]
    if request["plan_sha256"] != plan_hash(plan):
        raise ValueError("Plan checksum mismatch")
    directory = Path(request["directory"])
    if request["action"] == "start":
        directory.mkdir(parents=True, exist_ok=False)
        with record_failures(directory, status_key="state"):
            write_json(directory / "plan.json", plan)
            write_json(directory / "source_provenance.json", request["source_provenance"])
            (directory / "source_snapshot.zip").write_bytes(source)
            write_json(directory / "environment.json", environment())
            write_json(directory / "progress.json", {"state": "paused", "next_seed_index": 0})
            prepare_dataset(DatasetSpec(**plan["data"]), directory=directory / "prepared",
                            data_root=request["data_root"])
    elif request["action"] == "resume":
        restore_archive(request["resume_archive"], directory)
    else:
        raise ValueError("Select start or resume explicitly")
    progress = run_session(directory, plan, source_provenance=request["source_provenance"],
        on_complete=lambda output, archive: emit("complete", directory=str(output), archive=str(archive),
                                               metrics=read_json(output / "metrics.json")))
    archive = archive_session(directory, request["archive"])
    return {"directory": str(directory), "archive": archive, **progress}


def main(path):
    path = Path(path)
    with record_failures(path.parent, status_key="state"):
        result = execute(read_json(path))
        write_json(path.parent / "result.json", {**result, "worker_pid": os.getpid()})


if __name__ == "__main__":
    main(sys.argv[1])
