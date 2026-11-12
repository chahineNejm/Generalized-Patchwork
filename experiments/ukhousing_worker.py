"""Fresh-worker UKHousing comparison with verified prepared data and source."""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.oisst_worker import emit, source_snapshot
from pwgp.data.download import file_checksum
from pwgp.run_records import record_failures


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_prepared(directory, data):
    from experiments.ukhousing_benchmark import prepare_split

    directory = Path(directory)
    if json.loads((directory / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("Prepared UKHousing stage has not completed successfully")
    metadata = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    if metadata["data"] != data:
        raise ValueError("Prepared UKHousing configuration differs; explicitly prepare new data")
    for name, checksum in metadata["checksums"].items():
        if file_checksum(directory / name) != checksum:
            raise ValueError(f"Prepared UKHousing checksum mismatch: {name}")
    with np.load(directory / "data.npz", allow_pickle=False) as arrays:
        split = prepare_split(arrays["x"], arrays["y"], **data)
    with np.load(directory / "split.npz", allow_pickle=False) as arrays:
        for key in ("train", "validation", "test"):
            if not np.array_equal(arrays[key], split[key]):
                raise ValueError(f"Prepared UKHousing split identity mismatch: {key}")
    return split


def execute(request):
    from experiments.default_hyperparameters import ukhousing_comparison_defaults
    from experiments.run_selection import select_runs
    from experiments.ukhousing_benchmark import UKConfig, archive_run, prepare_split, run_ukhousing
    from pwgp.config import ProSparConfig
    from pwgp.data.ukhousing import prepare_ukhousing

    root, source = source_snapshot(request["source_provenance"])
    action = request["action"]
    if action == "settings":
        runs = select_runs(ukhousing_comparison_defaults(device=request["device"]),
                           models=request["models"], overrides=request["overrides"])
        return {"runs": [{"name": name, "config": asdict(config)} for name, config in runs]}
    directory = Path(request["directory"])
    if action == "prepare":
        directory.mkdir(parents=True, exist_ok=False)
        with record_failures(directory, status_key="state"):
            x, y, manifest = prepare_ukhousing(root / "data/ukhousing")
            split = prepare_split(x, y, **request["data"])
            np.savez_compressed(directory / "data.npz", x=x, y=y)
            np.savez_compressed(directory / "split.npz", **{key: split[key]
                                for key in ("train", "validation", "test")})
            write_json(directory / "data_manifest.json", manifest)
            write_json(directory / "preprocessing.json", {key: split[key] for key in
                       ("target_mean", "target_std", "seed", "input_transform", "target_transform")})
            write_json(directory / "prepared.json", {"data": request["data"], "checksums": {
                name: file_checksum(directory / name) for name in
                ("data.npz", "split.npz", "data_manifest.json", "preprocessing.json")}})
            write_json(directory / "status.json", {"state": "complete"})
    elif action == "train":
        split = load_prepared(request["prepared_directory"], request["data"])
        directory.mkdir(parents=True, exist_ok=False)
        with record_failures(directory, status_key="state"):
            (directory / "source_snapshot.zip").write_bytes(source)
            write_json(directory / "source_provenance.json", request["source_provenance"])
            write_json(directory / "runs.json", request["runs"])
            prepared = Path(request["prepared_directory"])
            for name in ("split.npz", "data_manifest.json", "preprocessing.json", "prepared.json"):
                (directory / name).write_bytes((prepared / name).read_bytes())
            summaries = []
            if not request["runs"] or len({row["name"] for row in request["runs"]}) != len(request["runs"]):
                raise ValueError("UKHousing comparison requires nonempty, uniquely named runs")
            for index, row in enumerate(tqdm(request["runs"], desc="compare:UKHousing", unit="model")):
                values = dict(row["config"])
                if values["prospar"] is not None:
                    values["prospar"] = ProSparConfig(**values["prospar"])
                config = UKConfig(**values)
                result = run_ukhousing(config, split, directory=directory / f"run_{index:02d}",
                    on_snapshot=lambda path, epoch: emit("snapshot", directory=str(path), epoch=epoch))
                metrics = json.loads((result / "metrics.json").read_text(encoding="utf-8"))
                summaries.append({"model": row["name"], "directory": str(result), **metrics})
                pd.DataFrame(summaries).to_csv(directory / "comparison.csv", index=False)
                archive = archive_run(result)
                emit("complete", name=row["name"], directory=str(result), archive=str(archive), metrics=metrics)
            write_json(directory / "status.json", {"state": "complete"})
    else:
        raise ValueError(f"Unknown UKHousing worker action: {action}")
    return {"directory": str(directory.resolve())}


def main(request_path):
    request_path = Path(request_path)
    with record_failures(request_path.parent, status_key="state"):
        request = json.loads(request_path.read_text(encoding="utf-8"))
        write_json(request_path.parent / "result.json", {**execute(request), "worker_pid": os.getpid()})
        write_json(request_path.parent / "status.json", {"state": "complete"})


if __name__ == "__main__":
    main(sys.argv[1])
