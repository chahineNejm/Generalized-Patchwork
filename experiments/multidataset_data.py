"""Approved dataset policies, immutable preparation and verified reuse."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from experiments.multidataset_benchmark import prepare_split, write_json
from pwgp.data.benchmarks import load_real, synthetic_dataset, verify_flight_npz
from pwgp.data.download import file_checksum
from pwgp.data.manager import ICETHICK_MD5
from pwgp.run_records import record_failures


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    dimension: int
    train_fraction: float | None
    validation_fraction: float
    seed: int
    input_path: str | None = None
    day: str | None = None
    bounds: tuple[float, float, float, float] | None = None
    synthetic: dict | None = None


def verify_manual_inputs(specs):
    missing = []
    for spec in specs:
        if spec.kind not in ("flight", "icethick"):
            continue
        if spec.input_path is None or not Path(spec.input_path).is_file():
            missing.append(f"{spec.name}: set input_path to the attached {spec.kind} input file")
            continue
        if spec.kind == "flight":
            verify_flight_npz(spec.input_path)
            continue
        expected = ICETHICK_MD5
        actual = file_checksum(Path(spec.input_path), "md5")
        if actual != expected:
            raise ValueError(f"{spec.name}: input checksum mismatch; expected {expected}, actual {actual}")
    if missing:
        raise FileNotFoundError("Required inputs are not attached; no data substitutions:\n" + "\n".join(missing))


def acquire_split(spec, *, data_root):
    root = Path(data_root)
    if spec.kind == "modis" and (spec.train_fraction is not None or spec.validation_fraction != 0):
        raise ValueError("MODIS uses the official split; train_fraction must be None and validation_fraction zero")
    if spec.kind not in ("oisst", "ukhousing", "modis") and spec.validation_fraction != 0:
        raise ValueError("This approved train/test adapter requires validation_fraction=0; no value is ignored")
    if spec.kind == "oisst":
        from experiments.oisst_benchmark import OISSTConfig, acquire

        config = OISSTConfig(spec.day, spec.bounds, spec.seed, spec.train_fraction,
                             spec.validation_fraction, True)
        split, manifest = acquire(config, data_root=root)
        split["feature_names"], split["target_name"] = ["longitude", "latitude"], "temperature (C)"
    elif spec.kind == "ukhousing":
        from experiments.ukhousing_benchmark import prepare_split as prepare_uk
        from pwgp.data.ukhousing import SOURCE_SNAPSHOT, prepare_ukhousing

        x, y, manifest = prepare_ukhousing(root / f"ukhousing_{SOURCE_SNAPSHOT}")
        split = prepare_uk(x, y, seed=spec.seed, train_fraction=spec.train_fraction,
                           validation_fraction=spec.validation_fraction)
        split["feature_names"], split["target_name"] = ["longitude", "latitude"], "log GBP"
    elif spec.kind == "modis":
        from experiments.heaton_benchmark import SHA256, SOURCE, load_heaton, prepare_data
        from pwgp.data.ukhousing import acquire

        path = root / "modis/AllSatelliteTemps.RData"
        acquire(SOURCE, path, SHA256)
        coordinates, truth, training, testing = load_heaton(path)
        x, _, trend, scale, preprocessing = prepare_data(coordinates, truth, training)
        used = np.flatnonzero(training | testing)
        training_rows = np.flatnonzero(training[used])
        test_rows = np.flatnonzero(testing[used])
        split = {"x": x[used], "y": torch.tensor((truth[used] - trend[used]) / scale, dtype=torch.float64),
            "raw_x": coordinates[used], "raw_y": truth[used], "train": training_rows,
            "validation": np.empty(0, dtype=np.int64), "test": test_rows,
            "target_mean": 0., "target_std": scale, "target_offset": trend[used],
            "feature_names": ["longitude", "latitude"], "target_name": "temperature (C)",
            "score_splits": ["test"], "seed": spec.seed, "preprocessing": preprocessing}
        manifest = {"dataset": "Heaton MODIS official split", "source_urls": [SOURCE],
                    "source_sha256": SHA256, "source_row_indices": "Stored in source_rows.npy"}
        split["source_rows"] = used
    else:
        if spec.kind in ("michalewicz", "g_function"):
            if spec.synthetic is None:
                raise ValueError("Synthetic design must be configured explicitly")
            dataset = synthetic_dataset(spec.kind, dimension=spec.dimension, seed=spec.seed, **spec.synthetic)
        else:
            dataset = load_real(spec.kind, cache=root / spec.kind, input_path=spec.input_path)
        split = prepare_split(dataset, train_fraction=spec.train_fraction, seed=spec.seed)
        if spec.synthetic is not None and (len(split["train"]) != spec.synthetic["training_points"]
                or len(split["test"]) != spec.synthetic["test_points"]):
            raise ValueError("Synthetic requested counts do not match the declared split fraction")
        manifest = {"dataset": spec.name, "metadata": dataset.metadata,
            "feature_names": list(dataset.feature_names), "target_name": dataset.target_name,
            "retained_rows": len(dataset.y), "files": [{"path": str(path), "sha256": file_checksum(path),
                "bytes": path.stat().st_size} for path in dataset.source_files]}
    if spec.kind in ("oisst", "ukhousing"):
        split["target_offset"] = np.full(len(split["raw_y"]), split["target_mean"])
        split["score_splits"] = ["validation", "test"]
        split["seed"] = spec.seed
        split["preprocessing"] = {key: split[key] for key in
            ("target_mean", "target_std", "input_transform", "target_transform")}
    if split["x"].shape != (len(split["y"]), spec.dimension):
        raise ValueError(f"{spec.name}: source dimensions differ from the declared schema")
    return split, manifest


def prepare_dataset(spec, *, directory, data_root):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with record_failures(directory, status_key="state"):
        verify_manual_inputs([spec])
        split, manifest = acquire_split(spec, data_root=data_root)
        from experiments.exploratory_analysis import explore_multidataset

        explore_multidataset(split, directory=directory / "visuals")
        arrays = {key: split[key].numpy() if isinstance(split[key], torch.Tensor) else split[key]
                  for key in ("x", "y", "raw_x", "raw_y", "train", "validation", "test", "target_offset")}
        np.savez_compressed(directory / "data.npz", **arrays)
        np.savez_compressed(directory / "split.npz", **{key: split[key] for key in ("train", "validation", "test")})
        if spec.kind == "modis":
            np.save(directory / "source_rows.npy", split["source_rows"])
            manifest["source_row_indices_sha256"] = file_checksum(directory / "source_rows.npy")
        metadata = {key: split[key] for key in ("target_mean", "target_std", "feature_names", "target_name",
                                               "score_splits", "seed", "preprocessing")}
        write_json(directory / "metadata.json", metadata)
        write_json(directory / "data_manifest.json", manifest)
        write_json(directory / "preprocessing.json", split["preprocessing"])
        names = ["data.npz", "split.npz", "metadata.json", "data_manifest.json", "preprocessing.json"]
        if spec.kind == "modis":
            names.append("source_rows.npy")
        write_json(directory / "prepared.json", {"data": asdict(spec),
            "checksums": {name: file_checksum(directory / name) for name in names}})
        write_json(directory / "status.json", {"state": "complete"})
    return directory


def load_prepared(spec, directory):
    directory = Path(directory)
    if json.loads((directory / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("Prepared dataset did not complete successfully")
    prepared = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    if prepared["data"] != json.loads(json.dumps(asdict(spec))):
        raise ValueError("Prepared dataset configuration identity mismatch; no automatic rebuilding")
    for name, checksum in prepared["checksums"].items():
        if file_checksum(directory / name) != checksum:
            raise ValueError(f"Prepared dataset checksum mismatch: {name}")
    with np.load(directory / "data.npz", allow_pickle=False) as arrays:
        split = {key: arrays[key] for key in arrays.files}
    with np.load(directory / "split.npz", allow_pickle=False) as arrays:
        for key in ("train", "validation", "test"):
            if not np.array_equal(split[key], arrays[key]):
                raise ValueError(f"Prepared split identity mismatch: {key}")
    split["x"], split["y"] = torch.from_numpy(split["x"]), torch.from_numpy(split["y"])
    split.update(json.loads((directory / "metadata.json").read_text(encoding="utf-8")))
    return split
