"""Approved dataset policies, immutable preparation and verified reuse."""

import json
from dataclasses import asdict, dataclass, replace
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
    subset_size: int | None = None
    simulation: dict | None = None


HEAT_SIMULATION_KEYS = {"cells", "diffusivity", "angles", "periods", "baseline", "amplitude",
                        "time", "time_steps", "relative_noise_std"}
HEAT_REGION_KEYS = {"gap_width", "gap_side", "near_interface_width"}
HEAT_INTERFACE_KEYS = {"interface_amplitude", "interface_teeth"}
REGION_SPLITS = ("test_gap", "test_near_interface", "test_interior")


def spec_record(spec):
    """Preserve existing identities when no subset or simulation is requested."""
    record = asdict(spec)
    for key in ("subset_size", "simulation"):
        if getattr(spec, key) is None:
            del record[key]
    return json.loads(json.dumps(record))


def periodic_heat_dataset(simulation, *, seed):
    """Periodic-heat FEM field on its full grid, with seeded Gaussian observation noise."""
    from experiments.periodic_heat_preview import PreviewConfig, simulate
    from pwgp.data.types import RegressionDataset

    optional = HEAT_REGION_KEYS | HEAT_INTERFACE_KEYS
    if not isinstance(simulation, dict) or set(simulation) - optional != HEAT_SIMULATION_KEYS:
        raise ValueError(f"periodic_heat simulation requires exactly {sorted(HEAT_SIMULATION_KEYS)}"
                         f" plus optionally all of {sorted(HEAT_REGION_KEYS)} and/or {sorted(HEAT_INTERFACE_KEYS)}")
    for group in (HEAT_REGION_KEYS, HEAT_INTERFACE_KEYS):
        if set(simulation) & group and not group <= set(simulation):
            raise ValueError(f"These settings must be given together: all of {sorted(group)}")
    noise = simulation["relative_noise_std"]
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("relative_noise_std must be finite and nonnegative")
    angles = tuple(simulation["angles"])
    config = PreviewConfig(cells=simulation["cells"], diffusivity=tuple(simulation["diffusivity"]),
        angle_pairs=(angles,), periods=tuple(simulation["periods"]), baseline=simulation["baseline"],
        amplitude=simulation["amplitude"], time=simulation["time"], time_steps=simulation["time_steps"],
        **{key: simulation[key] for key in HEAT_INTERFACE_KEYS if key in simulation})
    xy, _, temperature, tensors = simulate(config, angles)
    scale = noise * float(temperature.std())
    y = temperature + np.random.default_rng(seed).normal(scale=scale, size=len(temperature))
    return RegressionDataset("periodic_heat", xy, y, ("x", "y"), "temperature", metadata={
        "source": "experiments.periodic_heat_preview.simulate (P1 FEM, implicit Euler)",
        "simulation": simulation, "grid_points": len(y), "noise_std": scale, "seed": seed,
        "noise_definition": "relative_noise_std x SD of the noiseless field over all grid points",
        "material_tensors": tensors.tolist()})


def flight_subset(dataset, *, size, seed):
    """Uniform observed-row sampling without replacement, before preprocessing."""
    if type(size) is not int or not 1 < size <= len(dataset.y):
        raise ValueError("FLIGHT subset_size must be an integer in [2, source row count]")
    rows = np.random.default_rng(seed).choice(len(dataset.y), size=size, replace=False)
    return replace(dataset, x=dataset.x[rows], y=dataset.y[rows], metadata={**dataset.metadata,
        "subset": {"size": size, "source_rows": len(dataset.y), "seed": seed,
            "sampling": "numpy.default_rng.choice without replacement before preprocessing/split",
            "reference_status": "size-matched only; not verified paper records"}}), rows


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


def interface_gap_split(split, settings):
    """Move training rows in a band beside x=0.5 to test; label test regions; rescale targets."""
    width, side, near = settings["gap_width"], settings["gap_side"], settings["near_interface_width"]
    if not 0 < width < 0.5 or side not in ("right", "left") or not 0 < near < 0.5:
        raise ValueError("Require 0 < gap_width < 0.5, gap_side right/left and 0 < near_interface_width < 0.5")
    x = np.asarray(split["raw_x"], dtype=np.float64)[:, 0]
    offset = x - 0.5 if side == "right" else 0.5 - x
    in_gap = (offset >= 0) & (offset < width)
    near_mask = ~in_gap & (np.abs(x - 0.5) < near)
    train = split["train"][~in_gap[split["train"]]]
    test = np.concatenate((split["test"], split["train"][in_gap[split["train"]]]))
    regions = {"test_gap": test[in_gap[test]], "test_near_interface": test[near_mask[test]],
               "test_interior": test[~in_gap[test] & ~near_mask[test]]}
    if any(len(rows) == 0 for rows in (train, *regions.values())):
        raise ValueError("Interface-gap split requires nonempty training rows and test regions")
    center, scale = float(split["raw_y"][train].mean()), float(split["raw_y"][train].std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training target standard deviation must be positive and finite")
    preprocessing = {**split["preprocessing"], "target_mean": center, "target_std": scale,
        "target_statistics_fit_on": "training rows outside the interface gap",
        "interface_gap": {"width": width, "side": side, "near_interface_width": near,
            "moved_training_rows": int(in_gap[split["train"]].sum()),
            "region_counts": {key: len(rows) for key, rows in regions.items()}}}
    return {**split, **regions, "train": train, "test": test,
            "y": torch.tensor((split["raw_y"] - center) / scale, dtype=torch.float64),
            "target_mean": center, "target_std": scale, "target_offset": np.full(len(split["raw_y"]), center),
            "score_splits": ["test", *REGION_SPLITS], "preprocessing": preprocessing}


def acquire_split(spec, *, data_root):
    root = Path(data_root)
    if spec.subset_size is not None and spec.kind != "flight":
        raise ValueError("Explicit subset_size is supported only for FLIGHT")
    if (spec.simulation is not None) != (spec.kind == "periodic_heat"):
        raise ValueError("simulation settings are required for, and only for, periodic_heat")
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
    elif spec.kind == "periodic_heat":
        dataset = periodic_heat_dataset(spec.simulation, seed=spec.seed)
        split = prepare_split(dataset, train_fraction=spec.train_fraction, seed=spec.seed)
        if HEAT_REGION_KEYS <= set(spec.simulation):
            split = interface_gap_split(split, spec.simulation)
        manifest = {"dataset": spec.name, "metadata": dataset.metadata,
            "feature_names": list(dataset.feature_names), "target_name": dataset.target_name,
            "retained_rows": len(dataset.y), "files": []}
    else:
        if spec.kind in ("michalewicz", "g_function"):
            if spec.synthetic is None:
                raise ValueError("Synthetic design must be configured explicitly")
            dataset = synthetic_dataset(spec.kind, dimension=spec.dimension, seed=spec.seed, **spec.synthetic)
        else:
            dataset = load_real(spec.kind, cache=root / spec.kind, input_path=spec.input_path)
        if spec.subset_size is not None:
            dataset, source_rows = flight_subset(dataset, size=spec.subset_size, seed=spec.seed)
        split = prepare_split(dataset, train_fraction=spec.train_fraction, seed=spec.seed)
        if spec.subset_size is not None:
            split["source_rows"] = source_rows
        if spec.synthetic is not None and (len(split["train"]) != spec.synthetic["training_points"]
                or len(split["test"]) != spec.synthetic["test_points"]):
            raise ValueError("Synthetic requested counts do not match the declared split fraction")
        manifest = {"dataset": spec.name, "metadata": dataset.metadata,
            "feature_names": list(dataset.feature_names), "target_name": dataset.target_name,
            "retained_rows": len(dataset.y), "files": [{"path": str(path), "sha256": file_checksum(path),
                "bytes": path.stat().st_size} for path in dataset.source_files]}
    if spec.kind in ("oisst", "ukhousing"):
        split["target_offset"] = np.full(len(split["raw_y"]), split["target_mean"])
        split["score_splits"] = ["validation", "test"] if len(split["validation"]) else ["test"]
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
                  for key in ("x", "y", "raw_x", "raw_y", "train", "validation", "test", "target_offset",
                              *REGION_SPLITS) if key in split}
        np.savez_compressed(directory / "data.npz", **arrays)
        np.savez_compressed(directory / "split.npz", **{key: split[key] for key in ("train", "validation", "test")})
        if "source_rows" in split:
            np.save(directory / "source_rows.npy", split["source_rows"])
            manifest["source_row_indices_sha256"] = file_checksum(directory / "source_rows.npy")
        metadata = {key: split[key] for key in ("target_mean", "target_std", "feature_names", "target_name",
                                               "score_splits", "seed", "preprocessing")}
        write_json(directory / "metadata.json", metadata)
        write_json(directory / "data_manifest.json", manifest)
        write_json(directory / "preprocessing.json", split["preprocessing"])
        names = ["data.npz", "split.npz", "metadata.json", "data_manifest.json", "preprocessing.json"]
        if "source_rows" in split:
            names.append("source_rows.npy")
        write_json(directory / "prepared.json", {"data": spec_record(spec),
            "checksums": {name: file_checksum(directory / name) for name in names}})
        write_json(directory / "status.json", {"state": "complete"})
    return directory


def load_prepared(spec, directory):
    directory = Path(directory)
    if json.loads((directory / "status.json").read_text(encoding="utf-8"))["state"] != "complete":
        raise ValueError("Prepared dataset did not complete successfully")
    prepared = json.loads((directory / "prepared.json").read_text(encoding="utf-8"))
    if prepared["data"] != spec_record(spec):
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
    if spec.kind == "modis" or spec.subset_size is not None:
        split["source_rows"] = np.load(directory / "source_rows.npy", allow_pickle=False)
    split.update(json.loads((directory / "metadata.json").read_text(encoding="utf-8")))
    return split
