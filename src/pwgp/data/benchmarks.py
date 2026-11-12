"""Exact, declared schemas for the approved multi-dataset sources."""

import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import qmc

from pwgp.data.download import file_checksum
from pwgp.data.loaders import load_flight
from pwgp.data.manager import ICETHICK_MD5, PATCHWORK_BUNDLE_DRIVE_ID
from pwgp.data.types import RegressionDataset
from pwgp.data.ukhousing import acquire

SOURCES = {
    "california": ("https://ndownloader.figshare.com/files/5976036", "cal_housing.tgz",
                   "aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681"),
    "protein": ("https://archive.ics.uci.edu/static/public/265/physicochemical%2Bproperties%2Bof%2Bprotein%2Btertiary%2Bstructure.zip",
                "protein.zip", "ee6536c8cc415dc50d66bd247af7c9bc95c187ab8a4ea0f1f546b127206c1100"),
    "sarcos_train": ("https://gaussianprocess.org/gpml/data/sarcos_inv.mat", "sarcos_inv.mat",
                     "b8a249733253ba6097372fedee7696833fcf30de42037d5b4a7227f21a6d1d97"),
    "sarcos_test": ("https://gaussianprocess.org/gpml/data/sarcos_inv_test.mat", "sarcos_inv_test.mat",
                    "161a59b5c3b4f4b404584323f181607b2acbe620eb134dc720760dc3f38f5cec"),
}
PATCHWORK_SHA256 = "60a3777349cf942d87f5dbee8fbad44b86d0aefa1753538df56749b88312f1cc"
FLIGHT_SHA256 = "f111b35372eb2798cd43ab186203d5bb8733653260faa65c6307c87a933ef083"
FLIGHT_NPZ_SHA256 = "90ba687d9b4c7fef8d39cd76572a5a46df4b0232cac9fc0e96664e2c4aff5b3a"
FLIGHT_PROVENANCE_SHA256 = "0a2ae27c3d8681ffd572ec714ed28ffbe7213addd349f97e71f3c6f52754c4be"


def verify_flight_npz(path):
    """Accept only the explicitly converted, checksummed trusted author copy."""
    path = Path(path)
    provenance = path.with_suffix(".json")
    for file, expected in ((path, FLIGHT_NPZ_SHA256), (provenance, FLIGHT_PROVENANCE_SHA256)):
        if not file.is_file() or file_checksum(file) != expected:
            raise ValueError(f"FLIGHT conversion checksum mismatch or missing file: {file}; expected {expected}")
    metadata = json.loads(provenance.read_text(encoding="utf-8"))
    if metadata["source_sha256"] != FLIGHT_SHA256 or metadata["output_sha256"] != FLIGHT_NPZ_SHA256:
        raise ValueError("FLIGHT conversion provenance does not identify the approved input/output")
    return metadata


def source_file(key, cache):
    url, name, checksum = SOURCES[key]
    path = Path(cache) / name
    acquire(url, path, checksum)
    return path


def patchwork_archive(cache):
    import gdown

    path = Path(cache) / "patchwork-authors-bundle.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".zip.part")
        if gdown.download(id=PATCHWORK_BUNDLE_DRIVE_ID, output=str(temporary), quiet=False) is None:
            raise RuntimeError("Patchwork authors' bundle download failed")
        actual = file_checksum(temporary)
        if actual != PATCHWORK_SHA256:
            raise ValueError(f"Patchwork bundle checksum mismatch: expected {PATCHWORK_SHA256}, actual {actual}")
        temporary.rename(path)
    actual = file_checksum(path)
    if actual != PATCHWORK_SHA256:
        raise ValueError(f"Cached Patchwork bundle checksum mismatch: {path}; actual {actual}")
    return path


def load_real(name, *, cache, input_path):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    if name == "california":
        path = source_file(name, cache)
        with tarfile.open(path, "r:gz") as archive, archive.extractfile("CaliforniaHousing/cal_housing.data") as stream:
            raw = np.loadtxt(stream, delimiter=",")
        if raw.shape != (20640, 9) or (raw[:, 6] <= 0).any():
            raise ValueError("Unexpected canonical California schema or nonpositive household counts")
        x = np.column_stack((raw[:, 7], raw[:, 2], raw[:, 3] / raw[:, 6], raw[:, 4] / raw[:, 6],
                             raw[:, 5], raw[:, 5] / raw[:, 6], raw[:, 1], raw[:, 0]))
        y = raw[:, 8] / 100000
        features = ("MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude")
        target, paths = "median house value (100,000 USD)", (path,)
    elif name == "protein":
        path = source_file(name, cache)
        with zipfile.ZipFile(path) as archive:
            if archive.namelist() != ["CASP.csv"]:
                raise ValueError("Unexpected PROTEIN archive members")
            with archive.open("CASP.csv") as stream:
                frame = pd.read_csv(stream)
        columns = ["RMSD", *[f"F{i}" for i in range(1, 10)]]
        if list(frame.columns) != columns or frame.shape != (45730, 10):
            raise ValueError("Unexpected UCI CASP schema")
        x, y = frame[columns[1:]].to_numpy(np.float64), frame["RMSD"].to_numpy(np.float64)
        features, target, paths = tuple(columns[1:]), "RMSD", (path,)
    elif name == "sarcos":
        paths = (source_file("sarcos_train", cache), source_file("sarcos_test", cache))
        arrays = [loadmat(path)[key] for path, key in zip(paths, ("sarcos_inv", "sarcos_inv_test"), strict=True)]
        if [a.shape for a in arrays] != [(44484, 28), (4449, 28)]:
            raise ValueError("Unexpected GPML SARCOS schemas")
        raw = np.concatenate(arrays)
        x, y = raw[:, :21], raw[:, 21]
        features, target = tuple(f"joint_input_{i + 1}" for i in range(21)), "first joint torque"
    elif name in ("tco", "tco_l2"):
        path = patchwork_archive(cache)
        prefix = "ozone" if name == "tco" else "ozone_L2"
        with zipfile.ZipFile(path) as archive:
            with archive.open(f"datasets/{prefix}_x.csv") as stream:
                x = np.loadtxt(stream, delimiter=",")
            with archive.open(f"datasets/{prefix}_y.csv") as stream:
                y = np.loadtxt(stream, delimiter=",")
        expected = 48331 if name == "tco" else 182591
        if x.shape != (expected, 2) or y.shape != (expected,):
            raise ValueError(f"Unexpected authors' {name} schema")
        features = ("latitude", "longitude") if name == "tco" else ("longitude", "latitude")
        target, paths = "total column ozone", (path,)
    elif name == "icethick":
        path = Path(input_path)
        if file_checksum(path, "md5") != ICETHICK_MD5:
            raise ValueError("ICETHICK must match the official icethk.txt MD5")
        frame = pd.read_csv(path, sep="\t")
        columns = ["Season/Flight", "Year", "Day", "Longitude W", "Latitude S", "Thickness (m)"]
        if list(frame.columns) != columns:
            raise ValueError("Unexpected official ICETHICK tab-separated schema")
        numeric = frame[columns[3:]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        x, y = numeric[:, :2], numeric[:, 2]
        features, target, paths = ("longitude", "latitude"), "ice thickness (m)", (path,)
    elif name == "flight":
        path = Path(input_path)
        metadata = verify_flight_npz(path)
        dataset = load_flight(path)
        if len(dataset.y) != metadata["row_counts"]["output_rows"]:
            raise ValueError("FLIGHT NPZ row count differs from conversion provenance")
        return RegressionDataset(name, dataset.x, dataset.y, dataset.feature_names, dataset.target_name,
                                 (path, path.with_suffix(".json")), {**dataset.metadata, "conversion_provenance": metadata})
    else:
        raise ValueError(f"Unsupported declared real dataset: {name}")
    metadata = {"source_schema": name, "reference_snapshot_verified": False,
                "source_urls": [SOURCES[name][0]] if name in ("california", "protein") else []}
    if name == "sarcos":
        metadata["source_urls"] = [SOURCES[key][0] for key in ("sarcos_train", "sarcos_test")]
        metadata["split_policy"] = "Combine the two GPML files, then approved seeded 80/20; not the official split"
    if name in ("tco", "tco_l2"):
        metadata["source_urls"] = [f"https://drive.google.com/uc?id={PATCHWORK_BUNDLE_DRIVE_ID}"]
        metadata["reference_snapshot_verified"] = True
    return RegressionDataset(name, np.asarray(x, np.float64), np.asarray(y, np.float64),
                             features, target, paths, metadata).finite()


def synthetic_dataset(name, *, dimension, training_points, test_points, seed,
                      noise_variance, michalewicz_m, g_coefficients):
    if name not in ("michalewicz", "g_function"):
        raise ValueError("Unknown synthetic function")
    if (dimension < 1 or training_points < 1 or test_points < 1 or not np.isfinite(noise_variance)
            or noise_variance < 0):
        raise ValueError("Invalid explicit synthetic design parameters")
    generator = np.random.default_rng(seed)
    unit = qmc.LatinHypercube(dimension, seed=generator).random(training_points + test_points)
    if name == "michalewicz":
        if not np.isfinite(michalewicz_m) or michalewicz_m <= 0 or int(michalewicz_m) != michalewicz_m:
            raise ValueError("Michalewicz m must be a positive integer")
        x = np.pi * unit
        y = -(np.sin(x) * np.sin(np.arange(1, dimension + 1) * x**2 / np.pi)**(2 * michalewicz_m)).sum(1)
    else:
        a = np.asarray(g_coefficients, dtype=np.float64)
        if a.shape != (dimension,) or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("G-function requires explicit nonnegative coefficients, one per dimension")
        x = unit
        y = ((np.abs(4 * x - 2) + a) / (1 + a)).prod(1)
    y = y + generator.normal(scale=np.sqrt(noise_variance), size=len(y))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise FloatingPointError("Synthetic generation produced nonfinite values")
    return RegressionDataset(name, x, y, tuple(f"x{i + 1}" for i in range(dimension)), name,
        metadata={"design": "One Latin hypercube, then seeded 80/20 split",
            "training_points": training_points, "test_points": test_points, "seed": seed,
            "noise_variance": noise_variance, "michalewicz_m": michalewicz_m,
            "g_coefficients": list(g_coefficients),
            "source": "SFU reference formulas; independent vectorized implementation",
            "paper_difference": "80/20; test counts 12500/25000 rather than always 25000"})
