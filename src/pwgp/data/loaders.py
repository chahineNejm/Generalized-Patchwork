"""Load supported research datasets into a common regression representation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from pwgp.data.types import RegressionDataset


def _xy_from_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as data:
        if "X" not in data.files or "y" not in data.files:
            raise KeyError(f"Expected X/y arrays in {path}; found {data.files}")
        return np.asarray(data["X"], np.float64), np.asarray(data["y"], np.float64).reshape(-1)


def _xy_from_mat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    contents = loadmat(path)
    if "X" not in contents or "y" not in contents:
        raise KeyError(f"Expected exact X/y arrays in {path}; found {sorted(contents)}")
    return np.asarray(contents["X"], np.float64), np.asarray(contents["y"], np.float64).reshape(-1)


def load_spatial(path: Path, *, name: str, target_name: str) -> RegressionDataset:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        x, y = _xy_from_npz(path)
    elif suffix == ".mat":
        x, y = _xy_from_mat(path)
    elif suffix in {".txt", ".dat"}:
        raw = np.loadtxt(path)
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError(f"Spatial text data must have exactly 3 columns: {path}")
        x, y = raw[:, :2], raw[:, 2]
    elif suffix == ".csv":
        frame = pd.read_csv(path)
        required = ["longitude", "latitude", target_name]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing exact columns {missing} in {path}")
        x = frame[["longitude", "latitude"]].to_numpy(np.float64)
        y = frame[target_name].to_numpy(np.float64)
    else:
        raise ValueError(f"Unsupported spatial file type: {path.suffix}")
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"{name} requires two spatial predictors; got {x.shape}")
    return RegressionDataset(
        name=name,
        x=np.asarray(x, np.float64),
        y=np.asarray(y, np.float64).reshape(-1),
        feature_names=("longitude", "latitude"),
        target_name=target_name,
        source_files=(path,),
    ).finite()


def load_flight(path: Path) -> RegressionDataset:
    if path.suffix.lower() == ".npz":
        x, y = _xy_from_npz(path)
    elif path.suffix.lower() == ".mat":
        x, y = _xy_from_mat(path)
    else:
        frame = pd.read_csv(path)
        columns = [
            "age",
            "distance",
            "airtime",
            "deptime",
            "arrtime",
            "dayofweek",
            "dayofmonth",
            "month",
        ]
        missing = [column for column in [*columns, "arrdelay"] if column not in frame.columns]
        if missing:
            raise KeyError(f"Missing exact columns {missing} in {path}")
        x = frame[columns].to_numpy(np.float64)
        y = frame["arrdelay"].to_numpy(np.float64)
    if x.ndim != 2 or x.shape[1] != 8:
        raise ValueError(f"FLIGHT requires eight predictors; got {x.shape}")
    return RegressionDataset(
        name="flight",
        x=np.asarray(x, np.float64),
        y=np.asarray(y, np.float64).reshape(-1),
        feature_names=(
            "aircraft_age",
            "distance",
            "airtime",
            "departure_time",
            "arrival_time",
            "day_of_week",
            "day_of_month",
            "month",
        ),
        target_name="arrival_delay",
        source_files=(path,),
    ).finite()


def prepare_flight(
    flights_path: Path,
    planes_path: Path,
    output_path: Path,
    *,
    sample_size: int = 800_000,
    seed: int = 7,
) -> Path:
    """Prepare eight-predictor FLIGHT data by removing unusable rows, never imputing."""
    planes = pd.read_csv(planes_path, low_memory=False)
    required_plane_columns = ["tailnum", "year"]
    missing_plane_columns = [
        column for column in required_plane_columns if column not in planes.columns
    ]
    if missing_plane_columns:
        raise KeyError(f"Missing plane columns {missing_plane_columns} in {planes_path}")
    tail_column = "tailnum"
    year_column = "year"
    manufacture_year = pd.to_numeric(planes[year_column], errors="coerce")
    plane_year = pd.Series(
        manufacture_year.values, index=planes[tail_column].astype(str)
    ).to_dict()
    columns = [
        "Year",
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "DepTime",
        "ArrTime",
        "AirTime",
        "ArrDelay",
        "Distance",
        "TailNum",
        "Cancelled",
        "Diverted",
    ]
    generator = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    counts = dict.fromkeys(("input_rows", "removed_invalid_flags", "removed_cancelled_or_diverted",
                            "removed_invalid_features", "removed_invalid_age", "eligible_rows"), 0)
    for chunk in pd.read_csv(
        flights_path,
        compression="bz2",
        usecols=columns,
        chunksize=250_000,
        low_memory=False,
    ):
        counts["input_rows"] += len(chunk)
        flags = chunk[["Cancelled", "Diverted"]].apply(pd.to_numeric, errors="coerce")
        valid_flags = flags.isin([0, 1]).all(axis=1)
        noncancelled = flags.eq(0).all(axis=1)
        counts["removed_invalid_flags"] += int((~valid_flags).sum())
        counts["removed_cancelled_or_diverted"] += int((valid_flags & ~noncancelled).sum())
        chunk = chunk[valid_flags & noncancelled]
        age = pd.to_numeric(chunk["Year"], errors="coerce") - pd.to_numeric(
            chunk["TailNum"].astype(str).map(plane_year), errors="coerce"
        )
        prepared = pd.DataFrame(
            {
                "age": age,
                "distance": pd.to_numeric(chunk["Distance"], errors="coerce"),
                "airtime": pd.to_numeric(chunk["AirTime"], errors="coerce"),
                "deptime": pd.to_numeric(chunk["DepTime"], errors="coerce"),
                "arrtime": pd.to_numeric(chunk["ArrTime"], errors="coerce"),
                "dayofweek": pd.to_numeric(chunk["DayOfWeek"], errors="coerce"),
                "dayofmonth": pd.to_numeric(chunk["DayofMonth"], errors="coerce"),
                "month": pd.to_numeric(chunk["Month"], errors="coerce"),
                "arrdelay": pd.to_numeric(chunk["ArrDelay"], errors="coerce"),
            }
        )
        valid = np.isfinite(prepared.to_numpy(dtype=np.float64)).all(axis=1)
        counts["removed_invalid_features"] += int((~valid).sum())
        prepared = prepared.loc[valid]
        counts["removed_invalid_age"] += int(((prepared["age"] < 0) | (prepared["age"] > 100)).sum())
        prepared = prepared[(prepared["age"] >= 0) & (prepared["age"] <= 100)]
        counts["eligible_rows"] += len(prepared)
        if len(prepared):
            count = min(len(prepared), max(1, round(0.18 * len(prepared))))
            pieces.append(prepared.iloc[generator.choice(len(prepared), count, replace=False)])
    if not pieces:
        raise RuntimeError("No usable FLIGHT observations were produced")
    frame = pd.concat(pieces, ignore_index=True)
    if len(frame) > sample_size:
        frame = frame.iloc[
            generator.choice(len(frame), sample_size, replace=False)
        ].reset_index(drop=True)
    features = [
        "age",
        "distance",
        "airtime",
        "deptime",
        "arrtime",
        "dayofweek",
        "dayofmonth",
        "month",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=frame[features].to_numpy(np.float64),
        y=frame["arrdelay"].to_numpy(np.float64),
    )
    output_path.with_suffix(".json").write_text(json.dumps({
        "row_counts": {**counts, "sampled_rows": len(frame)},
        "seed": seed, "maximum_rows": sample_size, "within_chunk_fraction": 0.18,
        "imputation": "none", "malformed_numeric_policy": "remove_row",
        "reference_status": "raw reconstruction; original SVGP cleaning/split not verified",
    }, indent=2, allow_nan=False), encoding="utf-8")
    return output_path


def prepare_author_flight(source: Path, output: Path) -> Path:
    """Convert the trusted author pickle using the published SVGP feature mapping."""
    columns = ["plane_age", "Distance", "AirTime", "DepTime", "ArrTime",
               "DayOfWeek", "DayofMonth", "Month", "ArrDelay"]
    frame = pd.read_pickle(source)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Author airline pickle must contain a pandas DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing author airline columns: {missing}")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.to_numpy(dtype=np.float64)).all(axis=1)
    clean = numeric.loc[valid].copy()
    if clean.empty:
        raise ValueError("Author airline data has no usable rows")
    for column in ("DepTime", "ArrTime"):
        clean[column] = 60 * np.floor(clean[column] / 100) + clean[column] % 100
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, X=clean[columns[:-1]].to_numpy(np.float64),
                        y=clean["ArrDelay"].to_numpy(np.float64))
    output.with_suffix(".json").write_text(json.dumps({
        "source": str(source), "dataset": "Hensman VFF/SVGP author-prepared airline",
        "row_counts": {"input_rows": len(frame), "removed_invalid": int((~valid).sum()),
                       "output_rows": len(clean)},
        "time_conversion": "HHMM to minutes since midnight, as in airline_svigp_rbf.py",
        "feature_columns": columns[:-1], "target_column": "ArrDelay",
        "imputation": "none", "subsampling": "none", "split": "not applied",
        "normalization": "not applied; fit preprocessing on training rows only",
        "reference_status": "VFF author-prepared data, not verified original 2013 split",
    }, indent=2, allow_nan=False), encoding="utf-8")
    return output


def load_oisst(path: Path, *, bounds: tuple[float, float, float, float] | None = None) -> RegressionDataset:
    import xarray as xr
    with xr.open_dataset(path) as dataset:
        variable = "sst"
        if variable not in dataset.data_vars:
            raise KeyError(f"Required OISST variable 'sst' is absent from {path}")
        field = dataset[variable].squeeze(drop=True)
        if bounds is not None:
            west, east, south, north = bounds
            west_360, east_360 = west % 360, east % 360
            field = field.sel(lat=slice(south, north), lon=slice(west_360, east_360))
        longitude, latitude = np.meshgrid(field["lon"].values, field["lat"].values)
        x = np.column_stack((longitude.ravel(), latitude.ravel()))
        y = np.asarray(field.values, np.float64).ravel()
    return RegressionDataset(
        name="oisst",
        x=x,
        y=y,
        feature_names=("longitude", "latitude"),
        target_name=variable,
        source_files=(path,),
        metadata={"bounds": bounds},
    ).finite()

