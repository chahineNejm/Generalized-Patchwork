"""One-time, explicitly requested conversion in Python 3.6 / pandas 0.23.4."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source, output):
    expected = "f111b35372eb2798cd43ab186203d5bb8733653260faa65c6307c87a933ef083"
    if platform.python_version_tuple()[:2] != ("3", "6") or pd.__version__ != "0.23.4":
        raise RuntimeError("Conversion requires the explicit isolated Python 3.6 / pandas 0.23.4 environment")
    if checksum(source) != expected:
        raise ValueError("Trusted source pickle checksum mismatch; refusing to unpickle")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Conversion never overwrites an existing result")
    columns = ["plane_age", "Distance", "AirTime", "DepTime", "ArrTime",
               "DayOfWeek", "DayofMonth", "Month", "ArrDelay"]
    frame = pd.read_pickle(source)
    if not isinstance(frame, pd.DataFrame) or any(c not in frame.columns for c in columns):
        raise ValueError("Unexpected trusted airline DataFrame schema")
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    valid = np.isfinite(numeric.values.astype(np.float64)).all(axis=1)
    clean = numeric.loc[valid].copy()
    if clean.empty:
        raise ValueError("No usable airline rows")
    for column in ("DepTime", "ArrTime"):
        clean[column] = 60 * np.floor(clean[column] / 100) + clean[column] % 100
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), X=clean[columns[:-1]].values.astype(np.float64),
                        y=clean["ArrDelay"].values.astype(np.float64))
    provenance = {"source_sha256": expected, "output_sha256": checksum(output),
        "source": str(source.resolve()), "dataset": "Hensman VFF/SVGP author-prepared airline",
        "row_counts": {"input_rows": len(frame), "removed_invalid": int((~valid).sum()), "output_rows": len(clean)},
        "feature_columns": columns[:-1], "target_column": "ArrDelay",
        "time_conversion": "HHMM to minutes since midnight, as in airline_svigp_rbf.py",
        "imputation": "none", "subsampling": "none", "split": "not applied",
        "normalization": "not applied", "source_unchanged": checksum(source) == expected,
        "reference_status": "VFF author-prepared data; not verified original 2013 split",
        "conversion": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
                       "script_sha256": checksum(Path(__file__)), "executable": sys.executable}}
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
