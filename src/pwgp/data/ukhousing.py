"""Reconstruct the documented 2018 UK flats task from declared public sources."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pwgp.data.download import file_checksum

PRICE_URL = "https://price-paid-data.publicdata.landregistry.gov.uk/pp-2018.csv"
POSTCODE_URL = "https://data.freemaptools.com/download/full-uk-postcodes/ukpostcodes.zip"
SOURCE_SNAPSHOT = "2026-11-12"
PRICE_SHA256 = "dbf8e95e719630fa8b8d0c90bea52e9e282e7bb6c938951737d4ed65c7655eda"
POSTCODE_SHA256 = "d75818f49dc7de39c61fcc0577d27eb548caaa4a28509adc9f602da5cbc48906"


def acquire(url: str, path: Path, checksum: str | None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urlopen(Request(url, headers={"User-Agent": "pwgp-research/0.1"}), timeout=120) as response:
            total = int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None
            with temporary.open("wb") as stream, tqdm(total=total, unit="B", unit_scale=True, desc=path.name) as bar:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
                    bar.update(len(chunk))
        actual = file_checksum(temporary)
        if checksum is not None and actual != checksum:
            raise ValueError(f"Source checksum mismatch: {url}; expected SHA-256={checksum}; "
                             f"actual SHA-256={actual}; downloaded file={temporary}")
        os.replace(temporary, path)
    actual = file_checksum(path)
    if checksum is not None and actual != checksum:
        raise ValueError(f"Cached source checksum mismatch: {path}; expected SHA-256={checksum}; "
                         f"actual SHA-256={actual}")
    return actual


def prepare_ukhousing(cache: str | Path = "data/ukhousing") -> tuple[np.ndarray, np.ndarray, dict]:
    cache = Path(cache)
    prices_path, postcode_path = cache / "pp-2018.csv", cache / "ukpostcodes.zip"
    hashes = {"prices": acquire(PRICE_URL, prices_path, PRICE_SHA256),
              "postcodes": acquire(POSTCODE_URL, postcode_path, POSTCODE_SHA256)}
    with zipfile.ZipFile(postcode_path) as bundle:
        if bundle.namelist() != ["ukpostcodes.csv"]:
            raise ValueError(f"Unexpected postcode archive schema: {bundle.namelist()}")
        with bundle.open("ukpostcodes.csv") as stream:
            postcodes = pd.read_csv(stream, usecols=["postcode", "latitude", "longitude"])
    if postcodes["postcode"].duplicated().any():
        raise ValueError("Postcode coordinates must have unique postcode keys")
    columns = ["id", "price", "date", "postcode", "type", "new", "duration", "paon",
               "saon", "street", "locality", "city", "district", "county", "category", "status"]
    groups = []
    raw_count = flat_count = unmatched = 0
    for chunk in tqdm(pd.read_csv(prices_path, names=columns, header=None,
                                 usecols=[0, 1, 2, 3, 4], chunksize=200_000), desc="Prepare 2018 flats"):
        raw_count += len(chunk)
        flats = chunk[(chunk["type"] == "F") & (chunk["price"] >= 1000)].copy()
        flat_count += len(flats)
        joined = flats.merge(postcodes, how="left", on="postcode", validate="many_to_one")
        finite = np.isfinite(joined[["latitude", "longitude"]]).all(axis=1)
        unmatched += int((~finite).sum())
        keep = finite & joined["longitude"].between(-5.7, 1.8, inclusive="neither")
        keep &= joined["latitude"].between(50.0, 55.5, inclusive="neither")
        groups.append(joined.loc[keep, ["id", "longitude", "latitude", "price"]])
    frame = pd.concat(groups, ignore_index=True)
    if len(frame) < 10_000 or frame["id"].duplicated().any():
        raise ValueError("Unexpected UK flats reconstruction size or duplicate transaction IDs")
    x = frame[["longitude", "latitude"]].to_numpy(dtype=np.float64)
    y = np.log(frame["price"].to_numpy(dtype=np.float64))
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite reconstructed UKHousing data")
    manifest = {
        "dataset": "UKHousing 2018 flats reconstruction; not verified VNNGP snapshot",
        "source_snapshot": SOURCE_SNAPSHOT,
        "source_urls": [PRICE_URL, POSTCODE_URL], "source_sha256": hashes,
        "raw_transactions": raw_count, "flats_price_at_least_1000": flat_count,
        "unmatched_or_nonfinite_coordinates": unmatched, "retained": len(frame),
        "roi": {"longitude": [-5.7, 1.8], "latitude": [50.0, 55.5]},
        "target": "natural logarithm of GBP sale price",
        "duplicates": "Repeated coordinates retained as distinct transactions",
        "attribution": [
            "Contains HM Land Registry data (C) Crown copyright and database right 2021. OGL v3.0.",
            "Contains Ordnance Survey data (C) Crown copyright and database right 2021.",
            "Contains Royal Mail data (C) Royal Mail copyright and database right 2021.",
            "Source: Office for National Statistics licensed under OGL v3.0; via FreeMapTools.",
        ],
        "preparation_reference": "https://github.com/cunningham-lab/hipgp/blob/master/experiments-hip-gp/uk_housing_data.py",
    }
    np.savez_compressed(cache / "prepared.npz", x=x, y=y, transaction_id=frame["id"].to_numpy(dtype=str))
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return x, y, manifest


if __name__ == "__main__":
    print(json.dumps(prepare_ukhousing()[2], indent=2))
