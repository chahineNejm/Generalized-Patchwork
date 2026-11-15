import csv
import hashlib
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm

from pwgp.data.ukhousing import POSTCODE_URL, PRICE_URL


ROOT = Path("data/ukhousing_verified_20261112")
ROOT.mkdir(exist_ok=True)


def download(url, name):
    target = ROOT / name
    digest = hashlib.sha256()
    with urlopen(Request(url, headers={"User-Agent": "pwgp-research/0.1"}), timeout=120) as response:
        headers = dict(response.headers)
        resolved = response.url
        with target.open("xb") as stream, tqdm(desc=name, unit="B", unit_scale=True) as bar:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))
    return {"url": url, "resolved_url": resolved, "headers": headers,
            "bytes": target.stat().st_size, "sha256": digest.hexdigest()}


report = {"prices": download(PRICE_URL, "pp-2018.csv"),
          "postcodes": download(POSTCODE_URL, "ukpostcodes.zip"),
          "prices_repeat": download(PRICE_URL, "pp-2018-repeat.csv")}
assert report["prices"]["sha256"] == report["prices_repeat"]["sha256"]
count = 0
with (ROOT / "pp-2018.csv").open(encoding="utf-8", newline="") as stream:
    for row in csv.reader(stream, strict=True):
        assert len(row) == 16, (count, len(row))
        assert row[2].startswith("2018-"), (count, row[2])
        assert int(row[1]) > 0, (count, row[1])
        count += 1
report["prices"]["validated_rows"] = count
with zipfile.ZipFile(ROOT / "ukpostcodes.zip") as archive:
    assert archive.testzip() is None
    assert archive.namelist() == ["ukpostcodes.csv"]
    with archive.open("ukpostcodes.csv") as stream:
        postcodes = pd.read_csv(stream)
    assert {"postcode", "latitude", "longitude"} <= set(postcodes)
    assert not postcodes["postcode"].duplicated().any()
    report["postcodes"]["validated_rows"] = len(postcodes)
    report["postcodes"]["columns"] = list(postcodes)
for label, name in (("prices", "pp-2018.csv"), ("postcodes", "ukpostcodes.zip")):
    old = Path("data/ukhousing") / name
    report[label]["previous_sha256"] = hashlib.file_digest(old.open("rb"), "sha256").hexdigest()
    report[label]["identical_to_previous"] = report[label]["previous_sha256"] == report[label]["sha256"]
(ROOT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
