import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm


root = Path("data/multidataset_sources_20261112")
root.mkdir(exist_ok=True)
sources = {
    "cal_housing.tgz": "https://ndownloader.figshare.com/files/5976036",
    "protein.zip": "https://archive.ics.uci.edu/static/public/265/physicochemical%2Bproperties%2Bof%2Bprotein%2Btertiary%2Bstructure.zip",
    "sarcos_inv.mat": "https://gaussianprocess.org/gpml/data/sarcos_inv.mat",
    "sarcos_inv_test.mat": "https://gaussianprocess.org/gpml/data/sarcos_inv_test.mat",
}
report = {}
for name, url in sources.items():
    digest = hashlib.sha256()
    with urlopen(Request(url, headers={"User-Agent": "pwgp-research/0.1"}), timeout=120) as response:
        headers = dict(response.headers)
        with (root / name).open("xb") as stream, tqdm(desc=name, unit="B", unit_scale=True) as bar:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                bar.update(len(chunk))
    report[name] = {"url": url, "sha256": digest.hexdigest(), "bytes": (root / name).stat().st_size,
                    "headers": headers}
    (root / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(name, report[name]["sha256"], flush=True)
