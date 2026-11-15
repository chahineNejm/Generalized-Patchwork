import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

root = Path("data/flight_conversion_runtime")
records = []
for package, version, suffix in [("pandas", "0.23.4", "cp36-cp36m-win_amd64.whl"),
    ("numpy", "1.16.6", "cp36-cp36m-win_amd64.whl"),
    ("python-dateutil", "2.8.2", "py2.py3-none-any.whl"),
    ("pytz", "2021.3", "py2.py3-none-any.whl"), ("six", "1.16.0", "py2.py3-none-any.whl")]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json") as response:
        releases = json.load(response)["urls"]
    selected = [r for r in releases if r["filename"].endswith(suffix)]
    if len(selected) != 1:
        raise ValueError(f"Non-unique explicit wheel: {package}: {selected}")
    release = selected[0]
    path = root / "wheels" / release["filename"]
    path.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(release["url"], path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != release["digests"]["sha256"]:
        raise ValueError("PyPI wheel digest mismatch")
    with zipfile.ZipFile(path) as archive:
        archive.extractall(root / "python" / "site-packages")
    records.append({"package": package, "version": version, "url": release["url"], "sha256": release["digests"]["sha256"]})
    print(release["filename"])
(root / "wheel-provenance.json").write_text(json.dumps(records, indent=2))
