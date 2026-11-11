"""Registry-backed dataset cache and provenance manager."""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pwgp.data.download import download_file, file_checksum, safe_extract
from pwgp.data.loaders import load_flight, load_oisst, load_spatial, prepare_author_flight
from pwgp.data.types import RegressionDataset

PATCHWORK_BUNDLE_DRIVE_ID = "1gx6sJUKQgFHonbKlxrmrCEbDT_V5Xonu"
ICETHICK_MD5 = "3e4be5375cbc63ede60839a6dbc9f8ad"
FLIGHT_URL = "https://www.dropbox.com/s/32lz1vnjx3bg9hd/airline.pickle.zip?dl=1"
FLIGHT_REFERENCE = "https://github.com/jameshensman/VFF/tree/master/experiments/airline"


@dataclass(frozen=True)
class DatasetDescription:
    name: str
    description: str
    acquisition: str


DESCRIPTIONS = {
    "oisst": DatasetDescription(
        "oisst", "NOAA daily 0.25-degree sea-surface temperature v2.1", "automatic"
    ),
    "flight": DatasetDescription(
        "flight", "Author-prepared airline delay data used by Hensman's VFF/SVGP experiments", "author archive"
    ),
    "tco-l2": DatasetDescription(
        "tco-l2", "Total column ozone data from the Patchwork Kriging bundle", "automatic"
    ),
    "icethick": DatasetDescription(
        "icethick", "NSIDC-0099 West Antarctic ice thickness", "verified import"
    ),
}


class DatasetManager:
    def __init__(self, root: Path | str = "data") -> None:
        self.root = Path(root).expanduser().resolve()

    def list(self) -> tuple[DatasetDescription, ...]:
        return tuple(DESCRIPTIONS.values())

    def dataset_directory(self, name: str) -> Path:
        self._validate_name(name)
        return self.root / name

    def download(self, name: str, *, day: date | None = None) -> list[Path]:
        self._validate_name(name)
        if name == "oisst":
            if day is None:
                raise ValueError("OISST download requires a date")
            return self._download_oisst(day)
        if name == "flight":
            return self._download_flight()
        if name == "tco-l2":
            return self._download_patchwork_bundle()
        raise RuntimeError(
            "ICETHICK's official archive requires an interactive CAPTCHA. "
            "Download icethk.txt from https://www.usap-dc.org/view/dataset/609099 "
            "and run: pwgp-data import icethick <path>"
        )

    def import_file(self, name: str, source: Path | str) -> Path:
        self._validate_name(name)
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if name == "icethick":
            actual = file_checksum(source, "md5")
            if actual.lower() != ICETHICK_MD5:
                raise ValueError(
                    f"ICETHICK MD5 mismatch: expected {ICETHICK_MD5}, got {actual}"
                )
        filename = "airline.pickle" if name == "flight" and source.suffix in {".pkl", ".pickle"} else source.name
        destination = self.dataset_directory(name) / "raw" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        self._write_manifest(
            name,
            [destination],
            {"method": "import", "source": str(source)},
        )
        return destination

    def prepare(self, name: str) -> Path:
        """Convert downloaded raw files into the canonical model-ready form."""
        self._validate_name(name)
        if name != "flight":
            raise ValueError(f"{name} does not require a separate preparation step")
        raw = self.dataset_directory("flight") / "raw"
        output = self.dataset_directory("flight") / "prepared" / "flight-author.npz"
        prepared = prepare_author_flight(raw / "airline.pickle", output)
        self._write_manifest(
            "flight",
            [raw / "airline.pickle", prepared, prepared.with_suffix(".json")],
            {"method": "author-prepared", "reference": FLIGHT_REFERENCE,
             "cleaning": json.loads(prepared.with_suffix(".json").read_text(encoding="utf-8"))},
        )
        return prepared

    def files(self, name: str) -> list[Path]:
        directory = self.dataset_directory(name)
        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".part")
        )

    def manifest(self, name: str) -> dict[str, Any]:
        path = self.dataset_directory(name) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"No acquisition manifest for {name}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(
        self,
        name: str,
        *,
        path: Path | str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> RegressionDataset:
        self._validate_name(name)
        candidate = Path(path).expanduser().resolve() if path else self._default_data_file(name)
        if name == "oisst":
            return load_oisst(candidate, bounds=bounds)
        if name == "flight":
            return load_flight(candidate)
        if name == "icethick":
            return load_spatial(candidate, name=name, target_name="ice thickness")
        return load_spatial(candidate, name=name, target_name="total column ozone")

    def _download_oisst(self, day: date) -> list[Path]:
        stamp = day.strftime("%Y%m%d")
        filename = f"oisst-avhrr-v02r01.{stamp}.nc"
        url = (
            "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-"
            f"interpolation/v2.1/access/avhrr/{day:%Y%m}/{filename}"
        )
        destination = self.dataset_directory("oisst") / "raw" / f"{day:%Y}" / filename
        path, source = download_file(url, destination)
        self._write_manifest("oisst", [path], {"method": "download", "sources": [source]})
        return [path]

    def _download_flight(self) -> list[Path]:
        raw = self.dataset_directory("flight") / "raw"
        archive, source = download_file(FLIGHT_URL, raw / "airline.pickle.zip")
        if not zipfile.is_zipfile(archive):
            raise ValueError("Author airline source did not return a ZIP archive; no raw-data fallback")
        with zipfile.ZipFile(archive) as bundle:
            if bundle.namelist() != ["airline.pickle"]:
                raise ValueError("Author airline ZIP must contain exactly airline.pickle")
        safe_extract(archive, raw)
        paths = [archive, raw / "airline.pickle"]
        self._write_manifest("flight", paths, {"method": "author archive", "source": source,
                                               "reference": FLIGHT_REFERENCE})
        return paths

    def _download_patchwork_bundle(self) -> list[Path]:
        import gdown
        directory = self.dataset_directory("tco-l2")
        archive = directory / "raw" / "patchwork-authors-bundle"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            downloaded = gdown.download(
                id=PATCHWORK_BUNDLE_DRIVE_ID, output=str(archive), quiet=False
            )
            if not downloaded:
                raise RuntimeError("Could not download the Patchwork authors' bundle")
        extracted = safe_extract(archive, directory / "extracted")
        files = [path for path in extracted if path.is_file()]
        self._write_manifest(
            "tco-l2",
            [archive, *files],
            {
                "method": "google-drive",
                "drive_id": PATCHWORK_BUNDLE_DRIVE_ID,
            },
        )
        return files

    def _default_data_file(self, name: str) -> Path:
        if name == "flight":
            expected = self.dataset_directory(name) / "prepared" / "flight-author.npz"
            if not expected.is_file():
                raise FileNotFoundError("Run pwgp-data prepare flight before loading FLIGHT")
            return expected
        if name == "icethick":
            expected = self.dataset_directory(name) / "raw" / "icethk.txt"
            if not expected.is_file():
                raise FileNotFoundError("Import the official file as icethk.txt before loading")
            return expected
        if name == "tco-l2":
            expected = self.dataset_directory(name) / "prepared" / "tco-l2.npz"
            if not expected.is_file():
                raise FileNotFoundError(
                    "Pass an explicit TCO.L2 path or create data/tco-l2/prepared/tco-l2.npz"
                )
            return expected
        matches = [
            path for path in self.files(name) if path.suffix.lower() in {".nc", ".nc4"}
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one OISST file, found {len(matches)}; pass path= explicitly"
            )
        return matches[0]

    def _write_manifest(
        self, name: str, paths: list[Path], provenance: dict[str, Any]
    ) -> None:
        directory = self.dataset_directory(name)
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "dataset": asdict(DESCRIPTIONS[name]),
            "acquired_at": datetime.now(UTC).isoformat(),
            "provenance": provenance,
            "files": [
                {
                    "path": str(path.relative_to(directory)),
                    "bytes": path.stat().st_size,
                    "sha256": file_checksum(path),
                }
                for path in paths
                if path.is_file()
            ],
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _validate_name(name: str) -> None:
        if name not in DESCRIPTIONS:
            raise KeyError(f"Unknown dataset {name!r}; choose from {sorted(DESCRIPTIONS)}")

