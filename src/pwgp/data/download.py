"""Recoverable, checksum-aware download helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen


def file_checksum(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(
    url: str,
    destination: Path,
    *,
    checksum: str | None = None,
    checksum_algorithm: str = "sha256",
) -> tuple[Path, str]:
    """Download one declared source atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = file_checksum(destination, checksum_algorithm)
        if checksum is None or actual.lower() == checksum.lower():
            return destination, "cache"
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "pwgp-data/0.1"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if checksum:
        actual = file_checksum(temporary, checksum_algorithm)
        if actual.lower() != checksum.lower():
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch for {url}: expected {checksum}, got {actual}"
            )
    os.replace(temporary, destination)
    return destination, url


def safe_extract(archive: Path, destination: Path) -> list[Path]:
    """Extract zip/tar members while rejecting path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    def validate(name: str) -> None:
        candidate = (destination / name).resolve()
        if destination_root != candidate and destination_root not in candidate.parents:
            raise ValueError(f"Unsafe archive member: {name}")

    extracted: list[Path] = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                validate(member.filename)
            bundle.extractall(destination)
            extracted = [destination / member.filename for member in bundle.infolist()]
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                validate(member.name)
            bundle.extractall(destination, filter="data")
            extracted = [destination / member.name for member in bundle.getmembers()]
    else:
        raise ValueError(f"Unsupported archive: {archive}")
    return extracted

