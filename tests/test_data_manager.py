from pathlib import Path

import numpy as np
import pytest

from pwgp.data.manager import DatasetManager


def test_import_writes_provenance_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.npz"
    np.savez(source, X=np.ones((3, 8)), y=np.arange(3))
    manager = DatasetManager(tmp_path / "cache")

    imported = manager.import_file("flight", source)
    manifest = manager.manifest("flight")

    assert imported.exists()
    assert manifest["dataset"]["name"] == "flight"
    assert manifest["provenance"]["method"] == "import"
    assert len(manifest["files"][0]["sha256"]) == 64
    loaded = manager.load("flight", path=imported)
    assert loaded.x.shape == (3, 8)


def test_icethick_rejects_unverified_file(tmp_path: Path) -> None:
    source = tmp_path / "icethk.txt"
    source.write_text("not the official dataset", encoding="utf-8")
    manager = DatasetManager(tmp_path / "cache")

    with pytest.raises(ValueError, match="MD5 mismatch"):
        manager.import_file("icethick", source)


def test_unknown_dataset_is_rejected(tmp_path: Path) -> None:
    manager = DatasetManager(tmp_path)
    with pytest.raises(KeyError):
        manager.files("unknown")

