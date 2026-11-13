import json
from dataclasses import replace

import numpy as np
import pytest

from experiments import exploratory_analysis
from experiments import multidataset_data as data
from pwgp.data.types import RegressionDataset


def source():
    rng = np.random.default_rng(12)
    return RegressionDataset("flight", rng.normal(size=(1000, 2)), rng.normal(size=1000),
                             ("a", "b"), "delay")


def test_subset_is_distinct_seeded_and_keeps_exact_source_values():
    original = source()
    subset, rows = data.flight_subset(original, size=800, seed=31)
    np.testing.assert_array_equal(rows, np.random.default_rng(31).choice(1000, 800, replace=False))
    assert len(np.unique(rows)) == 800
    np.testing.assert_array_equal(subset.x, original.x[rows])
    np.testing.assert_array_equal(subset.y, original.y[rows])
    assert not np.array_equal(rows, data.flight_subset(original, size=800, seed=32)[1])
    assert subset.metadata["subset"]["source_rows"] == 1000


@pytest.mark.parametrize("size", [True, 1, 0, 1001, 800.0])
def test_invalid_subset_sizes_raise(size):
    with pytest.raises(ValueError, match="subset_size"):
        data.flight_subset(source(), size=size, seed=31)


def test_subset_preparation_scaling_reuse_and_row_checksum(tmp_path, monkeypatch):
    original = source()
    monkeypatch.setattr(data, "verify_manual_inputs", lambda specs: None)
    monkeypatch.setattr(data, "load_real", lambda *args, **kwargs: original)
    monkeypatch.setattr(exploratory_analysis, "explore_multidataset",
                        lambda split, *, directory: directory.mkdir())
    spec = data.DatasetSpec("flight", "flight", 2, .8, 0., 31, subset_size=800)
    path = data.prepare_dataset(spec, directory=tmp_path / "prepared", data_root=tmp_path)
    split = data.load_prepared(spec, path)
    assert (len(split["train"]), len(split["test"])) == (640, 160)
    rows = split["source_rows"]
    np.testing.assert_array_equal(split["raw_x"], original.x[rows])
    np.testing.assert_array_equal(split["raw_y"], original.y[rows])
    np.testing.assert_allclose(split["x"], (original.x[rows] - original.x[rows].min(0)) /
                               np.ptp(original.x[rows], axis=0))
    assert split["target_mean"] == original.y[rows][split["train"]].mean()
    manifest = json.loads((path / "data_manifest.json").read_text())
    prepared = json.loads((path / "prepared.json").read_text())
    assert manifest["metadata"]["subset"]["size"] == 800
    assert prepared["data"]["subset_size"] == 800
    assert "source_rows.npy" in prepared["checksums"]
    with pytest.raises(ValueError, match="identity mismatch"):
        data.load_prepared(replace(spec, subset_size=None), path)
    np.save(path / "source_rows.npy", rows[::-1])
    with pytest.raises(ValueError, match="checksum mismatch: source_rows"):
        data.load_prepared(spec, path)


def test_full_data_identity_unchanged_and_other_sources_reject_subsets(tmp_path):
    spec = data.DatasetSpec("flight", "flight", 2, .8, 0., 31)
    assert "subset_size" not in data.spec_record(spec)
    with pytest.raises(ValueError, match="only for FLIGHT"):
        data.acquire_split(replace(spec, kind="protein", subset_size=800), data_root=tmp_path)
