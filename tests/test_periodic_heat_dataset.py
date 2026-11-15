import numpy as np
import pytest

from experiments.default_hyperparameters import multidataset_defaults
from experiments.multidataset_data import DatasetSpec, acquire_split, periodic_heat_dataset, spec_record


def small(**values):
    return {"cells": 8, "diffusivity": [0.6, 0.025], "angles": [0, 90], "periods": [2, 2],
            "baseline": 1.0, "amplitude": 0.8, "time": 0.015, "time_steps": 4,
            "relative_noise_std": 0.001, **values}


def test_periodic_heat_split_is_seeded_and_uses_kaggle_preprocessing(tmp_path):
    spec = DatasetSpec("periodic_heat", "periodic_heat", 2, .8, 0., 31, simulation=small())
    first, manifest = acquire_split(spec, data_root=tmp_path)
    second, _ = acquire_split(spec, data_root=tmp_path)
    assert first["x"].shape == (81, 2) and len(first["train"]) == 64 and len(first["test"]) == 17
    np.testing.assert_array_equal(first["y"].numpy(), second["y"].numpy())
    assert abs(float(first["y"][first["train"]].mean())) < 1e-12
    noiseless = periodic_heat_dataset(small(relative_noise_std=0.0), seed=31)
    assert manifest["metadata"]["noise_std"] == pytest.approx(0.001 * noiseless.y.std())
    assert "simulation" in spec_record(spec)
    assert "simulation" not in spec_record(DatasetSpec("tco", "tco", 2, .8, 0., 31))


def test_periodic_heat_requires_exact_settings(tmp_path):
    with pytest.raises(ValueError, match="simulation requires exactly"):
        acquire_split(DatasetSpec("periodic_heat", "periodic_heat", 2, .8, 0., 31,
                                  simulation={"cells": 8}), data_root=tmp_path)
    with pytest.raises(ValueError, match="only for, periodic_heat"):
        acquire_split(DatasetSpec("tco", "tco", 2, .8, 0., 31, simulation=small()), data_root=tmp_path)


def test_catalog_periodic_heat_matches_kaggle_2d_capacity():
    data, runs, _ = multidataset_defaults(device="cpu", input_root="manual")
    gpw = dict(runs["periodic_heat"])["GPW"]
    assert data["periodic_heat"].simulation["periods"] == [10, 10]
    assert (gpw.inducing_points, gpw.patches, gpw.noise_std, gpw.weight_epsilon) == (1024, 8, 0.1, 0.01)
    assert dict(runs["periodic_heat"])["GPW"] == dict(runs["tco"])["GPW"]
