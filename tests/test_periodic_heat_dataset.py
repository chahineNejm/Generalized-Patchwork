from dataclasses import replace

import numpy as np
import pytest

from experiments.default_hyperparameters import multidataset_defaults
from experiments.multidataset_data import (
    DatasetSpec,
    acquire_split,
    periodic_heat_dataset,
    spec_record,
)


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


def test_interface_gap_split_regions_scaling_and_persistence(tmp_path):
    from experiments.multidataset_data import REGION_SPLITS, load_prepared, prepare_dataset

    settings = small(cells=16, gap_width=0.13, gap_side="right", near_interface_width=0.2)
    spec = DatasetSpec("periodic_heat_gap", "periodic_heat", 2, .8, 0., 31, simulation=settings)
    split, manifest = acquire_split(spec, data_root=tmp_path)
    x = split["raw_x"][:, 0]
    assert not ((x[split["train"]] >= 0.5) & (x[split["train"]] < 0.63)).any()
    regions = [split[key] for key in REGION_SPLITS]
    assert all(len(rows) for rows in regions)
    assert sorted(np.concatenate(regions)) == sorted(split["test"])
    assert len(np.intersect1d(split["train"], split["test"])) == 0
    assert len(split["train"]) + len(split["test"]) == len(split["y"])
    assert abs(float(split["y"][split["train"]].mean())) < 1e-12
    assert split["score_splits"] == ["test", *REGION_SPLITS]
    assert split["preprocessing"]["interface_gap"]["moved_training_rows"] > 0
    prepared = prepare_dataset(spec, directory=tmp_path / "prepared", data_root=tmp_path)
    loaded = load_prepared(spec, prepared)
    for key in ("train", "test", *REGION_SPLITS):
        np.testing.assert_array_equal(loaded[key], split[key])
    assert loaded["score_splits"] == ["test", *REGION_SPLITS]


def test_partial_gap_settings_raise(tmp_path):
    with pytest.raises(ValueError, match="all of"):
        acquire_split(DatasetSpec("periodic_heat", "periodic_heat", 2, .8, 0., 31,
                                  simulation=small(gap_width=0.1)), data_root=tmp_path)


def test_catalog_gap_variant_shares_field_and_models():
    data, runs, _ = multidataset_defaults(device="cpu", input_root="manual")
    gap = data["periodic_heat_gap"].simulation
    assert {k: v for k, v in gap.items() if k not in ("gap_width", "gap_side", "near_interface_width")} == \
        data["periodic_heat"].simulation
    assert (gap["gap_width"], gap["gap_side"], gap["near_interface_width"]) == (0.05, "right", 0.1)
    assert runs["periodic_heat_gap"] == runs["periodic_heat"]


def test_ablation_worker_reports_region_metrics(tmp_path, monkeypatch):
    import pandas as pd

    from experiments import default_hyperparameters
    from experiments import multidataset_worker as worker

    original = default_hyperparameters.multidataset_defaults

    def small_catalog(**kwargs):
        data, runs, deferred = original(**kwargs)
        data["periodic_heat_gap"] = DatasetSpec("periodic_heat_gap", "periodic_heat", 2, .8, 0., 31,
            simulation=small(cells=16, gap_width=0.13, gap_side="right", near_interface_width=0.2))
        return data, runs, deferred

    monkeypatch.setattr(default_hyperparameters, "multidataset_defaults", small_catalog)
    monkeypatch.setattr(worker, "source_snapshot", lambda value: (tmp_path, b"source"))
    monkeypatch.setattr(worker, "emit", lambda kind, **values: None)
    plan = worker.make_plan({"study": "gpw_stitch_rank", "device": "cpu", "input_root": "manual",
        "datasets": ["periodic_heat_gap"], "seeds": [31], "ranks": [0], "stitch_modes": ["none", "fixed"],
        "inducing_points": 16, "patches": 2, "epochs": 2, "warmup_epochs": 1, "early_stopping": None,
        "save_inducing_positions": False, "progress_every_epochs": 1, "data_overrides": {},
        "model_overrides": {"periodic_heat_gap": {"patch_neighbors": 1, "stitches_per_edge": 2,
            "gpw_partition": "grid", "gpw_grid_shape": [2, 1], "batch_size": 32,
            "prediction_batch_size": 32, "snapshot_every": 2}}})["plan"]
    common = {"plan": plan, "plan_sha256": worker.plan_hash(plan), "source_provenance": {}}
    worker.execute({**common, "action": "prepare", "directory": str(tmp_path / "prepared"), "data_root": str(tmp_path)})
    worker.execute({**common, "action": "train", "directory": str(tmp_path / "batch"),
                    "prepared_directory": str(tmp_path / "prepared"), "execute_training": True})
    table = pd.read_csv(tmp_path / "batch/comparison.csv")
    assert len(table) == 2
    for region in ("gap", "near_interface", "interior"):
        for metric in ("rmse", "nlpd", "coverage_95"):
            assert table[f"{region}_{metric}_standardized"].notna().all()


def test_zigzag_interface_changes_materials_and_straight_default_is_unchanged():
    from experiments.periodic_heat_preview import PreviewConfig, interface_position, simulate

    base = PreviewConfig(cells=16, diffusivity=(0.6, 0.025), angle_pairs=((0, 90),), periods=(5, 4),
                         time=0.015, time_steps=4)
    straight = simulate(base, (0, 90))[2]
    explicit = simulate(replace(base, interface_amplitude=0.0, interface_teeth=0), (0, 90))[2]
    np.testing.assert_array_equal(straight, explicit)
    zigzag = replace(base, interface_amplitude=0.06, interface_teeth=4)
    y = np.linspace(0, 0.25, 5)  # one zigzag tooth spans 1/teeth = 0.25 in y
    np.testing.assert_allclose(interface_position(y, zigzag), [0.56, 0.5, 0.44, 0.5, 0.56])
    assert not np.allclose(simulate(zigzag, (0, 90))[2], straight)
    for bad in ({"interface_amplitude": 0.06, "interface_teeth": 0}, {"interface_amplitude": 0.3, "interface_teeth": 4},
                {"interface_amplitude": 0.0, "interface_teeth": 2}):
        with pytest.raises(ValueError, match="interface"):
            replace(base, **bad).validate()


def test_catalog_jagged_variant():
    data, runs, _ = multidataset_defaults(device="cpu", input_root="manual")
    jagged = data["periodic_heat_jagged"].simulation
    assert jagged["periods"] == [5, 4] and (jagged["interface_amplitude"], jagged["interface_teeth"]) == (0.06, 4)
    assert (jagged["gap_width"], jagged["gap_side"]) == (0.05, "right")
    assert runs["periodic_heat_jagged"] == runs["periodic_heat"]
    with pytest.raises(ValueError, match="together"):
        periodic_heat_dataset({**data["periodic_heat"].simulation, "interface_amplitude": 0.06}, seed=31)
