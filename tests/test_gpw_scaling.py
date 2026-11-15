import json
from pathlib import Path

import pandas as pd
import pytest

from experiments import multidataset_worker as worker
from experiments.build_gpw_scaling_kaggle import build_notebook
from experiments.default_hyperparameters import gpw_scaling_defaults
from experiments.multidataset_data import DatasetSpec

ROOT = Path(__file__).resolve().parents[1]


def request(**values):
    defaults = gpw_scaling_defaults(device="cpu", input_root="manual")
    return {"study": "gpw_scaling", "device": "cpu", "input_root": "manual", "excluded_model_types": ["vnngp"],
            "model_overrides": {}, "save_inducing_positions": True, **defaults, **values}


def test_default_plan_has_matched_gpw_and_patchwork_per_patch_count():
    result = worker.make_plan(request())
    runs = result["plan"]["runs"]
    assert result["total_runs"] == 16 and result["plan"]["study"]["type"] == "gpw_scaling"
    assert [s["name"] for s in result["plan"]["data"]] == ["oisst", "flight"]
    assert result["plan"]["data"][1]["subset_size"] == 800000
    expected = {2: 64, 4: 64, 8: 32, 16: 32}
    for row in runs:
        config, count = row["config"], row["patch_count"]
        assert row["stitches_per_boundary"] == expected[count] and config["seed"] == 31
        if config["model"] == "gpw":
            assert (config["inducing_points"], config["patches"], config["stitches_per_edge"]) == (2048, count, expected[count])
            assert config["patch_neighbors"] == min(2, count - 1)
            assert (config["q_rank"], config["stitch_mode"]) == (32, "learned")
        else:
            assert config["model"] == "patchwork" and config["exact_partition"] == "pc_tree"
            assert (config["exact_tree_patches"], config["exact_stitches_per_edge"]) == (count, expected[count])
            assert config["exact_training_points"] == "gpw_initial"
            assert config["exact_gpw_reference"] == {"inducing_points": 2048, "patches": count,
                                                     "seed": 31, "threads": config["threads"]}
    assert {(r["dataset"], r["patch_count"], r["config"]["model"]) for r in runs} == {
        (d, p, m) for d in ("oisst", "flight") for p in (2, 4, 8, 16) for m in ("gpw", "patchwork")}


def test_seeds_vary_model_and_patchwork_reference_only():
    result = worker.make_plan(request(seeds=[31, 32]))
    assert result["total_runs"] == 32
    for row in result["plan"]["runs"]:
        if row["config"]["model"] == "patchwork":
            assert row["config"]["exact_gpw_reference"]["seed"] == row["config"]["seed"]
    assert all(spec["seed"] == 31 for spec in result["plan"]["data"])


@pytest.mark.parametrize("updates", [
    {"patch_counts": [3]}, {"patch_counts": [2, 2]}, {"inducing_points": 2050},
    {"stitches_per_boundary": {"2": 64}}, {"seeds": []},
    {"model_overrides": {"oisst": {"GPW": {"patches": 4}}}},
    {"model_overrides": {"oisst": {"Standard Patchwork (PC tree)": {"exact_stitches_per_edge": 3}}}},
    {"model_overrides": {"oisst": {"SVGP": {"epochs": 3}}}},
    {"model_overrides": {"tco": {"GPW": {"epochs": 3}}}}])
def test_invalid_scaling_requests_raise(updates):
    with pytest.raises(ValueError):
        worker.make_plan(request(**updates))


def test_notebook_is_reproducible_and_configuration_only(tmp_path):
    (tmp_path / "notebooks").mkdir()
    notebook = json.loads(build_notebook(tmp_path).read_text())
    assert notebook == json.loads((ROOT / "notebooks/GPW_Scaling_Kaggle.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), "scaling", "exec")
            assert entry["outputs"] == []
    assert "SEEDS = [31]" in code and "PATCH_COUNTS = [2, 4, 8, 16]" in code
    assert "INDUCING_POINTS = 2048" in code and 'study="gpw_scaling"' in code
    assert "RUN_TRAINING = False" in code and "gpw_scaling_" in code and "ablation_summary" not in code


def test_short_scaling_worker_runs_both_models_per_patch_count(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "source_snapshot", lambda value: (tmp_path, b"source"))
    events = []
    monkeypatch.setattr(worker, "emit", lambda kind, **values: events.append((kind, values)))
    from experiments import default_hyperparameters

    original = default_hyperparameters.multidataset_defaults

    def small_catalog(**kwargs):
        data, runs, deferred = original(**kwargs)
        data["periodic_heat"] = DatasetSpec("periodic_heat", "periodic_heat", 2, .8, 0., 31, simulation={
            "cells": 10, "diffusivity": [0.6, 0.025], "angles": [0, 90], "periods": [2, 2], "baseline": 1.0,
            "amplitude": 0.8, "time": 0.015, "time_steps": 4, "relative_noise_std": 0.001})
        return data, runs, deferred

    monkeypatch.setattr(default_hyperparameters, "multidataset_defaults", small_catalog)
    small = {"batch_size": 32, "prediction_batch_size": 32, "snapshot_every": 2, "q_rank": 2}
    plan = worker.make_plan(request(datasets=["periodic_heat"], patch_counts=[2, 4], inducing_points=16,
        stitches_per_boundary={"2": 3, "4": 2}, data_overrides={}, epochs=2, warmup_epochs=1, early_stopping=None,
        model_overrides={"periodic_heat": {"GPW": small, "Standard Patchwork (PC tree)": {"snapshot_every": 2}}}))["plan"]
    common = {"plan": plan, "plan_sha256": worker.plan_hash(plan), "source_provenance": {}}
    prepared = tmp_path / "prepared"
    worker.execute({**common, "action": "prepare", "directory": str(prepared), "data_root": str(tmp_path)})
    batch = tmp_path / "batch"
    worker.execute({**common, "action": "train", "directory": str(batch),
                    "prepared_directory": str(prepared), "execute_training": True})
    table = pd.read_csv(batch / "comparison.csv")
    assert len(table) == 4 and set(table["patch_count"]) == {2, 4}
    assert set(table["base_model"]) == {"GPW", "Standard Patchwork (PC tree)"}
    assert table.set_index("model")["stitches_per_boundary"].to_dict() == {
        "GPW P2 seed31": 3, "Standard Patchwork (PC tree) P2 seed31": 3,
        "GPW P4 seed31": 2, "Standard Patchwork (PC tree) P4 seed31": 2}
    assert sum(kind == "complete" for kind, _ in events) == 4
