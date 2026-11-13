import ast
import json
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import multidataset_worker as worker
from experiments.build_multidataset_kaggle import build_notebook
from experiments.default_hyperparameters import benchmark_repeat_defaults
from experiments.multidataset_data import DatasetSpec

ROOT = Path(__file__).resolve().parents[1]


def request(**values):
    return {**benchmark_repeat_defaults(device="cpu", input_root="manual"),
        "study": "benchmark_repeats", "device": "cpu", "input_root": "manual",
        "excluded_model_types": ["vnngp"], "data_overrides": {},
        "save_inducing_positions": True, **values}


def test_repeat_plan_changes_only_model_seed_and_matched_reference_seed():
    arguments = request()
    result = worker.make_plan(arguments)
    assert result["total_runs"] == 44 and result["total_epochs"] == 4400
    plan = result["plan"]
    assert [d["name"] for d in plan["data"]] == ["california", "protein", "tco", "oisst"]
    assert {d["seed"] for d in plan["data"]} == {31}
    assert plan["progress_every_epochs"] == 20
    assert {r["config"]["seed"] for r in plan["runs"]} == {32, 33}
    base = worker.make_plan({k:v for k,v in arguments.items() if k != "study"})["plan"]
    assert plan["data"] == base["data"]
    for row in plan["runs"]:
        original = next(r for r in base["runs"] if (r["dataset"], r["name"]) == (row["dataset"], row["base_name"]))
        expected = json.loads(json.dumps(original["config"]))
        expected["seed"] = row["config"]["seed"]
        if expected["exact_gpw_reference"] is not None:
            expected["exact_gpw_reference"]["seed"] = row["config"]["seed"]
        assert row["config"] == expected
        assert row["config"]["model"] != "vnngp"
        if row["config"]["model"] == "prospar":
            assert row["config"]["prospar"] == {"temperatures": 1., "exponent": 1.,
                "inducing_nuggets": .01, "prior_jitter": 0.}
    assert sum(r["dataset"] == "oisst" for r in plan["runs"]) == 8


@pytest.mark.parametrize("change", [
    {"seeds": []}, {"seeds": [32, 32]}, {"seeds": [True]}, {"seeds": [-1]},
    {"progress_every_epochs": 0}, {"progress_every_epochs": True},
    {"data_overrides": {"california": {"seed": 32}},
     "model_overrides": {"california": {"SVGP": {"seed": 32}, "GPW": {"seed": 32},
         "GPW (alternative)": {"seed": 32}, "ProSpar": {"seed": 32},
         "ProSpar (alternative)": {"seed": 32}, "Standard Patchwork (PC tree)": {"seed": 32}}}},
])
def test_invalid_repeat_controls_raise(change):
    with pytest.raises((ValueError, TypeError)):
        worker.make_plan(request(**change))


def test_repeat_notebook_reproducible_configuration_only(tmp_path):
    (tmp_path / "notebooks").mkdir()
    notebook = json.loads(build_notebook(tmp_path, repeats=True).read_text())
    assert notebook == json.loads((ROOT / "notebooks/Benchmark_Repeats_Kaggle.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), "repeats", "exec")
            assert entry["outputs"] == []
    assert "SEEDS = [32, 33]" in code and "PROGRESS_EVERY_EPOCHS = 20" in code
    assert 'study="benchmark_repeats"' in code and "RUN_TRAINING = False" in code
    assert "EPOCHS = 100" in code and "WARMUP_EPOCHS = 20" in code
    imports = [n.module for n in ast.walk(ast.parse(code)) if isinstance(n, ast.ImportFrom)]
    assert not any(n and n.startswith(("experiments", "pwgp")) for n in imports)


def test_repeats_keep_one_prepared_split_and_archive_each_model(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "source_snapshot", lambda value: (tmp_path, b"source"))
    events = []
    monkeypatch.setattr(worker, "emit", lambda kind, **values: events.append((kind, values)))
    names = ["SVGP", "GPW", "ProSpar", "Standard Patchwork (PC tree)"]
    changes = {name: {"inducing_points": 24, "patches": 4, "q_rank": 4, "stitches_per_edge": 2,
        "exact_tree_patches": 4, "exact_stitches_per_edge": 2, "batch_size": 64,
        "prediction_batch_size": 32, "snapshot_every": 2} for name in names}
    plan = worker.make_plan(request(datasets=["california"], models={"california": names},
        model_overrides={"california": changes}, epochs=2, warmup_epochs=1, early_stopping=None))["plan"]
    spec = DatasetSpec("tiny", "g_function", 3, .8, 0., 31, synthetic={"training_points": 160,
        "test_points": 40, "noise_variance": 1e-8, "michalewicz_m": 10, "g_coefficients": [0., .5, 1.]})
    plan["data"] = [asdict(spec)]
    for row in plan["runs"]:
        row["dataset"] = "tiny"
    common = {"plan": plan, "plan_sha256": worker.plan_hash(plan), "source_provenance": {}}
    prepared, batch = tmp_path / "prepared", tmp_path / "batch"
    worker.execute({**common, "action": "prepare", "directory": str(prepared), "data_root": str(tmp_path)})
    original_split = (prepared / "tiny/split.npz").read_bytes()
    worker.execute({**common, "action": "train", "directory": str(batch),
        "prepared_directory": str(prepared), "execute_training": True})
    assert original_split == (prepared / "tiny/split.npz").read_bytes() == (batch / "tiny/split.npz").read_bytes()
    for index, row in enumerate(plan["runs"]):
        folder = batch / "tiny" / f"run_{index:02d}"
        config = json.loads((folder / "config.json").read_text())
        control = json.loads((folder / "training_control.json").read_text())
        assert config["seed"] == row["config"]["seed"]
        assert control["progress_every_epochs"] == 20
        assert control["progress_description"] == f"train:tiny / {row['name']}"
        with zipfile.ZipFile(batch / f"tiny_run_{index:02d}.zip") as archive:
            assert "split.npz" in archive.namelist() and f"run_{index:02d}/metrics.json" in archive.namelist()
    for start in (0, 4):
        with np.load(batch / "tiny" / f"run_{start+1:02d}" / "initialization.npz") as gpw, np.load(
                batch / "tiny" / f"run_{start+3:02d}" / "partition.npz") as standard:
            np.testing.assert_array_equal(np.sort(gpw["indices"].flatten()), np.sort(standard["indices"]))
    rows = pd.read_csv(batch / "comparison.csv")
    assert len(rows) == 8 and set(rows["seed"]) == {32, 33} and set(rows["data_seed"]) == {31}
    assert set(rows["base_model"]) == set(names)
    assert len([event for event in events if event[0] == "complete"]) == 8
    assert not (batch / "ablation_summary.json").exists()
    original = worker.run_model
    calls = []

    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("deliberate repeat failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(worker, "run_model", fail_second)
    with pytest.raises(RuntimeError, match="deliberate repeat failure"):
        worker.execute({**common, "action": "train", "directory": str(tmp_path / "failed"),
            "prepared_directory": str(prepared), "execute_training": True})
    assert len(calls) == 2 and (tmp_path / "failed/tiny_run_00.zip").is_file()
    assert "deliberate repeat failure" in (tmp_path / "failed/failure.json").read_text()


def test_repeat_settings_work_in_fresh_source_workers(github_checkout, tmp_path):
    notebook = json.loads((ROOT / "notebooks/Benchmark_Repeats_Kaggle.ipynb").read_text())
    setup = github_checkout("".join(notebook["cells"][2]["source"]))
    script = setup + '\nimport os, runpy\nclient=runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]\n'
    script += 'request=' + repr({**request(), "action": "settings"}) + '\n'
    script += '''request["source_provenance"]=SOURCE_PROVENANCE
first=client(ROOT, request, module="experiments.multidataset_worker", jobs_directory=ROOT.parent / "jobs")
second=client(ROOT, request, module="experiments.multidataset_worker", jobs_directory=ROOT.parent / "jobs")
assert first["total_runs"] == 44 and first["plan"] == second["plan"]
assert first["worker_pid"] != os.getpid() and first["worker_pid"] != second["worker_pid"]
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", "-"], input=script,
        text=True, capture_output=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
