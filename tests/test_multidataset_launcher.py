import ast
import json
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments import multidataset_worker as worker
from experiments.build_multidataset_kaggle import build_notebook
from experiments.default_hyperparameters import (
    multidataset_defaults,
    multidataset_notebook_defaults,
)
from experiments.multidataset_data import (
    DatasetSpec,
    load_prepared,
    prepare_dataset,
    verify_manual_inputs,
)
from pwgp.data.benchmarks import synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]


def request(**values):
    return {"device": "cpu", "input_root": "manual", "datasets": ["california"],
        "models": {}, "model_overrides": {}, "data_overrides": {}, "epochs": 3,
        "warmup_epochs": 1, "save_inducing_positions": True, **values}


def test_catalog_all_approved_capacities_and_schedules():
    data, runs, deferred = multidataset_defaults(device="cpu", input_root="manual")
    assert len(data) == 22 and len(deferred) == 2
    plan = worker.make_plan(request(datasets=list(data)))
    assert (plan["total_runs"], plan["total_epochs"]) == (179, 537)
    assert all(row["config"]["epochs"] == 3 for row in plan["plan"]["runs"])
    for name, spec in data.items():
        for _, config in runs[name]:
            if config.model == "prospar":
                assert config.prospar.temperatures == .1 and config.prospar.exponent == 2.
                assert config.prospar.inducing_nuggets == .01 and config.prospar.prior_jitter == 0.
        if spec.synthetic is not None:
            assert spec.synthetic["test_points"] == spec.synthetic["training_points"] // 4
            assert spec.synthetic["noise_variance"] == 1e-8
            assert spec.synthetic["g_coefficients"] == [i / 2 for i in range(spec.dimension)]
            assert {c.inducing_points for _, c in runs[name]} == {1024}
        if name not in ("oisst", "modis"):
            assert [c.vnngp_neighbors for _, c in runs[name] if c.model == "vnngp"] == [32, 128]
    assert {c.inducing_points for _, c in runs["flight"]} == {2048}
    assert {c.inducing_points for _, c in runs["california"]} == {512}


def test_no_vnngp_all_datasets_flight_last_preserves_other_configs():
    data, _, _ = multidataset_defaults(device="cpu", input_root="manual")
    datasets = [name for name in data if name != "flight"] + ["flight"]
    complete = worker.make_plan(request(datasets=datasets))["plan"]
    result = worker.make_plan(request(datasets=datasets, excluded_model_types=["vnngp"]))
    assert result["total_runs"] == 137 and result["total_epochs"] == 411
    assert result["plan"]["data"][-1]["name"] == "flight"
    assert result["plan"]["excluded_model_types"] == ["vnngp"]
    assert result["plan"]["runs"] == [r for r in complete["runs"] if r["config"]["model"] != "vnngp"]
    assert result["plan"]["runs"][-1]["dataset"] == "flight"
    with pytest.raises(ValueError, match="conflicts"):
        worker.make_plan(request(excluded_model_types=["vnngp"], models={"california": ["VNNGP K32"]}))


def test_reduced_batch_retains_both_sizes_and_only_pc_tree():
    defaults = multidataset_notebook_defaults(device="cpu", input_root="manual")
    result = worker.make_plan(request(**defaults, excluded_model_types=["vnngp"]))
    assert (len(result["plan"]["data"]), result["total_runs"], result["total_epochs"]) == (11, 61, 6100)
    assert "flight" not in defaults["datasets"]
    synthetic = [name for name in defaults["datasets"] if name.startswith(("michalewicz", "g_function"))]
    assert synthetic == ["michalewicz_d5_n50000", "g_function_d5_n50000"]
    full = worker.make_plan(request(datasets=defaults["datasets"], epochs=100, warmup_epochs=20))["plan"]
    expected = [r for r in full["runs"] if r["config"]["model"] != "vnngp" and
                (r["config"]["model"] != "patchwork" or r["config"]["exact_partition"] == "pc_tree")]
    assert result["plan"]["runs"] == expected
    assert result["plan"]["early_stopping"] == {"patience": 10, "relative_min_improvement": .001}


@pytest.mark.parametrize("change,match", [
    ({"datasets": ["precipitation"]}, "deferred"),
    ({"datasets": []}, "nonempty"),
    ({"datasets": ["california", "california"]}, "distinct"),
    ({"epochs": 1}, "no automatic"),
    ({"epochs": 2.5}, "integer"),
    ({"save_inducing_positions": "yes"}, "boolean"),
    ({"models": {"california": ["not a model"]}}, "Unknown"),
    ({"data_overrides": {"california": {"dimension": 2}}}, "schema"),
    ({"model_overrides": {"california": {"SVGP": {"epochs": 2.5}}}}, "integers"),
    ({"excluded_model_types": ["unknown"]}, "known model types"),
    ({"excluded_model_types": "vnngp"}, "known model types"),
    ({"excluded_model_types": ["vnngp", "vnngp"]}, "distinct"),
])
def test_invalid_plans_do_not_substitute_values(change, match):
    with pytest.raises((ValueError, TypeError, KeyError), match=match):
        worker.make_plan(request(**change))


def test_selected_baseline_keeps_explicit_gpw_reference_and_overrides():
    result = worker.make_plan(request(models={"california": ["Standard Patchwork (PC tree)"]},
        model_overrides={"california": {"GPW": {"inducing_points": 256},
                                        "Standard Patchwork (PC tree)": {"epochs": 2}}}))
    config = result["plan"]["runs"][0]["config"]
    assert config["exact_gpw_reference"]["inducing_points"] == 256
    assert config["epochs"] == 2


@pytest.mark.parametrize("name", ["michalewicz", "g_function"])
def test_synthetic_exact_formula_noiseless_and_deterministic(name):
    options = {"dimension": 3, "training_points": 80, "test_points": 20, "seed": 31,
               "noise_variance": 0., "michalewicz_m": 10, "g_coefficients": [0., .5, 1.]}
    data = synthetic_dataset(name, **options)
    repeat = synthetic_dataset(name, **options)
    np.testing.assert_array_equal(data.x, repeat.x)
    x = data.x
    expected = (-(np.sin(x) * np.sin(np.arange(1, 4) * x*x / np.pi)**20).sum(1)
                if name == "michalewicz" else ((np.abs(4*x-2) + [0., .5, 1.]) / [1., 1.5, 2.]).prod(1))
    np.testing.assert_allclose(data.y, expected, atol=1e-14)


def tiny_spec():
    return DatasetSpec("tiny", "g_function", 3, .8, 0., 31,
        synthetic={"training_points": 160, "test_points": 40, "noise_variance": 1e-8,
                   "michalewicz_m": 10, "g_coefficients": [0., .5, 1.]})


def test_prepared_data_identity_checksums_and_eda(tmp_path):
    spec = tiny_spec()
    directory = prepare_dataset(spec, directory=tmp_path / "prepared", data_root=tmp_path)
    loaded = load_prepared(spec, directory)
    assert len(loaded["train"]) == 160 and len(loaded["test"]) == 40
    assert (directory / "visuals/distribution.png").is_file()
    assert json.loads((directory / "visuals/summary.json").read_text())["dimensions"] == 3
    with pytest.raises(ValueError, match="identity"):
        load_prepared(replace(spec, seed=32), directory)
    (directory / "data.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_prepared(spec, directory)


def test_missing_manual_sources_raise_before_preparation(tmp_path):
    data, _, _ = multidataset_defaults(device="cpu", input_root=tmp_path)
    with pytest.raises(FileNotFoundError, match="no data substitutions"):
        verify_manual_inputs([data["flight"], data["icethick"]])


def test_notebook_reproducible_no_model_code_and_training_gated(tmp_path):
    (tmp_path / "notebooks").mkdir()
    notebook = json.loads(build_notebook(tmp_path).read_text())
    assert notebook == json.loads((ROOT / "notebooks/All_Datasets_Kaggle.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            compile("".join(cell["source"]), "kaggle", "exec")
    imports = [n.module for n in ast.walk(ast.parse(code)) if isinstance(n, ast.ImportFrom)]
    assert not any(n and n.startswith(("experiments", "pwgp")) for n in imports)
    assert "RUN_TRAINING = False" in code and "live.update(Image" in code
    assert "EPOCHS = 100" in code and "WARMUP_EPOCHS = 20" in code
    assert "_OISST_STAGES" in code and "class " not in code
    settings = next("".join(c["source"]) for c in notebook["cells"] if
                    c["cell_type"] == "code" and "EXCLUDED_MODEL_TYPES =" in "".join(c["source"]))
    assignments = {n.targets[0].id: n.value for n in ast.walk(ast.parse(settings))
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert len(ast.literal_eval(assignments["DATASETS"])) == 11
    assert "flight" not in ast.literal_eval(assignments["DATASETS"])
    assert ast.literal_eval(assignments["EXCLUDED_MODEL_TYPES"]) == ["vnngp"]
    assert ast.literal_eval(assignments["EARLY_STOPPING"]) == {"patience": 10, "relative_min_improvement": .001}
    assert 'display(FileLink(event["archive"]))' in code


def test_settings_in_fresh_worker_without_reconnect(github_checkout, tmp_path):
    notebook = json.loads((ROOT / "notebooks/All_Datasets_Kaggle.ipynb").read_text())
    setup = github_checkout("".join(notebook["cells"][2]["source"]))
    script = setup + '\nimport os, runpy\nclient=runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]\n'
    script += 'request=' + repr({**request(models={"california": ["SVGP"]}), "action": "settings"}) + '\n'
    script += '''request["source_provenance"]=SOURCE_PROVENANCE
first=client(ROOT, request, module="experiments.multidataset_worker", jobs_directory=ROOT.parent / "logs")
second=client(ROOT, request, module="experiments.multidataset_worker", jobs_directory=ROOT.parent / "logs")
assert first["worker_pid"] != os.getpid() and second["worker_pid"] != first["worker_pid"]
assert first["total_runs"] == 1 and first["plan"] == second["plan"]
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", "-"], input=script,
                            text=True, capture_output=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_short_pipeline_archives_and_loud_failure(tmp_path, monkeypatch):
    from dataclasses import asdict

    monkeypatch.setattr(worker, "source_snapshot", lambda value: (tmp_path, b"source"))
    monkeypatch.setattr(worker, "emit", lambda *args, **kwargs: None)
    spec = tiny_spec()
    config = worker.make_plan(request(models={"california": ["SVGP", "GPW"]},
        model_overrides={"california": {n: {"inducing_points": 24, "patches": 4, "q_rank": 2,
            "batch_size": 64, "snapshot_every": 3, "prediction_batch_size": 32} for n in ("SVGP", "GPW")}}))["plan"]
    config["data"] = [asdict(spec)]
    for row in config["runs"]:
        row["dataset"] = "tiny"
    common = {"source_provenance": {}, "plan": config, "plan_sha256": worker.plan_hash(config)}
    prepared = tmp_path / "prepared"
    worker.execute({**common, "action": "prepare", "directory": str(prepared), "data_root": str(tmp_path)})
    train = {**common, "action": "train", "directory": str(tmp_path / "batch"),
             "prepared_directory": str(prepared), "execute_training": False}
    with pytest.raises(RuntimeError, match="disabled"):
        worker.execute(train)
    assert not (tmp_path / "batch").exists()
    worker.execute({**train, "execute_training": True})
    assert (tmp_path / "batch/tiny_run_00.zip").exists()
    destination = tmp_path / "results.zip"
    worker.execute({**common, "action": "archive", "directory": str(tmp_path / "batch"), "destination": str(destination)})
    with zipfile.ZipFile(destination) as archive:
        assert "comparison.csv" in archive.namelist()
        assert "tiny/data_visuals/summary.json" in archive.namelist()
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("deliberate failure")

    monkeypatch.setattr(worker, "run_model", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        worker.execute({**train, "execute_training": True, "directory": str(tmp_path / "failed")})
    assert calls == [1]
    assert "deliberate failure" in (tmp_path / "failed/failure.json").read_text()

    def complete_then_fail(*args, **kwargs):
        import shutil

        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("second model failed")
        shutil.copytree(tmp_path / "batch/tiny/run_00", kwargs["directory"])
        return kwargs["directory"]

    calls.clear()
    monkeypatch.setattr(worker, "run_model", complete_then_fail)
    partial = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="second model"):
        worker.execute({**train, "execute_training": True, "directory": str(partial)})
    assert len(calls) == 2
    with zipfile.ZipFile(partial / "tiny_run_00.zip") as archive:
        assert "run_00/metrics.json" in archive.namelist()
    import pandas as pd

    assert len(pd.read_csv(partial / "comparison.csv")) == 1
    assert json.loads((partial / "status.json").read_text())["state"] == "failed"
