import ast
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from experiments import ukhousing_worker
from experiments.build_ukhousing_comparison_colab import build_notebook
from experiments.default_hyperparameters import ukhousing_comparison_defaults, ukhousing_defaults
from experiments.run_selection import select_runs

ROOT = Path(__file__).resolve().parents[1]
DATA = {"seed": 31, "train_fraction": 0.64, "validation_fraction": 0.16}


def test_approved_sizes_and_protocol():
    runs = ukhousing_comparison_defaults(device="cpu")
    assert len(runs) == 10
    assert [(c.model, c.patches) for _, c in runs if c.model in ("gpw", "prospar")] == [
        ("gpw", 8), ("gpw", 4), ("prospar", 8), ("prospar", 4)]
    for _, config in runs:
        assert config.inducing_points == 1024
        assert (config.epochs, config.warmup_epochs) == (100, 20)
        if config.model == "prospar":
            assert asdict(config.prospar) == {"temperatures": 1.0, "exponent": 1.0,
                                             "inducing_nuggets": 0.01, "prior_jitter": 0.0}
    assert [c.vnngp_neighbors for _, c in runs if c.model == "vnngp"] == [32, 256]
    assert ukhousing_defaults(device="cpu").exact_training_points == "all"


def test_multiple_gpw_runs_have_explicit_eight_patch_reference():
    defaults = ukhousing_comparison_defaults(device="cpu")
    selected = select_runs(defaults, models=[name for name, _ in defaults], overrides={})
    for _, config in selected:
        if config.model == "patchwork":
            assert config.exact_gpw_reference == {"inducing_points": 1024, "patches": 8,
                                                  "seed": 31, "threads": 2}
    selected = select_runs(defaults, models=["Standard Patchwork (PC tree)"],
                           overrides={"GPW": {"inducing_points": 2048}})
    assert selected[0][1].exact_gpw_reference["inducing_points"] == 2048


def test_new_notebook_is_reproducible_launcher_only(tmp_path):
    (tmp_path / "notebooks").mkdir()
    notebook = json.loads(build_notebook(tmp_path).read_text())
    assert notebook == json.loads((ROOT / "notebooks/UKHousing_Comparison_Colab.ipynb").read_text())
    source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for c in notebook["cells"]:
        if c["cell_type"] == "code":
            assert not c["outputs"]
            compile("".join(c["source"]), "colab", "exec")
    tree = ast.parse(source)
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(name and name.startswith(("pwgp", "experiments")) for name in imports)
    assert "live.update(Image" in source
    assert 'module="experiments.ukhousing_worker"' in source
    assert "class " not in source
    assert "_OISST_STAGES" in source


def test_worker_settings_in_fresh_process(github_checkout, tmp_path):
    notebook = json.loads((ROOT / "notebooks/UKHousing_Comparison_Colab.ipynb").read_text())
    setup = github_checkout("".join(notebook["cells"][2]["source"]))
    script = setup + '''
import os, runpy
client = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]
request = {"action": "settings", "source_provenance": SOURCE_PROVENANCE, "device": "cpu",
           "models": ["GPW", "GPW 4 x 256", "ProSpar 4 x 256"], "overrides": {}}
result = client(ROOT, request, module="experiments.ukhousing_worker")
assert result["worker_pid"] != os.getpid()
assert [row["config"]["patches"] for row in result["runs"]] == [8, 4, 4]
exec(''' + repr(setup) + ''')
second = client(ROOT, request, module="experiments.ukhousing_worker")
assert second["worker_pid"] != result["worker_pid"]
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", "-"], input=script,
                            text=True, capture_output=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from pwgp.data import ukhousing

    monkeypatch.setattr(ukhousing_worker, "source_snapshot", lambda provenance: (tmp_path, b"source"))
    points = np.stack(np.meshgrid(np.linspace(0.02, 0.98, 20), np.linspace(0.02, 0.98, 20)), -1).reshape(-1, 2)
    x = points * [7.5, 5.5] + [-5.7, 50]
    y = np.sin(8 * points[:, 0]) + np.cos(5 * points[:, 1])
    monkeypatch.setattr(ukhousing, "prepare_ukhousing", lambda root: (x, y, {"test": "synthetic"}))
    directory = tmp_path / "prepared"
    ukhousing_worker.execute({"action": "prepare", "source_provenance": {},
                             "directory": str(directory), "data": DATA})
    return directory


def test_prepared_reuse_rejects_changes(prepared):
    split = ukhousing_worker.load_prepared(prepared, DATA)
    assert len(split["train"]) == 256
    with pytest.raises(ValueError, match="configuration differs"):
        ukhousing_worker.load_prepared(prepared, {**DATA, "seed": 7})
    (prepared / "data.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        ukhousing_worker.load_prepared(prepared, DATA)


def test_all_ten_configurations_save_archive_and_report(prepared, tmp_path, monkeypatch):
    defaults = ukhousing_comparison_defaults(device="cpu")
    common = {"epochs": 1, "warmup_epochs": 0, "inducing_points": 64,
              "q_rank": 2, "stitches_per_edge": 2, "batch_size": 64,
              "prediction_batch_size": 32, "snapshot_every": 1,
              "vnngp_batch_size": 32, "vnngp_neighbors": 4}
    overrides = {name: common for name, _ in defaults}
    runs = select_runs(defaults, models=list(overrides), overrides=overrides)
    events = []
    monkeypatch.setattr(ukhousing_worker, "emit", lambda kind, **values: events.append({"kind": kind, **values}))
    batch = tmp_path / "batch"
    ukhousing_worker.execute({"action": "train", "source_provenance": {}, "data": DATA,
        "prepared_directory": str(prepared), "directory": str(batch),
        "runs": [{"name": name, "config": asdict(config)} for name, config in runs]})
    assert len([e for e in events if e["kind"] == "snapshot"]) == 10
    completed = [e for e in events if e["kind"] == "complete"]
    assert len(completed) == 10
    assert all(Path(e["archive"]).is_file() for e in completed)
    assert json.loads((batch / "status.json").read_text())["state"] == "complete"


def test_failure_stops_later_runs(prepared, tmp_path, monkeypatch):
    from experiments import ukhousing_benchmark

    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("deliberate model failure")

    monkeypatch.setattr(ukhousing_benchmark, "run_ukhousing", fail)
    runs = ukhousing_comparison_defaults(device="cpu")[:2]
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"action": "train", "source_provenance": {}, "data": DATA,
        "prepared_directory": str(prepared), "directory": str(tmp_path / "failed_batch"),
        "runs": [{"name": name, "config": asdict(config)} for name, config in runs]}))
    with pytest.raises(RuntimeError, match="deliberate model failure"):
        ukhousing_worker.main(request)
    assert calls == [1]
    assert "deliberate model failure" in (tmp_path / "failure.json").read_text()
