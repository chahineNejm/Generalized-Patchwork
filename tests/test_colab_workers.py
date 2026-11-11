import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.colab_source import github_bootstrap
from tests import test_partial_reruns

prepared = test_partial_reruns.prepared


def run_script(source, tmp_path):
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", "-"], input=source, text=True, encoding="utf-8",
                            cwd=tmp_path, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_setup_reuses_checkout_and_fresh_workers_after_source_update(github_checkout, tmp_path):
    source = github_checkout(github_bootstrap("/content/test", isolated=True))
    script = f'''
import os, runpy
code = {source!r}
exec(code)
parent_pid = os.getpid()
from experiments.default_hyperparameters import oisst_defaults
assert oisst_defaults(device="cpu")[1][0][1].epochs == 100
client = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]
request = {{"action": "settings", "device": "cpu", "models": ["GPW", "SVGP"],
           "overrides": {{"GPW": {{"epochs": 2, "warmup_epochs": 1}}}}, "data_overrides": {{}}}}
first = client(ROOT, {{**request, "source_provenance": SOURCE_PROVENANCE}})
assert first["worker_pid"] != parent_pid
assert [row["name"] for row in first["runs"]] == ["GPW", "SVGP"]
assert first["runs"][0]["config"]["epochs"] == 2
(ROOT / "data").mkdir()
(ROOT / "data/keep.bin").write_bytes(b"dataset")
(ROOT / "results/keep.bin").write_bytes(b"results")
exec(code)
assert (ROOT / "data/keep.bin").read_bytes() == b"dataset"
assert (ROOT / "results/keep.bin").read_bytes() == b"results"
remote = Path(REPOSITORY)
path = remote / "experiments/default_hyperparameters.py"
original = path.read_text()
assert original.count("epochs=100, warmup_epochs=20, inducing_points=512") == 1
path.write_text(original.replace("epochs=100, warmup_epochs=20, inducing_points=512",
                                "epochs=101, warmup_epochs=20, inducing_points=512"))
for args in (["add", "experiments/default_hyperparameters.py"],
             ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.org", "commit", "-m", "new defaults"]):
    subprocess.run(["git", "-C", str(remote), *args], check=True, capture_output=True)
old_provenance = dict(SOURCE_PROVENANCE)
exec(code)
assert SOURCE_PROVENANCE["commit"] != old_provenance["commit"]
assert oisst_defaults(device="cpu")[1][0][1].epochs == 100
second = client(ROOT, {{**request, "source_provenance": SOURCE_PROVENANCE}})
assert second["worker_pid"] != parent_pid
assert second["worker_pid"] != first["worker_pid"]
assert second["runs"][1]["config"]["epochs"] == 101
assert (ROOT / "data/keep.bin").read_bytes() == b"dataset"
assert (ROOT / "results/keep.bin").read_bytes() == b"results"
'''
    run_script(script, tmp_path)


@pytest.mark.parametrize("damage", ["tracked", "untracked", "remote"])
def test_reuse_rejects_source_damage_without_deleting_data(github_checkout, tmp_path, damage):
    source = github_checkout(github_bootstrap("/content/test", isolated=True))
    script = f'''
code = {source!r}
exec(code)
(ROOT / "data").mkdir()
(ROOT / "data/keep.bin").write_bytes(b"keep")
damage = {damage!r}
if damage == "tracked":
    (ROOT / "README.md").write_text("local edit")
elif damage == "untracked":
    (ROOT / "experiments/untracked_source.py").write_text("value = 1")
else:
    subprocess.run(["git", "-C", str(ROOT), "remote", "set-url", "origin", "wrong-repository"], check=True)
try:
    exec(code)
except (ValueError, subprocess.CalledProcessError):
    assert (ROOT / "data/keep.bin").read_bytes() == b"keep"
else:
    raise AssertionError("Damaged checkout was silently reused")
'''
    run_script(script, tmp_path)


def test_worker_failure_is_recorded_and_exit_status_is_checked(github_checkout, tmp_path):
    source = github_checkout(github_bootstrap("/content/test", isolated=True))
    script = source + '''
import runpy
client = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]
for action, provenance in (("invalid", SOURCE_PROVENANCE),
                           ("settings", {**SOURCE_PROVENANCE, "commit": "wrong-commit"})):
    try:
        client(ROOT, {"action": action, "source_provenance": provenance})
    except subprocess.CalledProcessError:
        pass
    else:
        raise AssertionError("Failed worker returned successfully")
jobs = list((ROOT / "results/worker_jobs").iterdir())
assert len(jobs) == 2
for job in jobs:
    failure = json.loads((job / "failure.json").read_text())
    assert "Traceback" in failure["traceback"]
    assert json.loads((job / "status.json").read_text())["state"] == "failed"
    assert (job / "client_failure.txt").exists()
    assert "ValueError" in (job / "worker.log").read_text()
    assert not (job / "result.json").exists()
'''
    run_script(script, tmp_path)


def test_worker_training_streams_visuals_and_reanalysis_matches(prepared, github_checkout, tmp_path):
    from dataclasses import asdict

    config, _, directory, _ = prepared
    source = github_checkout(github_bootstrap("/content/test", isolated=True))
    script = source + f'''
import runpy
client = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]
provenance = {{"source_provenance": SOURCE_PROVENANCE}}
settings = client(ROOT, {{**provenance, "action": "settings", "device": "cpu", "models": ["SVGP"],
    "overrides": {{"SVGP": {{"epochs": 1, "warmup_epochs": 0, "inducing_points": 4,
                            "batch_size": 64, "prediction_batch_size": 16, "snapshot_every": 1}}}},
    "data_overrides": {{}}}})
data = {asdict(config)!r}
client(ROOT, {{**provenance, "action": "load", "data": data, "directory": {str(directory)!r}}})
events = []
batch = client(ROOT, {{**provenance, "action": "train", "data": data, "runs": settings["runs"],
    "prepared_directory": {str(directory)!r}, "directory": str(ROOT / "results/test_batch")}}, on_event=events.append)
assert [e["epoch"] for e in events if e["kind"] == "snapshot"] == [0, 1]
completed = [e for e in events if e["kind"] == "complete"]
assert len(completed) == 1
path = Path(completed[0]["directory"])
assert (path / "visuals/progress.png").stat().st_size > 0
assert Path(completed[0]["archive"]).is_file()
analysis = client(ROOT, {{**provenance, "action": "analysis", "analysis_source": "latest_batch",
    "batch_directory": batch["directory"], "saved_runs": [], "directory": str(ROOT / "results/test_analysis")}})
original = json.loads((path / "metrics.json").read_text())
recomputed = json.loads((Path(analysis["directory"]) / "00/metrics.json").read_text())
assert original["rmse"] == recomputed["rmse"] and original["nlpd"] == recomputed["nlpd"]
assert len({{settings["worker_pid"], batch["worker_pid"], analysis["worker_pid"]}}) == 3
'''
    run_script(script, tmp_path)


def test_notebook_does_not_import_experiment_models_in_parent():
    import ast

    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "notebooks/OISST_Colab.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.walk(ast.parse("".join(cell["source"]))):
            if isinstance(node, ast.ImportFrom):
                assert not node.module.startswith(("experiments", "pwgp", "torch"))
            elif isinstance(node, ast.Import):
                assert not any(alias.name.startswith(("experiments", "pwgp", "torch")) for alias in node.names)
