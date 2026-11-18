"""Main-table selection, finite queue continuation and a shared session deadline."""

import ast
import json
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from experiments import vnngp_cpu_schedule as schedule
from experiments import vnngp_cpu_worker as cpu
from experiments.build_vnngp_cpu_kaggle import build_notebook
from experiments.colab_source import github_bootstrap
from experiments.default_hyperparameters import vnngp_cpu_table_defaults


def request(**changes):
    values = {"datasets": ["california", "protein"], "seeds": [31, 32], "input_root": "unused",
              "data_overrides": {}, "overrides": {"epochs": 3, "warmup_epochs": 1,
              "neighbors": 3, "batch_size": 5, "threads": 1}}
    return {**values, **changes}


def mock_data(monkeypatch):
    x = torch.linspace(.05, .95, 17, dtype=torch.float64)[:, None]
    y = torch.sin(6 * x[:, 0])
    split = {"x": x, "y": y, "raw_x": x.numpy(), "raw_y": y.numpy(),
             "train": np.arange(12), "validation": np.array([], dtype=np.int64),
             "test": np.arange(12, 17), "target_mean": 0., "target_std": 1.,
             "target_offset": np.zeros(len(y)), "score_splits": ["test"], "seed": 31}

    def prepare(_spec, *, directory, data_root):
        directory.mkdir()
        (directory / "prepared.json").write_text('{}')

    def load(spec, _directory):
        return {**split, "dataset": spec.name}

    monkeypatch.setattr(cpu, "prepare_dataset", prepare)
    monkeypatch.setattr(cpu, "load_prepared", load)
    monkeypatch.setattr(schedule, "load_prepared", load)
    monkeypatch.setattr(cpu, "source_snapshot", lambda _: (None, b"source"))


def execution(tmp_path, values=None):
    settings = schedule.make_schedule(request() if values is None else values)
    return {"action": "start", "execute_training": True, "plan": settings["plan"],
            "plan_sha256": settings["plan_sha256"], "source_provenance": {"commit": "test"},
            "directory": str(tmp_path / "job"), "data_root": str(tmp_path / "data"),
            "archive": str(tmp_path / "paused.zip")}


def test_table_defaults_and_notebook(tmp_path):
    defaults = vnngp_cpu_table_defaults()
    names = ["california", "protein", "tco", "oisst", "tco_l2", "icethick", "sarcos", "modis",
             "flight", "michalewicz_d5_n50000", "g_function_d5_n50000"]
    assert defaults["datasets"] == names
    settings = schedule.make_schedule(request(datasets=names, seeds=defaults["seeds"],
        data_overrides=defaults["data_overrides"], overrides={}))
    assert settings["total_runs"] == 33 and settings["maximum_epochs"] == 3300
    parts = settings["plan"]["dataset_plans"]
    assert next(p for p in parts if p["data"]["name"] == "flight")["data"]["subset_size"] == 800000
    for part in parts:
        assert [c["seed"] for c in part["configs"]] == [31, 32, 33]
        assert part["data"]["seed"] == 31
        assert all(c["vnngp_neighbors"] == 32 for c in part["configs"])
        assert all(c["epochs"] == 100 and c["warmup_epochs"] == 20 for c in part["configs"])
    notebook = json.loads(build_notebook(tmp_path).read_text())
    sources = "".join("".join(c["source"]) for c in notebook["cells"])
    assert "DATASET = None" not in sources
    assert f"DATASETS = {names!r}" in sources and 'SEEDS = [31, 32, 33]' in sources
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


@pytest.mark.parametrize("changes", [{"datasets": []}, {"datasets": ["tco", "tco"]},
    {"datasets": ["unknown"]}, {"dataset": "tco"}, {"data_overrides": {"flight": {"subset_size": 800000}}}])
def test_invalid_selection_raises(changes):
    with pytest.raises(ValueError):
        schedule.make_schedule(request(**changes))


def test_multi_dataset_pause_resume_preserves_completed_prefix(tmp_path, monkeypatch):
    mock_data(monkeypatch)
    calls, events = [], []
    build, training = cpu.build_model, cpu.train_chunk

    def build_model(config, split):
        calls.append((split["dataset"], config.seed))
        return build(config, split)

    count = 0

    def pause_second(*args, **kwargs):
        nonlocal count
        count += 1
        return training(*args, **kwargs, max_batches=1 if count == 2 else None)

    monkeypatch.setattr(cpu, "build_model", build_model)
    monkeypatch.setattr(cpu, "train_chunk", pause_second)
    monkeypatch.setattr(cpu, "emit", lambda *args, **kwargs: events.append(kwargs))
    values = execution(tmp_path)
    result = cpu.execute(values)
    assert result["state"] == "paused" and result["next_dataset_index"] == 0
    assert calls == [("california", 31), ("california", 32)]
    with zipfile.ZipFile(events[0]["archive"]) as archive:
        assert json.loads(archive.read("plan.json"))["datasets"] == ["california", "protein"]
        assert "comparison.csv" in archive.namelist()
        assert "california/seed31/metrics.json" in archive.namelist()
    changed = json.loads(json.dumps(values["plan"]))
    changed["datasets"].reverse()
    changed["dataset_plans"].reverse()
    with pytest.raises(ValueError, match="schedule changed"):
        cpu.execute({**values, "action": "resume", "plan": changed,
            "plan_sha256": schedule.plan_hash(changed), "directory": str(tmp_path / "changed"),
            "resume_archive": result["archive"]})
    assert calls == [("california", 31), ("california", 32)]
    monkeypatch.setattr(cpu, "train_chunk", training)
    resumed = cpu.execute({**values, "action": "resume", "directory": str(tmp_path / "continued"),
        "resume_archive": result["archive"], "archive": str(tmp_path / "completed.zip")})
    assert resumed["state"] == "complete" and resumed["next_dataset_index"] == 2
    assert calls == [("california", 31), ("california", 32), ("california", 32), ("protein", 31), ("protein", 32)]
    rows = pd.read_csv(tmp_path / "continued/comparison.csv")
    assert len(rows) == 4 and set(rows.dataset) == {"california", "protein"}
    assert set(rows.seed) == {31, 32} and len(events) == 4
    with pytest.raises(ValueError, match="already complete"):
        cpu.execute({**values, "action": "resume", "directory": str(tmp_path / "finished"),
            "resume_archive": resumed["archive"], "archive": str(tmp_path / "forbidden.zip")})


def test_missing_inputs_and_training_failures_block_later_datasets(tmp_path, monkeypatch):
    mock_data(monkeypatch)
    values = execution(tmp_path)
    prepared = []

    def missing(_):
        raise FileNotFoundError("missing manual data")

    monkeypatch.setattr(schedule, "verify_manual_inputs", missing)
    prepare = cpu.prepare_dataset
    monkeypatch.setattr(cpu, "prepare_dataset", lambda *args, **kwargs: prepared.append(1))
    with pytest.raises(FileNotFoundError, match="missing manual"):
        cpu.execute(values)
    assert not prepared
    monkeypatch.setattr(schedule, "verify_manual_inputs", lambda _: None)
    monkeypatch.setattr(cpu, "prepare_dataset", prepare)
    built = []

    def fail_model(config, split):
        built.append(split["dataset"])
        raise RuntimeError("deliberate training failure")

    monkeypatch.setattr(cpu, "build_model", fail_model)
    path = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate"):
        cpu.execute({**values, "directory": str(path)})
    assert built == ["california"]
    assert "Traceback" in json.loads((path / "failure.json").read_text())["traceback"]
    assert not (path / "protein").exists()


def test_one_shared_deadline_across_datasets(tmp_path, monkeypatch):
    mock_data(monkeypatch)
    now, deadlines = [0.0], []
    monkeypatch.setattr(schedule.time, "monotonic", lambda: now[0])

    def finish(root, part, *, deadline, on_seed_saved, **_):
        deadlines.append(deadline)
        for config in part["configs"]:
            seed = root / f"seed{config['seed']}"
            seed.mkdir()
            cpu.write_json(seed / "metrics.json", {"rmse_standardized": 1.0})
        cpu.write_json(root / "progress.json", {"state": "complete", "next_seed_index": len(part["configs"])})
        on_seed_saved()
        now[0] += 600
        return {"state": "complete"}

    monkeypatch.setattr(cpu, "run_session", finish)
    values = execution(tmp_path, request(datasets=["california", "protein", "tco"], seeds=[31],
        overrides={"session_hours": .25}))
    result = cpu.execute(values)
    assert deadlines == [900, 900]
    assert result["state"] == "paused" and result["next_dataset_index"] == 2
    assert not (tmp_path / "job/tco").exists()
    completed = cpu.execute({**values, "action": "resume", "directory": str(tmp_path / "continued"),
        "resume_archive": result["archive"], "archive": str(tmp_path / "completed.zip")})
    assert deadlines == [900, 900, 2100]
    assert completed["state"] == "complete" and completed["next_dataset_index"] == 3


def test_table_settings_in_fresh_verified_worker(github_checkout, tmp_path):
    setup = github_checkout(github_bootstrap("/unused", isolated=True))
    values = vnngp_cpu_table_defaults()
    settings = request(datasets=values["datasets"], seeds=values["seeds"],
                       data_overrides=values["data_overrides"], overrides={})
    script = setup + '\nimport runpy\n'
    script += 'client=runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]\n'
    script += 'request=' + repr({**settings, "action": "settings"}) + '\n'
    script += '''request["source_provenance"]=SOURCE_PROVENANCE
result=client(ROOT, request, module="experiments.vnngp_cpu_worker", jobs_directory=ROOT.parent / "jobs")
assert result["total_runs"] == 33
assert len(result["plan"]["datasets"]) == 11
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", "-"], input=script,
        text=True, capture_output=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
