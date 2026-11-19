"""CPU continuation must reproduce uninterrupted VNNGP optimization exactly."""

import ast
import json
import random
import subprocess
import sys
import zipfile
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.build_vnngp_cpu_kaggle import build_notebook
from experiments.colab_source import github_bootstrap
from experiments.ukhousing_benchmark import UKConfig, train
from experiments.vnngp_cpu_training import make_stopper, train_chunk
from experiments.vnngp_cpu_worker import archive_session, make_plan, restore_archive
from pwgp.models.vnngp import VNNGP

CONTROL = {"patience": 10, "relative_min_improvement": 0.001}


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(31)
    random.seed(31)
    np.random.seed(31)
    x = torch.linspace(0.05, .95, 17, dtype=torch.float64)[:, None]
    config = UKConfig(model="vnngp", device="cpu", seed=31, epochs=4, warmup_epochs=1,
                      vnngp_neighbors=3, vnngp_batch_size=5, threads=1)
    model = VNNGP(x, neighbors=3, batch_size=5)
    return model, torch.sin(x[:, 0] * 6), config


def run(model, y, config, path, *, resume=False, limit=None, identity="same"):
    return train_chunk(model, y, config, stopper=make_stopper(CONTROL, config.warmup_epochs),
        checkpoint=path, identity=identity, resume=resume, deadline=float("inf"),
        checkpoint_seconds=300, description="test:VNNGP", progress_every_epochs=20,
        on_epoch=lambda *_: None, max_batches=limit)


@pytest.mark.parametrize("cut", [1, 3, 4, 5, 9])
def test_resume_matches_uninterrupted(tmp_path, cut):
    full, y, config = fixture()
    run(full, y, config, tmp_path / "full.pt")
    split, y, config = fixture()
    assert not run(split, y, config, tmp_path / "split.pt", limit=cut)
    rebuilt, y, config = fixture()
    assert run(rebuilt, y, config, tmp_path / "split.pt", resume=True)
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, rebuilt.state_dict()[key], rtol=0, atol=0)
    a = torch.load(tmp_path / "full.pt", weights_only=True)
    b = torch.load(tmp_path / "split.pt", weights_only=True)
    assert a["losses"] == b["losses"]
    assert a["stopper"] == b["stopper"]
    assert torch.equal(a["rng"]["torch"], b["rng"]["torch"])
    torch.testing.assert_close(full.predict(torch.tensor([[.42]], dtype=torch.float64))[0],
                               rebuilt.predict(torch.tensor([[.42]], dtype=torch.float64))[0], rtol=0, atol=0)


def test_matches_existing_training_loop(tmp_path):
    existing, y, config = fixture()
    split = {"x": existing.variational_strategy.inducing_points, "y": y, "train": np.arange(len(y))}
    expected = train(existing, split, config, lambda *_: False)
    candidate, y, config = fixture()
    run(candidate, y, config, tmp_path / "state.pt")
    saved = torch.load(tmp_path / "state.pt", weights_only=True)
    assert saved["losses"] == expected
    for key, value in existing.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)


def test_resume_preserves_early_stopping_and_last_model(tmp_path, monkeypatch):
    monkeypatch.setitem(CONTROL, "patience", 2)
    monkeypatch.setitem(CONTROL, "relative_min_improvement", 1e9)
    full, y, config = fixture()
    config = replace(config, epochs=8)
    run(full, y, config, tmp_path / "full.pt")
    interrupted, y, _ = fixture()
    assert not run(interrupted, y, config, tmp_path / "split.pt", limit=10)
    resumed, y, _ = fixture()
    assert run(resumed, y, config, tmp_path / "split.pt", resume=True)
    state = torch.load(tmp_path / "split.pt", weights_only=True)
    assert len(state["losses"]) == 4 and state["stopper"]["stopped"]
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def test_refuse_changed_identity_and_config(tmp_path):
    model, y, config = fixture()
    path = tmp_path / "state.pt"
    run(model, y, config, path, limit=2)
    for current_config, identity in [(config, "different"), (replace(config, epochs=5), "same")]:
        rebuilt, y, _ = fixture()
        with pytest.raises(ValueError, match="identity mismatch"):
            run(rebuilt, y, current_config, path, resume=True, identity=identity)
    with pytest.raises(FileExistsError):
        run(model, y, config, path)


def test_deadline_pauses_before_first_step(tmp_path):
    model, y, config = fixture()
    path = tmp_path / "state.pt"
    assert not train_chunk(model, y, config, stopper=make_stopper(CONTROL, config.warmup_epochs),
        checkpoint=path, identity="same", resume=False, deadline=0, checkpoint_seconds=300,
        description="test", progress_every_epochs=20, on_epoch=lambda *_: None)
    assert torch.load(path, weights_only=True)["losses"] == []
    rebuilt, y, config = fixture()
    assert run(rebuilt, y, config, path, resume=True)


def test_portable_archive_and_integrity(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "value.json").write_text('{"state":"paused"}')
    path = tmp_path / "resume.zip"
    archive_session(job, path)
    restored = restore_archive(path, tmp_path / "restored")
    assert json.loads((restored / "value.json").read_text()) == {"state": "paused"}
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as archive:
        archive.writestr("../outside", "invalid")
    with pytest.raises(ValueError, match="Unsafe"):
        restore_archive(bad, tmp_path / "bad")
    corrupt = tmp_path / "corrupt.zip"
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(corrupt, "w") as target:
        for name in source.namelist():
            target.writestr(name, "wrong" if name == "value.json" else source.read(name))
    with pytest.raises(ValueError, match="checksum mismatch"):
        restore_archive(corrupt, tmp_path / "corrupt")


def test_settings_and_notebook(tmp_path):
    request = {"dataset": "protein", "seeds": [31], "input_root": "unused",
               "data_overrides": {}, "overrides": {}}
    result = make_plan(request)
    assert result["total_runs"] == 1
    assert result["plan"]["configs"][0]["vnngp_neighbors"] == 32
    assert result["plan"]["configs"][0]["vnngp_jitter"] == 1e-4
    changed = make_plan({**request, "overrides": {"vnngp_jitter": 1e-3}})
    assert changed["plan"]["configs"][0]["vnngp_jitter"] == 1e-3
    assert result["plan"]["configs"][0]["noise_std"]**2 == pytest.approx(.6931)
    with pytest.raises(ValueError, match="distinct"):
        make_plan({**request, "seeds": [31, 31]})
    notebook = json.loads(build_notebook(tmp_path).read_text())
    for item in notebook["cells"]:
        if item["cell_type"] == "code":
            ast.parse("".join(item["source"]))
    source = "".join("".join(item["source"]) for item in notebook["cells"])
    assert 'SEEDS = [31, 32, 33]' in source
    assert 'RUN_TRAINING = False' in source
    assert 'RESUME_ARCHIVE = None' in source
    assert 'VNNGP_JITTER = 0.0001' in source
    assert '"vnngp_jitter": VNNGP_JITTER' in source


def test_worker_start_pause_resume_seeds_and_results(tmp_path, monkeypatch):
    from experiments import vnngp_cpu_worker as worker

    x = torch.linspace(.05, .95, 17, dtype=torch.float64)[:, None]
    y = torch.sin(6 * x[:, 0])
    split = {"x": x, "y": y, "raw_x": x.numpy(), "raw_y": y.numpy(),
             "train": np.arange(12), "validation": np.array([], dtype=np.int64),
             "test": np.arange(12, 17), "target_mean": 0., "target_std": 1.,
             "target_offset": np.zeros(len(y)), "score_splits": ["test"], "seed": 31}

    def prepare(_spec, *, directory, data_root):
        directory.mkdir()
        (directory / "prepared.json").write_text('{}')

    monkeypatch.setattr(worker, "source_snapshot", lambda _: (tmp_path, b"source"))
    monkeypatch.setattr(worker, "prepare_dataset", prepare)
    monkeypatch.setattr(worker, "load_prepared", lambda *_: split)
    events = []
    monkeypatch.setattr(worker, "emit", lambda *args, **kwargs: events.append(kwargs))
    training = worker.train_chunk
    monkeypatch.setattr(worker, "train_chunk", lambda *args, **kwargs: training(*args, **kwargs, max_batches=1))
    settings = make_plan({"dataset": "protein", "seeds": [31, 32], "input_root": "unused",
        "data_overrides": {}, "overrides": {"epochs": 3, "warmup_epochs": 1,
        "neighbors": 3, "batch_size": 5, "threads": 1, "vnngp_jitter": 1e-3}})
    request = {"action": "start", "execute_training": True, "plan": settings["plan"],
        "plan_sha256": settings["plan_sha256"], "directory": str(tmp_path / "job"),
        "data_root": str(tmp_path / "data"), "source_provenance": {"commit": "test"},
        "archive": str(tmp_path / "paused.zip")}
    paused = worker.execute(request)
    assert paused["state"] == "paused" and paused["next_seed_index"] == 0
    assert not events
    monkeypatch.setattr(worker, "train_chunk", training)
    completed = worker.execute({**request, "action": "resume", "directory": str(tmp_path / "continued"),
        "resume_archive": paused["archive"], "archive": str(tmp_path / "completed.zip")})
    assert completed["state"] == "complete" and completed["next_seed_index"] == 2
    assert len(events) == 2
    for seed in [31, 32]:
        path = tmp_path / "continued" / f"seed{seed}"
        assert json.loads((path / "status.json").read_text())["state"] == "complete"
        assert json.loads((path / "metrics.json").read_text())["rmse_standardized"] > 0
        assert (path / "test_predictions.npz").exists()
        assert json.loads((path / "model_metadata.json").read_text())["numerical"]["prior_jitter"] == 1e-3
        assert (tmp_path / f"continued_seed{seed}.zip").exists()
    with pytest.raises(ValueError, match="already complete"):
        worker.execute({**request, "action": "resume", "directory": str(tmp_path / "finished"),
            "resume_archive": completed["archive"], "archive": str(tmp_path / "forbidden.zip")})


def test_cpu_settings_in_fresh_verified_worker(github_checkout, tmp_path):
    setup = github_checkout(github_bootstrap("/unused", isolated=True))
    script = setup + '''
import os, runpy
client = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]
request = {"action": "settings", "source_provenance": SOURCE_PROVENANCE,
           "dataset": "oisst", "seeds": [31], "input_root": "unused",
           "overrides": {}, "data_overrides": {}}
result = client(ROOT, request, module="experiments.vnngp_cpu_worker", jobs_directory=ROOT.parent / "jobs")
assert result["worker_pid"] != os.getpid()
assert result["total_runs"] == 1
assert result["plan"]["configs"][0]["device"] == "cpu"
assert result["plan"]["configs"][0]["vnngp_neighbors"] == 32
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", "-"], input=script,
                            text=True, capture_output=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
