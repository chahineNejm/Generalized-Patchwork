import ast
import json
import subprocess
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import multidataset_worker as worker
from experiments.build_gpw_ablation_kaggle import build_notebook
from experiments.default_hyperparameters import (
    gpw_ablation_defaults,
    gpw_flight_ablation_defaults,
    multidataset_notebook_defaults,
)
from experiments.gpw_ablation import (
    bind_initialization,
    geometry_digest,
    model_geometry,
    summarize_results,
    verify_geometry,
)
from experiments.multidataset_data import DatasetSpec
from experiments.ukhousing_benchmark import UKConfig, build_model

ROOT = Path(__file__).resolve().parents[1]


def request(**updates):
    defaults = gpw_ablation_defaults(device="cpu", input_root="manual")
    return {**defaults, "study": "gpw_stitch_rank", "device": "cpu", "input_root": "manual",
            "save_inducing_positions": True, "model_overrides": {}, "data_overrides": {}, **updates}


def test_flight_plan_preserves_protocol_and_six_requested_conditions(tmp_path):
    defaults = gpw_flight_ablation_defaults(device="cpu", input_root="manual")
    result = worker.make_plan(request(**defaults))
    assert result["total_runs"] == 6 and result["total_epochs"] == 600
    assert len(result["plan"]["data"]) == 1
    spec = result["plan"]["data"][0]
    assert (spec["name"], spec["seed"], spec["train_fraction"], spec["validation_fraction"]) == ("flight", 31, .8, 0.)
    assert Path(spec["input_path"]).name == "airline.npz"
    configs = [r["config"] for r in result["plan"]["runs"]]
    assert {(c["q_rank"], c["stitch_mode"]) for c in configs} == {
        (r, m) for r in (0, 64) for m in ("none", "fixed", "learned")}
    assert {(c["seed"], c["inducing_points"], c["patches"], c["batch_size"]) for c in configs} == {(31, 1024, 8, 2048)}
    (tmp_path / "notebooks").mkdir()
    notebook = json.loads(build_notebook(tmp_path, flight=True).read_text())
    assert notebook == json.loads((ROOT / "notebooks/GPW_FLIGHT_Ablations_Kaggle.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), "flight-ablation", "exec")
            assert entry["outputs"] == []
    assert "DATASETS = ['flight']" in code and "SEEDS = [31]" in code and "RANKS = [0, 64]" in code
    assert "RUN_TRAINING = False" in code and "_OISST_STAGES" in code
    with pytest.raises(ValueError, match="not both"):
        build_notebook(tmp_path, flight=True, shared_kernel=True)


@pytest.mark.parametrize("mode", ["none", "fixed", "learned"])
def test_rank64_model_has_finite_gradients_and_matched_geometry(mode):
    torch.set_num_threads(2)
    split = tiny_split()
    config = UKConfig(device="cpu", seed=31, inducing_points=80, patches=4,
                      q_rank=64, stitch_mode=mode, stitches_per_edge=2)
    reference = bind_initialization(split, config)
    model = build_model(config, split)
    assert verify_geometry(model, split) == geometry_digest(reference)
    loss = -model.elbo_batch(split["x"][:8], split["y"][:8], 200)[0]
    assert torch.isfinite(loss)
    loss.backward()
    assert model.q.v.shape[1] == 64
    assert model.q.v.grad is not None and torch.isfinite(model.q.v.grad).all()


def test_approved_default_plan_has_72_runs_and_fixed_data_seed():
    result = worker.make_plan(request())
    assert [s["name"] for s in result["plan"]["data"]] == ["california", "protein", "tco", "oisst"]
    assert (len(result["plan"]["data"]), result["total_runs"], result["total_epochs"]) == (4, 72, 7200)
    assert {spec["seed"] for spec in result["plan"]["data"]} == {31}
    configs = [row["config"] for row in result["plan"]["runs"]]
    assert {c["model"] for c in configs} == {"gpw"}
    assert {c["seed"] for c in configs} == {31, 32, 33}
    assert {c["q_rank"] for c in configs} == {0, 32}
    assert {c["stitch_mode"] for c in configs} == {"none", "fixed", "learned"}
    assert {(c["inducing_points"], c["patches"]) for c in configs} == {(1024, 8)}
    assert all(c["epochs"] == 100 and c["warmup_epochs"] == 20 for c in configs)
    assert result["plan"]["progress_every_epochs"] == 20
    baseline = {r["dataset"]: r["config"] for r in result["plan"]["runs"] if r["config"]["seed"] == 31}
    for row in result["plan"]["runs"]:
        local = {k: v for k,v in row["config"].items() if k not in ("seed", "q_rank", "stitch_mode")}
        assert local == {k:v for k,v in baseline[row["dataset"]].items() if k not in ("seed", "q_rank", "stitch_mode")}


@pytest.mark.parametrize("seeds,total", [([31, 32, 33], 198), ([31, 32, 33, 34, 35], 330)])
def test_original_eleven_dataset_study_remains_explicitly_available(seeds, total):
    datasets = multidataset_notebook_defaults(device="cpu", input_root="manual")["datasets"]
    result = worker.make_plan(request(datasets=datasets, seeds=seeds))
    assert (len(result["plan"]["data"]), result["total_runs"], result["total_epochs"]) == (11, total, total*100)


@pytest.mark.parametrize("updates", [
    {"seeds": [31, 31]}, {"seeds": [True]}, {"ranks": [0, 1025]}, {"ranks": [0, 0]},
    {"stitch_modes": ["unknown"]}, {"inducing_points": 1025}, {"patches": 0},
    {"epochs": 20}, {"save_inducing_positions": "yes"},
    {"data_overrides": {"oisst": {"seed": 32}}},
    {"model_overrides": {"oisst": {"q_rank": 8}}},
    {"model_overrides": {"oisst": {"vnngp_neighbors": 32}}},
    {"study": "unknown"},
    {"progress_every_epochs": 0}, {"progress_every_epochs": True},
])
def test_invalid_ablation_plans_fail_without_adjustment(updates):
    with pytest.raises((ValueError, TypeError)):
        worker.make_plan(request(**updates))


def tiny_split():
    x = .01 + .98*torch.rand(240, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(31))
    return {"x": x, "y": torch.sin(8*x[:, 0]), "train": np.arange(200), "seed": 31}


@pytest.mark.parametrize("rank", [0, 32])
@pytest.mark.parametrize("mode", ["none", "fixed", "learned"])
def test_modes_preserve_matched_geometry_and_dense_moments(rank, mode):
    torch.set_num_threads(2)
    split = tiny_split()
    config = UKConfig(device="cpu", seed=32, inducing_points=40, patches=4,
                      q_rank=rank, stitch_mode=mode, stitches_per_edge=2)
    reference = bind_initialization(split, config)
    model = build_model(config, split)
    assert geometry_digest(model_geometry(model)) == geometry_digest(reference)
    assert model.stitch.trainable == (mode == "learned")
    assert (model.stitches().numel() == 0) == (mode == "none")
    prior = model.prior()
    x = split["x"][:5]
    terms = model.local_query_terms(x, prior)
    weights, alpha = model.regression_weights(x, prior, return_fusion=True)
    local = model.q.inverse_d_cholesky()
    precision = torch.block_diag(*(torch.cholesky_inverse(l) for l in local)) + model.q.v @ model.q.v.T
    covariance = torch.linalg.inv(precision)
    flat = weights.reshape(len(x), -1)
    expected = torch.einsum("bi,ij,bj->b", flat, covariance, flat)
    expected += torch.einsum("bi,bij,bj->b", alpha, terms["residual"], alpha)
    mean, variance = model.latent(x, prior)
    torch.testing.assert_close(variance, expected, atol=1e-9, rtol=1e-8)
    torch.testing.assert_close(mean, flat @ model.q.mean.flatten())
    objective = -model.elbo_batch(x, split["y"][:5], 200)[0]
    objective.backward()
    assert model.support.raw.grad is not None and torch.isfinite(model.support.raw.grad).all()
    assert ("stitch.raw" in dict(model.named_parameters())) == (mode == "learned")
    if mode == "learned":
        assert model.stitch.raw.grad is not None and torch.isfinite(model.stitch.raw.grad).all()


def test_geometry_is_identical_across_six_variants_and_changes_between_seeds():
    split = tiny_split()
    seen = []
    for seed in (31, 32):
        config = UKConfig(device="cpu", seed=seed, inducing_points=40, patches=4, stitches_per_edge=2)
        bind_initialization(split, config)
        hashes = {geometry_digest(model_geometry(build_model(replace(config, q_rank=r, stitch_mode=m), split)))
                  for r in (0, 32) for m in ("none", "fixed", "learned")}
        assert len(hashes) == 1
        seen.extend(hashes)
    assert seen[0] != seen[1]


def test_mismatched_reference_geometry_raises_before_training():
    split = tiny_split()
    config = UKConfig(device="cpu", inducing_points=40, patches=4, stitches_per_edge=2)
    bind_initialization(split, config)
    model = build_model(config, split)
    with torch.no_grad():
        model.candidate_stitch_locations.add_(.001)
    with pytest.raises(ValueError, match="differ from matched reference"):
        verify_geometry(model, split)


def test_notebook_is_reproducible_configuration_only_and_stage_guarded(tmp_path):
    (tmp_path / "notebooks").mkdir()
    notebook = json.loads(build_notebook(tmp_path).read_text())
    assert notebook == json.loads((ROOT / "notebooks/GPW_Ablations_Kaggle.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), "ablation", "exec")
            assert entry["outputs"] == []
    assert "RUN_TRAINING = False" in code and "training_view.update(Image" in code
    assert "SEEDS = [31, 32, 33]" in code and "RANKS = [0, 32]" in code
    assert "INDUCING_POINTS = 1024" in code and "PATCHES = 8" in code
    assert "PROGRESS_EVERY_EPOCHS = 20" in code
    assert "_OISST_STAGES" in code and "class " not in code
    imports = [n.module for n in ast.walk(ast.parse(code)) if isinstance(n, ast.ImportFrom)]
    assert not any(n and n.startswith(("experiments", "pwgp")) for n in imports)


def test_short_ablation_worker_saves_six_runs_and_stops_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "source_snapshot", lambda value: (tmp_path, b"source"))
    events = []
    monkeypatch.setattr(worker, "emit", lambda kind, **values: events.append((kind, values)))
    plan = worker.make_plan(request(datasets=["california"], seeds=[32], inducing_points=40, patches=4,
        epochs=2, warmup_epochs=1, early_stopping=None,
        model_overrides={"california": {"stitches_per_edge": 2, "batch_size": 64,
            "prediction_batch_size": 32, "snapshot_every": 2}}))["plan"]
    spec = DatasetSpec("tiny", "g_function", 3, .8, 0., 31, synthetic={
        "training_points": 240, "test_points": 60, "noise_variance": 1e-8,
        "michalewicz_m": 10, "g_coefficients": [0., .5, 1.]})
    plan["data"] = [asdict(spec)]
    for row in plan["runs"]:
        row["dataset"] = "tiny"
    common = {"plan": plan, "plan_sha256": worker.plan_hash(plan), "source_provenance": {}}
    prepared = tmp_path / "prepared"
    worker.execute({**common, "action": "prepare", "directory": str(prepared), "data_root": str(tmp_path)})
    batch = tmp_path / "batch"
    worker.execute({**common, "action": "train", "directory": str(batch),
        "prepared_directory": str(prepared), "execute_training": True})
    identities = []
    for i,row in enumerate(plan["runs"]):
        folder = batch / "tiny" / f"run_{i:02d}"
        identity = json.loads((folder / "initialization_identity.json").read_text())
        identities.append(identity["geometry_sha256"])
        control = json.loads((folder / "training_control.json").read_text())
        assert control["progress_description"] == f"train:tiny / {row['name']}"
        assert control["progress_every_epochs"] == 20
        assert identity["data_seed"] == 31 and identity["model_training_seed"] == 32
        mode = row["config"]["stitch_mode"]
        with np.load(folder / "stitches_epoch_0.npz", allow_pickle=False) as a, np.load(
                folder / "stitches_epoch_2.npz", allow_pickle=False) as b:
            if mode == "fixed":
                np.testing.assert_array_equal(a["stitches"], b["stitches"])
            elif mode == "learned":
                assert np.max(np.abs(a["stitches"]-b["stitches"])) > 0
            else:
                assert a["stitches"].size == b["stitches"].size == 0
        with zipfile.ZipFile(batch / f"tiny_run_{i:02d}.zip") as z:
            assert f"run_{i:02d}/initialization_identity.json" in z.namelist()
    assert len(set(identities)) == 1
    assert len([e for e in events if e[0] == "complete"]) == 6
    summary = json.loads((batch / "ablation_summary.json").read_text())
    assert len(summary) == 6 and all(r["completed_seeds"] == 1 for r in summary)
    assert all(r["metrics"]["rmse_standardized"]["sample_sd"] is None for r in summary)
    assert (batch / "visuals/tiny.png").is_file()
    original = worker.run_model
    calls = []

    def complete_then_fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("deliberate ablation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(worker, "run_model", complete_then_fail)
    with pytest.raises(RuntimeError, match="deliberate ablation"):
        worker.execute({**common, "action": "train", "directory": str(tmp_path / "failed"),
            "prepared_directory": str(prepared), "execute_training": True})
    assert len(calls) == 2 and (tmp_path / "failed/tiny_run_00.zip").is_file()
    assert "deliberate ablation" in (tmp_path / "failed/failure.json").read_text()


def test_seed_summary_uses_sample_sd_and_rejects_duplicates():
    rows = [{"dataset": "x", "q_rank": 0, "stitch_mode": "none", "seed": seed,
             "rmse_standardized": value} for seed,value in ((31,1.), (32,3.))]
    metric = summarize_results(rows)[0]["metrics"]["rmse_standardized"]
    assert metric["mean"] == 2 and metric["sample_sd"] == np.sqrt(2)
    with pytest.raises(ValueError, match="repeated seeds"):
        summarize_results(rows + [rows[0]])


def test_ablation_settings_use_fresh_workers_without_reconnect(github_checkout, tmp_path):
    notebook = json.loads((ROOT / "notebooks/GPW_Ablations_Kaggle.ipynb").read_text())
    setup = github_checkout("".join(notebook["cells"][2]["source"]))
    script = setup + '\nimport os, runpy\nclient=runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]\n'
    script += 'request=' + repr({**request(), "action": "settings"}) + '\n'
    script += '''request["source_provenance"]=SOURCE_PROVENANCE
first=client(ROOT, request, module="experiments.multidataset_worker", jobs_directory=ROOT.parent / "logs")
second=client(ROOT, request, module="experiments.multidataset_worker", jobs_directory=ROOT.parent / "logs")
assert first["worker_pid"] != os.getpid() and second["worker_pid"] != first["worker_pid"]
assert first["total_runs"] == 72 and first["plan"] == second["plan"]
'''
    result = subprocess.run([sys.executable, "-X", "utf8", "-I", "-"], input=script,
                            text=True, capture_output=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
