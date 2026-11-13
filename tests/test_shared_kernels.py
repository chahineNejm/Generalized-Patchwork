import ast
import json
import subprocess
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments import multidataset_worker as worker
from experiments.build_gpw_ablation_kaggle import build_notebook
from experiments.default_hyperparameters import gpw_shared_kernel_defaults
from experiments.gpw_ablation import bind_initialization, geometry_digest, model_geometry
from experiments.multidataset_data import DatasetSpec, load_prepared
from experiments.ukhousing_benchmark import UKConfig, build_model
from pwgp import GeneralizedPatchworkGP
from pwgp.parameters import LocalARDHyperparameters

ROOT = Path(__file__).resolve().parents[1]


def request(**values):
    return {**gpw_shared_kernel_defaults(device="cpu", input_root="manual"),
        "study": "gpw_kernel_sharing", "device": "cpu", "input_root": "manual",
        "model_overrides": {}, "data_overrides": {}, "save_inducing_positions": True, **values}


def test_tied_kernel_values_objective_and_gradients_leave_q_free():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    supports = .1 + .8*torch.rand(3, 4, 2, dtype=torch.float64)
    stitches = .1 + .8*torch.rand(2, 2, 2, dtype=torch.float64)
    options = {"q_rank": 3, "lengthscale": torch.tensor([.2,.35], dtype=torch.float64)}
    local = GeneralizedPatchworkGP(supports, stitches, [(0,1),(1,2)], **options)
    shared = GeneralizedPatchworkGP(supports, stitches, [(0,1),(1,2)], shared_kernel=True, **options)
    with torch.no_grad():
        local.q.mean.normal_(std=.1)
        local.q.raw_inverse_d_cholesky.mul_(.9)
    shared.q.load_state_dict(local.q.state_dict())
    assert shared.hyp.raw_lengthscales.shape == (1, 2)
    assert shared.hyp.raw_outputscales.shape == (1,)
    assert sum(p.numel() for p in shared.hyp.parameters()) == 4
    assert sum(p.numel() for p in shared.q.parameters()) == sum(p.numel() for p in local.q.parameters())
    assert all(parameter.requires_grad for parameter in shared.q.parameters())
    x = .1 + .8*torch.rand(9, 2, dtype=torch.float64)
    y = torch.sin(7*x[:,0])
    for first,second in zip(shared.predict(x), local.predict(x), strict=True):
        torch.testing.assert_close(first, second)
    shared_loss = -shared.elbo_batch(x,y,len(x))[0]
    local_loss = -local.elbo_batch(x,y,len(x))[0]
    torch.testing.assert_close(shared_loss,local_loss)
    shared_loss.backward()
    local_loss.backward()
    torch.testing.assert_close(shared.hyp.raw_lengthscales.grad,
        local.hyp.raw_lengthscales.grad.sum(0,keepdim=True),rtol=1e-7,atol=1e-8)
    torch.testing.assert_close(shared.hyp.raw_outputscales.grad,
        local.hyp.raw_outputscales.grad.sum().reshape(1),rtol=1e-7,atol=1e-8)
    for name,parameter in shared.q.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad,dict(local.q.named_parameters())[name].grad)
    before = {name:p.detach().clone() for name,p in shared.q.named_parameters()}
    optimizer = torch.optim.Adam(shared.parameters(),lr=.001)
    optimizer.step()
    for name,parameter in shared.q.named_parameters():
        assert not torch.equal(before[name],parameter), name
    ls, os = shared.hyp.lengthscales(),shared.hyp.outputscales()
    assert torch.equal(ls,ls[:1].expand_as(ls)) and torch.equal(os,os[:1].expand_as(os))


@pytest.mark.parametrize("change", [
    {"lengthscale": torch.tensor([[.1,.2],[.3,.2]],dtype=torch.float64)},
    {"outputscale": torch.tensor([1.,2.],dtype=torch.float64)}, {"shared": "yes"},
])
def test_shared_initialization_does_not_average_or_project(change):
    with pytest.raises((ValueError,TypeError),match="identical|boolean"):
        LocalARDHyperparameters(2,2,**{"shared":True,**change})


def test_shared_plan_is_twelve_runs_matching_local_reference():
    result = worker.make_plan(request())
    local = worker.make_plan({**request(),"study":"gpw_stitch_rank","ranks":[32],"stitch_modes":["learned"]})
    assert result["total_runs"] == 12 and result["total_epochs"] == 1200
    assert result["plan"]["data"] == local["plan"]["data"]
    assert {s["seed"] for s in result["plan"]["data"]} == {31}
    assert {r["config"]["seed"] for r in result["plan"]["runs"]} == {31,32,33}
    for row,reference in zip(result["plan"]["runs"],local["plan"]["runs"],strict=True):
        config = row["config"]
        assert config["kernel_mode"] == "shared"
        assert {**config,"kernel_mode":"local"} == reference["config"]
        assert (config["inducing_points"],config["patches"],config["q_rank"],config["stitch_mode"]) == (1024,8,32,"learned")


@pytest.mark.parametrize("change", [
    {"seeds":[]}, {"seeds":[31,31]}, {"q_rank":1025}, {"progress_every_epochs":0},
    {"data_overrides":{"oisst":{"seed":32}}},
    {"model_overrides":{"oisst":{"kernel_mode":"local"}}},
    {"ranks":[0,32]}, {"stitch_modes":["none"]},
])
def test_shared_plans_reject_conflicting_controls(change):
    with pytest.raises((ValueError,TypeError)):
        worker.make_plan(request(**change))


def test_shared_notebook_reproducible_and_configuration_only(tmp_path):
    (tmp_path/"notebooks").mkdir()
    notebook = json.loads(build_notebook(tmp_path,shared_kernel=True).read_text())
    assert notebook == json.loads((ROOT/"notebooks/GPW_Shared_Kernels_Kaggle.ipynb").read_text())
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code")
    for entry in notebook["cells"]:
        if entry["cell_type"]=="code":
            compile("".join(entry["source"]),"shared-kernel","exec")
            assert entry["outputs"] == []
    assert "SEEDS = [31, 32, 33]" in code and "Q_RANK = 32" in code
    assert 'study="gpw_kernel_sharing"' in code and "RUN_TRAINING = False" in code
    assert "PROGRESS_EVERY_EPOCHS = 20" in code and "training_view.update(Image" in code
    imports = [node.module for node in ast.walk(ast.parse(code)) if isinstance(node,ast.ImportFrom)]
    assert not any(name and name.startswith(("experiments","pwgp")) for name in imports)


def test_short_shared_study_keeps_geometry_q_and_archives(tmp_path,monkeypatch):
    monkeypatch.setattr(worker,"source_snapshot",lambda value:(tmp_path,b"source"))
    events = []
    monkeypatch.setattr(worker,"emit",lambda kind,**values:events.append((kind,values)))
    plan = worker.make_plan(request(datasets=["california"],inducing_points=24,patches=4,q_rank=4,
        epochs=2,warmup_epochs=1,early_stopping=None,model_overrides={"california":{
            "stitches_per_edge":2,"batch_size":64,"prediction_batch_size":32,"snapshot_every":2}}))["plan"]
    spec = DatasetSpec("tiny","g_function",3,.8,0.,31,synthetic={"training_points":160,"test_points":40,
        "noise_variance":1e-8,"michalewicz_m":10,"g_coefficients":[0.,.5,1.]})
    plan["data"] = [asdict(spec)]
    for row in plan["runs"]:
        row["dataset"] = "tiny"
    common = {"plan":plan,"plan_sha256":worker.plan_hash(plan),"source_provenance":{}}
    prepared,batch = tmp_path/"prepared",tmp_path/"batch"
    worker.execute({**common,"action":"prepare","directory":str(prepared),"data_root":str(tmp_path)})
    split_bytes = (prepared/"tiny/split.npz").read_bytes()
    worker.execute({**common,"action":"train","directory":str(batch),"prepared_directory":str(prepared),"execute_training":True})
    assert split_bytes == (batch/"tiny/split.npz").read_bytes()
    fingerprints = []
    for index,row in enumerate(plan["runs"]):
        folder = batch/"tiny"/f"run_{index:02d}"
        record = json.loads((folder/"kernel_comparison_reference.json").read_text())
        metadata = json.loads((folder/"model_metadata.json").read_text())
        fingerprints.append(record["geometry_sha256"])
        assert record["model_training_seed"] == 31+index and record["data_seed"] == 31
        assert metadata["kernel_mode"] == "shared" and metadata["kernel_parameter_count"] == 4
        split = load_prepared(spec,prepared/"tiny")
        local = UKConfig(**record["local_reference_config"])
        bind_initialization(split,local)
        model = build_model(local,split)
        assert geometry_digest(model_geometry(model)) == record["geometry_sha256"]
        initial = torch.load(folder/"initial_model.pt",weights_only=True)
        final = torch.load(folder/"model.pt",weights_only=True)
        for name in ("q.mean","q.raw_inverse_d_cholesky","q.v"):
            assert initial[name].shape == final[name].shape
            assert not torch.equal(initial[name],final[name]), name
        control = json.loads((folder/"training_control.json").read_text())
        assert control["progress_every_epochs"] == 20 and "kernels-shared" in control["progress_description"]
        with zipfile.ZipFile(batch/f"tiny_run_{index:02d}.zip") as archive:
            assert f"run_{index:02d}/kernel_comparison_reference.json" in archive.namelist()
    assert len(set(fingerprints)) == 3
    summary = json.loads((batch/"kernel_summary.json").read_text())
    assert len(summary) == 1 and summary[0]["completed_seeds"] == 3
    rows = pd.read_csv(batch/"comparison.csv")
    assert set(rows["kernel_mode"]) == {"shared"} and set(rows["seed"]) == {31,32,33}
    assert len([event for event in events if event[0]=="complete"]) == 3
    assert not (batch/"ablation_summary.json").exists()
    original = worker.run_model
    calls = []

    def fail_second(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:
            raise RuntimeError("deliberate shared-kernel failure")
        return original(*args,**kwargs)

    monkeypatch.setattr(worker,"run_model",fail_second)
    with pytest.raises(RuntimeError,match="deliberate shared-kernel failure"):
        worker.execute({**common,"action":"train","directory":str(tmp_path/"failed"),
            "prepared_directory":str(prepared),"execute_training":True})
    assert len(calls)==2 and (tmp_path/"failed/tiny_run_00.zip").is_file()
    assert "deliberate shared-kernel failure" in (tmp_path/"failed/failure.json").read_text()


def test_kernel_modes_are_explicit_and_gpw_only():
    split = {"x":torch.full((4,2),.5,dtype=torch.float64),"y":torch.ones(4),"train":np.arange(4)}
    for config in (replace(UKConfig(),kernel_mode="unknown"),replace(UKConfig(),model="svgp",kernel_mode="shared")):
        with pytest.raises(ValueError,match="kernel_mode"):
            build_model(config,split)


def test_shared_settings_in_fresh_source_workers(github_checkout,tmp_path):
    notebook = json.loads((ROOT/"notebooks/GPW_Shared_Kernels_Kaggle.ipynb").read_text())
    setup = github_checkout("".join(notebook["cells"][2]["source"]))
    script = setup+'\nimport os,runpy\nclient=runpy.run_path(str(ROOT/"experiments/colab_client.py"))["run_worker"]\n'
    script += 'request='+repr({**request(),"action":"settings"})+'\n'
    script += '''request["source_provenance"]=SOURCE_PROVENANCE
first=client(ROOT,request,module="experiments.multidataset_worker",jobs_directory=ROOT.parent/"jobs")
second=client(ROOT,request,module="experiments.multidataset_worker",jobs_directory=ROOT.parent/"jobs")
assert first["total_runs"]==12 and first["plan"]==second["plan"]
assert first["worker_pid"]!=os.getpid() and first["worker_pid"]!=second["worker_pid"]
'''
    result = subprocess.run([sys.executable,"-X","utf8","-I","-"],input=script,
        text=True,capture_output=True,cwd=tmp_path,check=False)
    assert result.returncode == 0,result.stdout+result.stderr
