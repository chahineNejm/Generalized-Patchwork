import ast
import json
import subprocess
import sys
import weakref
import zipfile
from dataclasses import replace
from pathlib import Path

import gpytorch
import numpy as np
import pytest
import torch

from experiments import ukhousing_benchmark
from experiments.ukhousing_benchmark import UKConfig, archive_run, prepare_split, run_ukhousing
from pwgp.config import ProSparConfig
from pwgp.data.ukhousing import acquire
from pwgp.models.standard_patchwork import StandardPatchworkGP
from pwgp.models.vnngp import VNNGP, strict_gpytorch

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("maximum_points", [7, 32])
def test_direct_patchwork_matches_dense_joint_likelihood_and_conditioning(maximum_points):
    torch.manual_seed(8)
    x = 0.05 + 0.9 * torch.rand(20, 2, dtype=torch.float64)
    y = torch.sin(8 * x[:, 0])
    model = StandardPatchworkGP(x, y, maximum_points=maximum_points, stitches_per_edge=2)
    state = model.conditioned_state()
    n, q = len(x), len(model.stitch_points)
    ky = (model.hyp.noise_std().square() + model.jitter) * torch.eye(n, dtype=x.dtype)
    g = model.jitter * torch.eye(q, dtype=x.dtype)
    b = x.new_zeros(n, q)
    for patch in range(model.patches):
        mask = (model.labels == patch).double()
        ky = ky + model.kernel(x, x, patch) * mask[:, None] * mask[None]
        signs = model.coefficients[patch]
        g = g + model.kernel(model.stitch_points, model.stitch_points, patch) * signs[:, None] * signs[None]
        b = b + model.kernel(x, model.stitch_points, patch) * mask[:, None] * signs[None]
    query = 0.05 + 0.9 * torch.rand(6, 2, dtype=x.dtype)
    joint = torch.cat((torch.cat((ky, b), 1), torch.cat((b.T, g), 1)), 0)
    target = torch.cat((y, x.new_zeros(q)))
    expected_nll = -torch.distributions.MultivariateNormal(torch.zeros_like(target), joint).log_prob(target)
    torch.testing.assert_close(state["nll"], expected_nll)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(state["nll"], parameters, retain_graph=True)
    expected_gradients = torch.autograd.grad(expected_nll, parameters, retain_graph=True)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected)
    if q:
        conditioned = ky - b @ torch.linalg.solve(g, b.T)
        conditional_nll = -torch.distributions.MultivariateNormal(torch.zeros_like(y), conditioned).log_prob(y)
        stitch_nll = -torch.distributions.MultivariateNormal(x.new_zeros(q), g).log_prob(x.new_zeros(q))
        torch.testing.assert_close(state["nll"], conditional_nll + stitch_nll)
    expected_mean, expected_variance = [], []
    for point in query:
        inside = ((point >= model.region_bounds[:, 0]) & (point < model.region_bounds[:, 1])).all(-1)
        patch = int(inside.nonzero().item())
        cy = model.kernel(point[None], x, patch).flatten() * (model.labels == patch)
        cd = model.kernel(point[None], model.stitch_points, patch).flatten() * model.coefficients[patch]
        cross = torch.cat((cy, cd))
        expected_mean.append(cross @ torch.linalg.solve(joint, target))
        expected_variance.append(model.hyp.outputscales()[patch] - cross @ torch.linalg.solve(joint, cross))
    mean, variance = model.predict(query, observation_noise=False, state=state)
    torch.testing.assert_close(mean, torch.stack(expected_mean))
    torch.testing.assert_close(variance, torch.stack(expected_variance))
    model.loss().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_vnngp_search_and_predict_match_official_strategy():
    torch.manual_seed(9)
    x = torch.rand(25, 2, dtype=torch.float64)
    model = VNNGP(x, neighbors=4, batch_size=8)
    graph = model.variational_strategy.nn_xinduce_idx
    for i in range(4, len(x)):
        nearest = torch.cdist(x[i:i + 1].float(), x[:i].float()).topk(4, largest=False).indices.flatten()
        torch.testing.assert_close(graph[i - 4], nearest)
    model.train()
    model(None)
    with torch.no_grad():
        model.variational_strategy._variational_distribution.variational_mean.normal_(std=0.2)
    model.eval()
    query = torch.rand(6, 2, dtype=x.dtype)
    with torch.no_grad(), strict_gpytorch():
        official = model(query)
        mean, variance = model.predict(query, observation_noise=False)
    torch.testing.assert_close(mean, official.mean)
    torch.testing.assert_close(variance, official.lazy_covariance_matrix.diagonal(dim1=-2, dim2=-1))


@pytest.mark.parametrize("neighbors,batch_size", [(4, 8), (32, 512)])
def test_vnngp_stochastic_kl_values_and_gradients_match_upstream(neighbors, batch_size):
    torch.manual_seed(31)
    torch.set_num_threads(2)
    x = torch.rand(neighbors + batch_size, 2, dtype=torch.float64)
    model = VNNGP(x, neighbors=neighbors, batch_size=batch_size)
    strategy = model.variational_strategy
    q = strategy._variational_distribution
    with torch.no_grad():
        q.variational_mean.normal_(std=0.2)
        q._variational_stddev.uniform_(0.1, 0.5)
    indices = torch.arange(neighbors, len(x))
    expected = gpytorch.variational.NNVariationalStrategy._stochastic_kl_helper(strategy, indices)
    actual = strategy._stochastic_kl_helper(indices)
    parameters = [q.variational_mean, q._variational_stddev,
                  model.covar_module.raw_outputscale,
                  model.covar_module.base_kernel.raw_lengthscale]
    expected_gradients = torch.autograd.grad(expected, parameters)
    actual_gradients = torch.autograd.grad(actual, parameters)
    torch.testing.assert_close(actual, expected, rtol=1e-7, atol=1e-7)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-6, atol=1e-6)


def test_vnngp_training_does_not_call_general_solve(monkeypatch):
    torch.manual_seed(31)
    x = torch.rand(20, 2, dtype=torch.float64)
    model = VNNGP(x, neighbors=4, batch_size=8)

    def forbidden(*args, **kwargs):
        raise AssertionError("VNNGP training called torch.linalg.solve")

    monkeypatch.setattr(torch.linalg, "solve", forbidden)
    objective = gpytorch.mlls.VariationalELBO(model.likelihood, model, num_data=len(x))
    for _ in range(2):
        model.train()
        model.zero_grad()
        with strict_gpytorch():
            output = model(None)
            indices = model.variational_strategy.current_training_indices
            loss = -objective(output, torch.sin(x[indices, 0]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("name", ["svgp", "gpw", "patchwork", "vnngp", "prospar"])
def test_model_runner_end_to_end(name, tmp_path, monkeypatch):
    references = []
    build = ukhousing_benchmark.build_model

    def tracked_model(config, split):
        model = build(config, split)
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(ukhousing_benchmark, "build_model", tracked_model)
    rng = np.random.default_rng(31)
    normalized = rng.uniform(0.05, 0.95, (100, 2))
    x = normalized * [7.5, 5.5] + [-5.7, 50]
    y = 12 + np.sin(8 * normalized[:, 0]) + 0.1 * rng.normal(size=100)
    split = prepare_split(x, y)
    config = replace(UKConfig(), model=name, device="cpu", epochs=2, warmup_epochs=1,
                     inducing_points=12, patches=3, q_rank=4, stitches_per_edge=2,
                     vnngp_neighbors=4, vnngp_batch_size=16, exact_maximum_points=16,
                     batch_size=32, prediction_batch_size=8, snapshot_every=2)
    if name == "patchwork":
        config = replace(config, warmup_epochs=20)
    if name == "prospar":
        config = replace(config, prospar=ProSparConfig(
            temperatures=(1.0, 2.0, 3.0), exponent=1.0, inducing_nuggets=0.02, prior_jitter=0.0))
    result = run_ukhousing(config, split, directory=tmp_path / name)
    assert references[0]() is None
    assert (result / "model.pt").stat().st_size > 0
    assert (result / "checkpoint.pt").stat().st_size > 0
    metrics = json.loads((result / "metrics.json").read_text())
    assert np.isfinite(metrics["nlpd"])
    assert metrics["test_points"] == len(split["test"])
    assert json.loads((result / "status.json").read_text())["state"] == "complete"
    assert (result / "progress.png").stat().st_size > 0
    assert (result / "maps.png").stat().st_size > 0
    metadata = json.loads((result / "model_metadata.json").read_text())
    assert metadata["package_version"]
    if name == "prospar":
        assert metadata["model_class"].endswith("ProSparGP")
        learned = json.loads((result / "learned_parameters.json").read_text())
        assert len(learned["inducing_nuggets"]) == 3
        assert learned["prior_jitter"] == 0.0
    with np.load(result / "test_predictions.npz") as predictions:
        np.testing.assert_equal(predictions["indices"], split["test"])


def test_archive_run_includes_artifacts_and_shared_provenance(tmp_path):
    root = tmp_path / "comparison"
    run = root / "vnngp"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"state": "complete"}')
    (run / "model.pt").write_bytes(b"checkpoint")
    shared = ["source_snapshot.zip", "split.npz", "preprocessing.json",
              "data_manifest.json", "comparison.csv", "source_provenance.json"]
    for name in shared:
        (root / name).write_bytes(name.encode())
    archive = archive_run(run)
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {*shared, "vnngp/status.json", "vnngp/model.pt"}
        for name in bundle.namelist():
            assert bundle.read(name) == (root / name).read_bytes()
    with pytest.raises(FileExistsError):
        archive_run(run)
    (run / "status.json").write_text('{"state": "running"}')
    with pytest.raises(ValueError, match="Only completed"):
        archive_run(run)


def test_ukhousing_notebook_calls_github_and_keeps_four_models():
    notebook = json.loads((ROOT / "notebooks/UKHousing_Colab.ipynb").read_text())
    assert "SOURCE_B64" not in "".join(notebook["cells"][2]["source"])
    assert notebook["metadata"]["gpw_source"]["branch"] == "main"
    for cell in notebook["cells"]:
        assert not cell.get("outputs")
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if not source.startswith("%pip"):
                compile(source, "ukhousing_colab", "exec")
    setup = "".join(notebook["cells"][5]["source"])
    setup_tree = ast.parse(setup)
    probes = next(node for node in setup_tree.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "GPU_KL_PROBES")
    assert ast.literal_eval(probes.value) == [(4, 8), (32, 512)]
    assert "for step in range(2):" in setup
    run_cell = next("".join(cell["source"]) for cell in notebook["cells"]
                    if "summaries, completed = [], []" in "".join(cell["source"]))
    assert run_cell.index("archive_run(result)") < run_cell.index("files.download(str(run_archive))")
    assert run_cell.index("torch.cuda.empty_cache()") < run_cell.index("files.download(str(run_archive))")
    for name in ["SVGP 1024", "GPW 8 x 128", "Standard Patchwork", "VNNGP K32"]:
        assert name in setup


def test_target_standardization_uses_training_only():
    rng = np.random.default_rng(31)
    x = rng.uniform([-5, 50.1], [1.5, 55], size=(100, 2))
    y = rng.normal(size=100)
    initial = prepare_split(x, y)
    y[initial["test"]] += 1000
    changed = prepare_split(x, y)
    assert initial["target_mean"] == changed["target_mean"]
    assert initial["target_std"] == changed["target_std"]


def test_ukhousing_github_source_imports_in_fresh_interpreter(tmp_path, github_checkout):
    notebook = json.loads((ROOT / "notebooks/UKHousing_Colab.ipynb").read_text())
    script = github_checkout("".join(notebook["cells"][2]["source"])) + """
from experiments.ukhousing_benchmark import UKConfig, build_model, prepare_split
from pwgp.models.standard_patchwork import StandardPatchworkGP
from pwgp.models.vnngp import VNNGP
import pwgp
assert Path(pwgp.__file__).resolve().is_relative_to(ROOT / 'src')
assert UKConfig().epochs == 100
assert UKConfig().warmup_epochs == 20
"""
    result = subprocess.run([sys.executable, "-I", "-"], input=script, text=True,
                            capture_output=True, cwd=tmp_path, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_ukhousing_bad_cached_checksum_raises_without_network(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached source checksum mismatch"):
        acquire("https://example.invalid/no-network", path, "0" * 64)
