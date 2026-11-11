import io
import json
import weakref
import zipfile
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch

from experiments import oisst_benchmark as benchmark
from experiments.default_hyperparameters import oisst_defaults
from pwgp.data.types import RegressionDataset


def example_split(config):
    rng = np.random.default_rng(12)
    x = rng.uniform([280.1, 20.1], [329.9, 49.9], size=(100, 2))
    y = np.sin(x[:, 0]) + np.cos(x[:, 1]) + 12
    return benchmark.prepare_split(RegressionDataset("fixture", x, y, ("lon", "lat"), "C"), config)


def test_approved_defaults_and_explicit_overrides():
    data, runs = oisst_defaults(device="cpu")
    assert data.bounds == (-80, -30, 20, 50) and data.seed == 31
    assert [config.model for _, config in runs] == ["svgp", "gpw", "prospar", "vnngp", "patchwork", "patchwork"]
    for _, config in runs:
        assert (config.epochs, config.warmup_epochs, config.inducing_points) == (100, 20, 512)
    prospar = runs[2][1].prospar
    assert asdict(prospar) == {"temperatures": 1, "exponent": 1,
                              "inducing_nuggets": 0.01, "prior_jitter": 0}
    changed = replace(runs[1][1], epochs=150)
    assert changed.epochs == 150 and oisst_defaults(device="cpu")[1][1][1].epochs == 100


def test_split_uses_training_targets_only_and_never_moves_points():
    data, _ = oisst_defaults(device="cpu")
    split = example_split(data)
    torch.testing.assert_close(split["y"][split["train"]].mean(), torch.tensor(0., dtype=torch.float64),
                               atol=1e-12, rtol=0)
    assert [len(split[key]) for key in ("train", "validation", "test")] == [64, 16, 20]
    np.testing.assert_allclose(split["x"].numpy() * [50, 30] + [280, 20], split["raw_x"])


def test_five_model_comparison_artifacts_and_memory_release(tmp_path, monkeypatch):
    data, runs = oisst_defaults(device="cpu")
    split = example_split(data)
    monkeypatch.setattr(benchmark, "acquire", lambda *a, **k: (split, {"source": "synthetic fixture"}))
    references = []
    original = benchmark.build_model

    def tracked(config, split):
        model = original(config, split)
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(benchmark, "build_model", tracked)
    runs = [(name, replace(config, epochs=2,
                          warmup_epochs=config.warmup_epochs if config.model == "patchwork" else 1,
                          inducing_points=8, patches=2,
                          q_rank=2, patch_neighbors=1, stitches_per_edge=2, vnngp_neighbors=4,
                          vnngp_batch_size=16, exact_maximum_points=16, exact_stitches_per_edge=2,
                          exact_grid_shape=(2, 1),
                          batch_size=32, snapshot_every=2, prediction_batch_size=8)) for name, config in runs]
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as source:
        source.writestr("fixture.txt", "synthetic source")
    completed = []
    result = benchmark.run_comparison(data, runs, directory=tmp_path / "run", data_root=tmp_path,
        source_provenance={"mode": "fixture", "commit": "test"}, source_zip=stream.getvalue(),
        on_complete=lambda path, archive, metrics: completed.append((path, archive, metrics)))
    assert len(completed) == 6 and all(ref() is None for ref in references)
    assert json.loads((result / "status.json").read_text())["state"] == "complete"
    assert json.loads((result / "exploratory/summary.json").read_text())["observations"] == 100
    for (name, config), (path, archive, metrics) in zip(runs, completed, strict=True):
        assert metrics["model"] == name and np.isfinite(metrics["nlpd"])
        environment = json.loads((path / "environment.json").read_text())
        assert environment["fusion_covariance"] == (
            "Cov(local_functions | U, Delta_S=0)" if config.model == "gpw" else None)
        for filename in ("training.png", "metrics.png", "progress.png", "maps.png", "hyperparameters.png"):
            assert (path / "visuals" / filename).stat().st_size > 0
        if config.model != "patchwork":
            with np.load(path / "inducing_positions.npz") as positions:
                assert positions["epochs"].tolist() == [0, 2]
            assert (path / "visuals/inducing_positions.png").exists()
        else:
            assert not (path / "inducing_positions.npz").exists()
            environment = json.loads((path / "environment.json").read_text())
            assert environment["optimizer"]["warmup"] == "not_used"
            assert json.loads((path / "config.json").read_text())["warmup_epochs"] == 20
            metadata = json.loads((path / "model_metadata.json").read_text())
            assert metadata["training_observations_used"] == 8
            assert (path / "visuals/partition.png").is_file()
            with np.load(path / "training_points.npz") as selected, np.load(
                    completed[1][0] / "initialization.npz") as initial:
                np.testing.assert_array_equal(selected["indices"], initial["indices"].ravel())
                np.testing.assert_array_equal(selected["x"], initial["supports"].reshape(-1, 2))
            with np.load(result / "gpw_initialization.npz") as shared, np.load(
                    completed[1][0] / "initialization.npz") as initial:
                np.testing.assert_array_equal(shared["indices"], initial["indices"])
        with zipfile.ZipFile(archive) as bundle:
            assert "source_provenance.json" in bundle.namelist()
            assert "source_snapshot.zip" in bundle.namelist()
            assert "split.npz" in bundle.namelist()


def test_optional_positions_and_failure_stops_subsequent_runs(tmp_path, monkeypatch):
    data, runs = oisst_defaults(device="cpu")
    split = example_split(data)
    config = replace(runs[0][1], epochs=1, warmup_epochs=0, inducing_points=4,
                     batch_size=64, prediction_batch_size=16, snapshot_every=1)
    result = benchmark.run_model(config, split, directory=tmp_path / "no_positions",
                                 save_inducing_positions=False)
    assert not (result / "inducing_positions.npz").exists()
    reached = []
    monkeypatch.setattr(benchmark, "acquire", lambda *a, **k: (split, {"source": "fixture"}))

    def fail(config, *args, **kwargs):
        reached.append(config.model)
        raise RuntimeError("original training failure")

    monkeypatch.setattr(benchmark, "run_model", fail)
    with pytest.raises(RuntimeError, match="original training failure"):
        benchmark.run_comparison(data, runs[:2], directory=tmp_path / "failed", data_root=tmp_path,
                                 source_provenance={"mode": "fixture"}, source_zip=b"fixture")
    assert reached == ["svgp"]
    assert "original training failure" in json.loads((tmp_path / "failed/failure.json").read_text())["traceback"]


@pytest.mark.parametrize("requested", ["cuda:0", "cuda:2"])
def test_cold_cuda_initializes_requested_device_before_memory_reset(tmp_path, monkeypatch, requested):
    _, runs = oisst_defaults(device=requested)
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def synchronize(device):
        assert device == torch.device(requested)
        calls.append("initialized")

    def reset(device):
        assert calls == ["initialized"]
        assert device == torch.device(requested)
        calls.append("reset")
        raise RuntimeError("test stops after initialized allocator reset")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset)
    with pytest.raises(RuntimeError, match="test stops after initialized allocator reset"):
        benchmark.run_model(runs[0][1], {}, directory=tmp_path / "run",
                            save_inducing_positions=False)
    assert calls == ["initialized", "reset"]


def test_cuda_initialization_failure_is_loud_and_does_not_reset_or_build(tmp_path, monkeypatch):
    _, runs = oisst_defaults(device="cuda:0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail(device):
        raise RuntimeError("original CUDA initialization failure")

    def forbidden(*args, **kwargs):
        pytest.fail("Memory reset/model build must not follow failed initialization")

    monkeypatch.setattr(torch.cuda, "synchronize", fail)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", forbidden)
    monkeypatch.setattr(benchmark, "build_model", forbidden)
    with pytest.raises(RuntimeError, match="original CUDA initialization failure"):
        benchmark.run_model(runs[0][1], {}, directory=tmp_path / "failed",
                            save_inducing_positions=False)
    failure = json.loads((tmp_path / "failed/failure.json").read_text())
    assert "original CUDA initialization failure" in failure["traceback"]


@pytest.mark.parametrize("model", ["svgp", "gpw", "prospar", "vnngp"])
def test_used_warmup_is_validated_without_changing_values(model, tmp_path):
    _, runs = oisst_defaults(device="cpu")
    config = replace(next(config for _, config in runs if config.model == model), epochs=2)
    with pytest.raises(ValueError, match=rf"{model}:.*warmup_epochs=20.*epochs=2"):
        benchmark.run_model(config, {}, directory=tmp_path / model, save_inducing_positions=False)
    assert config.epochs == 2 and config.warmup_epochs == 20


@pytest.mark.parametrize("field,value", [("epochs", 0), ("snapshot_every", 0)])
def test_exact_patchwork_still_rejects_invalid_active_controls(field, value, tmp_path):
    _, runs = oisst_defaults(device="cpu")
    config = replace(runs[-1][1], **{field: value})
    with pytest.raises(ValueError, match=rf"patchwork: {field}={value} must be positive"):
        benchmark.run_model(config, {}, directory=tmp_path / "run", save_inducing_positions=False)
