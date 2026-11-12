import json
import weakref
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments import multidataset_benchmark
from experiments.multidataset_benchmark import prepare_split, run_model
from experiments.ukhousing_benchmark import UKConfig
from pwgp.config import ProSparConfig
from pwgp.data.types import RegressionDataset


def dataset(dimension=3):
    x = np.random.default_rng(31).uniform(-3, 5, size=(160, dimension))
    y = 2 + np.sin(x[:, 0]) + 0.1 * x[:, -1]
    return RegressionDataset("test", x, y, tuple(f"feature_{i}" for i in range(dimension)), "target")


def test_explicit_80_20_input_bounds_and_training_only_target_scaling():
    data = dataset()
    split = prepare_split(data, train_fraction=0.8, seed=31)
    assert len(split["train"]) == 128 and len(split["test"]) == 32
    assert len(split["validation"]) == 0 and split["score_splits"] == ["test"]
    torch.testing.assert_close(split["x"].amin(0), torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(split["x"].amax(0), torch.ones(3, dtype=torch.float64))
    y = data.y.copy()
    y[split["test"]] += 1000
    changed = prepare_split(RegressionDataset("test", data.x, y, data.feature_names, "target"),
                            train_fraction=0.8, seed=31)
    assert changed["target_std"] == split["target_std"]
    assert changed["target_mean"] == split["target_mean"]
    assert split["preprocessing"]["input_statistics_fit_on"] == "all input rows; no target information"


def test_missing_rows_removed_not_filled_and_constant_features_raise():
    data = dataset()
    y = data.y.copy()
    y[0] = np.nan
    split = prepare_split(RegressionDataset("test", data.x, y, data.feature_names, "target"),
                          train_fraction=0.8, seed=31)
    assert len(split["y"]) == 159
    assert split["preprocessing"]["removed_nonfinite_at_split"] == 1
    x = data.x.copy()
    x[:, -1] = 1
    with pytest.raises(ValueError, match="never dropped"):
        prepare_split(RegressionDataset("test", x, data.y, data.feature_names, "target"),
                      train_fraction=0.8, seed=31)


@pytest.mark.parametrize("name", ["gpw", "svgp", "vnngp", "prospar", "patchwork"])
def test_highdim_runner_short_schedule_artifacts_and_memory(name, tmp_path, monkeypatch):
    split = prepare_split(dataset(9), train_fraction=0.8, seed=31)
    config = UKConfig(model=name, device="cpu", epochs=3, warmup_epochs=1, inducing_points=24,
        patches=4, q_rank=4, stitches_per_edge=2, batch_size=64, vnngp_neighbors=4,
        vnngp_batch_size=32, snapshot_every=3, prediction_batch_size=16,
        exact_training_points="gpw_initial", exact_partition="pc_tree", exact_tree_patches=4,
        exact_gpw_reference={"inducing_points": 24, "patches": 4, "seed": 31, "threads": 2},
        prospar=ProSparConfig(temperatures=1., exponent=1., inducing_nuggets=0.01, prior_jitter=0.))
    references, events = [], []
    original = multidataset_benchmark.build_model

    def build(*args):
        model = original(*args)
        references.append(weakref.ref(model))
        return model

    monkeypatch.setattr(multidataset_benchmark, "build_model", build)
    result = run_model(config, split, directory=tmp_path / name, save_inducing_positions=True,
                       on_snapshot=lambda path, epoch: events.append(epoch))
    assert references[0]() is None
    assert events == [0, 3]
    assert len(json.loads((result / "history.json").read_text())["losses"]) == 3
    assert json.loads((result / "status.json").read_text())["state"] == "complete"
    assert np.isfinite(json.loads((result / "metrics.json").read_text())["nlpd_standardized"])
    metadata = json.loads((result / "model_metadata.json").read_text())
    assert metadata["dimensions"] == 9
    assert (result / "visuals/predictions.png").is_file()
    if name == "prospar":
        assert metadata["paper_inducing_points"] == 5000
        assert metadata["actual_inducing_points"] == 24
    if name == "patchwork":
        assert metadata["training_observations_used"] == 24
        assert not (result / "inducing_positions.npz").exists()
    else:
        with np.load(result / "inducing_positions.npz") as saved:
            assert saved["supports"].shape[-1] == 9


def test_invalid_short_warmup_raises_and_is_recorded(tmp_path):
    config = replace(UKConfig(), device="cpu", epochs=3, warmup_epochs=20)
    split = prepare_split(dataset(), train_fraction=0.8, seed=31)
    with pytest.raises(ValueError, match="no automatic"):
        run_model(config, split, directory=tmp_path / "bad", save_inducing_positions=False)
    assert "no automatic" in (tmp_path / "bad/failure.json").read_text()
