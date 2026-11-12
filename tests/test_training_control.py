import json

import numpy as np
import pytest
import torch

from experiments.multidataset_benchmark import prepare_split, run_model
from experiments.training_control import EarlyStoppingConfig, TrainingLossStopper
from experiments.ukhousing_benchmark import UKConfig, build_model, train
from pwgp.config import ProSparConfig
from pwgp.data.types import RegressionDataset


def test_warmup_ignored_then_exact_ten_epoch_patience():
    stopper = TrainingLossStopper(EarlyStoppingConfig(10, .001), warmup_epochs=20)
    for epoch in range(1, 31):
        assert stopper.update(epoch, 100.) is False
    assert stopper.best_epoch == 21 and stopper.bad_epochs == 9
    assert stopper.update(31, 100.) is True
    with pytest.raises(ValueError, match="stop after triggering"):
        stopper.update(32, 100.)


@pytest.mark.parametrize("values,expected", [
    ([100., 99.95, 99.89], (99.89, 0)),
    ([-1., -1.0005, -1.002], (-1.002, 0)),
    ([0., 0., 0.], (0., 2)),
    ([0., 0., -.001], (-.001, 0))])
def test_relative_threshold_handles_negative_and_zero_losses(values, expected):
    stopper = TrainingLossStopper(EarlyStoppingConfig(10, .001), warmup_epochs=0)
    for epoch, loss in enumerate(values, 1):
        assert stopper.update(epoch, loss) is False
    assert (stopper.best_loss, stopper.bad_epochs) == expected


@pytest.mark.parametrize("values", [(0, .001), (True, .001), (10, float("nan")), (10, -.001)])
def test_invalid_controls_raise(values):
    with pytest.raises(ValueError):
        EarlyStoppingConfig(*values)


def test_nonfinite_loss_and_out_of_order_epochs_raise():
    stopper = TrainingLossStopper(EarlyStoppingConfig(10, .001), warmup_epochs=20)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        stopper.update(1, float("nan"))
    with pytest.raises(ValueError, match="consecutive"):
        stopper.update(2, 1.)


def fixture_split():
    x = np.random.default_rng(31).uniform(0, 1, (160, 3))
    data = RegressionDataset("test", x, np.sin(7*x[:, 0]), ("x1", "x2", "x3"), "target")
    return prepare_split(data, train_fraction=.8, seed=31)


def small_config(name, **values):
    return UKConfig(model=name, device="cpu", epochs=8, warmup_epochs=1, inducing_points=24,
        patches=4, q_rank=2, stitches_per_edge=2, batch_size=128, vnngp_neighbors=4, vnngp_batch_size=128,
        prediction_batch_size=32, exact_training_points="gpw_initial", exact_partition="pc_tree", exact_tree_patches=4,
        exact_gpw_reference={"inducing_points": 24, "patches": 4, "seed": 31, "threads": 2},
        prospar=ProSparConfig(temperatures=1., exponent=1., inducing_nuggets=.01, prior_jitter=0.), **values)


@pytest.mark.parametrize("name", ["gpw", "svgp", "prospar", "patchwork", "vnngp"])
def test_all_training_loops_honor_explicit_stop_callback(name):
    torch.set_num_threads(2)
    split, config = fixture_split(), small_config(name)
    model = build_model(config, split)
    epochs = []

    def callback(epoch, losses):
        epochs.append(epoch)
        return epoch == 3

    losses = train(model, split, config, callback)
    assert epochs == [1, 2, 3] and len(losses) == 3
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_stopped_run_saves_last_model_final_snapshot_and_summary(tmp_path):
    from dataclasses import replace

    config = replace(small_config("svgp"), epochs=100, warmup_epochs=2, snapshot_every=10,
        learning_rate_q=0., learning_rate_structure=0., learning_rate_hyperparameters=0.)
    events = []
    directory = run_model(config, fixture_split(), directory=tmp_path / "run", save_inducing_positions=True,
        early_stopping={"patience": 2, "relative_min_improvement": .001},
        on_snapshot=lambda directory, epoch: events.append(epoch))
    summary = json.loads((directory / "training_summary.json").read_text())
    assert summary["epochs_completed"] == 5 and summary["stopped_early"] is True
    assert summary["checkpoint_restored"] is False and summary["evaluated_epoch"] == 5
    assert summary["reason"] == "training_loss_patience" and summary["significant_best_epoch"] == 3
    assert events == [0, 5]
    with np.load(directory / "inducing_positions.npz") as positions:
        np.testing.assert_array_equal(positions["epochs"], [0, 5])
    final = torch.load(directory / "model.pt", weights_only=True)
    checkpoint = torch.load(directory / "checkpoint.pt", weights_only=True)
    for key in final:
        torch.testing.assert_close(final[key], checkpoint[key])
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["epochs_completed"] == 5 and metrics["stopped_early"] is True
