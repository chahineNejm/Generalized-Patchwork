"""Exact generator and original-unit variance checks for the local toy."""

import json

import numpy as np
import pytest

from experiments.default_hyperparameters import gpw_exponential_toy_defaults
from experiments.exploratory_analysis import toy_prediction_fields
from pwgp.data.gp_toy import exponential_covariance, sample_exponential_gp


def test_exponential_distance_not_squared():
    covariance = exponential_covariance([[0., 0.], [3., 4.]], variance=10., lengthscale=1.)
    np.testing.assert_allclose(covariance, [[10., 10*np.exp(-5)], [10*np.exp(-5), 10.]])


def test_joint_draw_reproducible():
    settings = {"points": 40, "extent": 6., "variance": 10., "lengthscale": 1.,
                "noise_variance": 1., "seed": 31, "threads": 2}
    data, latent = sample_exponential_gp(**settings)
    repeated, other_latent = sample_exponential_gp(**settings)
    np.testing.assert_array_equal(data.x, repeated.x)
    np.testing.assert_array_equal(data.y, repeated.y)
    np.testing.assert_array_equal(latent, other_latent)
    assert data.x.shape == (40, 2)
    assert data.metadata["sampling_jitter"] == 0
    assert not np.array_equal(latent, data.y)


def test_invalid_covariance_raises():
    with pytest.raises(ValueError, match="positive"):
        exponential_covariance([[0., 0.]], variance=0., lengthscale=1.)


def test_saved_variance_noise_once(tmp_path):
    (tmp_path / "learned_parameters.json").write_text(json.dumps({"noise_std": .2}))
    np.savez(tmp_path / "test_predictions.npz", observation_variance=[2., 3.], mean=[0., 1.])
    values = toy_prediction_fields(tmp_path, target_std=5.)
    np.testing.assert_allclose(values["latent_variance"], [1., 2.])
    assert values["noise_variance"] == 1.
    np.savez(tmp_path / "test_predictions.npz", observation_variance=[.5])
    with pytest.raises(ValueError, match="smaller"):
        toy_prediction_fields(tmp_path, target_std=5.)


def test_approved_defaults():
    settings = gpw_exponential_toy_defaults()
    assert settings["points"] == 8000
    assert settings["q_rank"] == 64
    assert settings["seeds"] == [31]
    assert settings["stitch_modes"] == ["none", "fixed", "learned"]
    assert [(v["patches"], v["inducing_points"], v["stitches_per_edge"]) for v in settings["configurations"]] == [
        (8, 256, 8), (4, 1024, 16)]


def test_tiny_runner_all_modes(tmp_path):
    from experiments.exponential_toy import run_toy

    settings = {**gpw_exponential_toy_defaults(), "points": 120, "q_rank": 2,
                "epochs": 2, "warmup_epochs": 1, "early_stopping": None,
                "configurations": [{"name": "test_only", "inducing_points": 12,
                                    "patches": 3, "stitches_per_edge": 2}]}
    output = run_toy(directory=tmp_path / "toy", settings=settings)
    assert json.loads((output / "status.json").read_text())["completed_runs"] == 3
    identities = []
    for mode in settings["stitch_modes"]:
        run = output / "test_only" / f"seed31_{mode}"
        identities.append(json.loads((run / "initialization_identity.json").read_text())["geometry_sha256"])
        with np.load(run / "inducing_positions.npz") as saved:
            assert saved["epochs"].tolist() == [0, 2]
        assert (run / "visuals/field_predictions.png").stat().st_size > 1000
        assert (output / f"test_only_seed31_{mode}.zip").is_file()
    assert len(set(identities)) == 1
    assert (output / "visuals/test_only_seed31/means_variances.png").is_file()
