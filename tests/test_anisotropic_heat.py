import json
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from experiments.anisotropic_heat import HeatConfig, heat_operator, run, simulate, support_clouds


def test_heat_operator_and_solution_match_dense_reference():
    config = HeatConfig(side=8)
    xy, operator = heat_operator(config.side, config.diffusivity)
    matrix = operator.toarray()
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(matrix.sum(1), 0, atol=1e-12)
    assert np.linalg.eigvalsh(matrix).max() < 1e-10
    _, initial, temperature = simulate(config)
    np.testing.assert_allclose(temperature, expm(config.time * matrix) @ initial, atol=1e-12)
    assert xy.shape == (64, 2)
    np.testing.assert_allclose(temperature.sum(), initial.sum(), rtol=1e-12)
    assert matrix[0, 8] > matrix[0, 1]
    assert matrix[56, 57] > matrix[48, 56]


def test_cloud_orientation_preserves_distances_without_projection():
    config = HeatConfig()
    a, b = support_clouds(config, (0, 90)).numpy(), support_clouds(config, (90, 0)).numpy()
    for patch in range(2):
        np.testing.assert_allclose(np.linalg.norm(a[patch, :, None] - a[patch, None], axis=-1),
                                   np.linalg.norm(b[patch, :, None] - b[patch, None], axis=-1), atol=1e-14)
    with pytest.raises(ValueError, match="strictly inside"):
        support_clouds(replace(config, cloud_axes=(2, 2)), (0, 90))


def test_heat_experiment_saves_points_predictions_and_failure(tmp_path):
    config = HeatConfig(side=12, observations=40, supports_per_patch=8, stitches=3, q_rank=2,
                        epochs=2, warmup_epochs=1, orientations=((0, 90),), snapshot_every=1)
    output = run(config, directory=tmp_path / "success")
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    with np.load(output / "orientation_0/predictions.npz") as data:
        assert data["initial_supports"].shape == (2, 8, 2)
        assert np.linalg.norm(data["final_supports"] - data["initial_supports"]) > 0
        assert (data["latent_variance"] > 0).all()
    assert (output / "points.png").stat().st_size > 0
    with pytest.raises(ValueError, match="side must"):
        run(replace(config, side=11), directory=tmp_path / "failed")
    failed = json.loads((tmp_path / "failed/status.json").read_text())
    assert failed["state"] == "failed" and "ValueError" in failed["traceback"]
