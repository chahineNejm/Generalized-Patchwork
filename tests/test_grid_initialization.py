from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.gpw_ablation import bind_initialization, verify_geometry
from experiments.run_selection import bind_patchwork_reference
from experiments.ukhousing_benchmark import UKConfig, build_model, gpw_reference, initial_gpw_points
from pwgp.initialization import grid_initialization, grid_stitches


def split(n=24):
    axis = np.linspace(0, 1, n + 1)
    xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
    y = np.sin(6 * xy[:, 0]) + (xy[:, 0] > 0.5)
    rows = np.random.default_rng(0).permutation(len(y))
    return {"x": torch.tensor(xy), "y": torch.tensor(y), "train": rows[:500], "test": rows[500:],
            "validation": np.empty(0, dtype=np.int64)}


def config(**values):
    return replace(UKConfig(model="gpw", inducing_points=32, patches=2, stitches_per_edge=6, q_rank=2,
        patch_neighbors=1, gpw_partition="grid", gpw_grid_shape=(2, 1), device="cpu", batch_size=64), **values)


def test_grid_supports_lie_in_their_cell_and_stitches_on_the_boundary():
    x = split()["x"]
    initial = grid_initialization(x, (2, 1), 16, seed=31)
    supports = initial["supports"]
    assert supports.shape == (2, 16, 2)
    assert (supports[0, :, 0] < 0.5).all() and (supports[1, :, 0] >= 0.5).all()
    assert ((supports > 0) & (supports < 1)).all()
    np.testing.assert_allclose(initial["centers"].numpy(), [[0.25, 0.5], [0.75, 0.5]])
    assert torch.equal(x[initial["indices"]], supports)
    again = grid_initialization(x, (2, 1), 16, seed=31)
    assert torch.equal(again["indices"], initial["indices"])
    edges, stitches = grid_stitches(x, (2, 1), 6)
    assert edges == [(0, 1)] and stitches.shape == (1, 6, 2)
    np.testing.assert_allclose(stitches[0, :, 0].numpy(), 0.5)
    np.testing.assert_allclose(stitches[0, :, 1].numpy(), np.linspace(0, 1, 8)[1:-1])


@pytest.mark.parametrize("mode,constraints,trainable", [("none", 0, False), ("fixed", 6, False), ("learned", 6, True)])
def test_grid_gpw_modes_share_geometry_and_match_reference(mode, constraints, trainable):
    data = split()
    cfg = config(stitch_mode=mode)
    bind_initialization(data, cfg)
    model = build_model(cfg, data)
    verify_geometry(model, data)
    assert model.stitch_coefficients.shape[0] == constraints and model.stitch.trainable == trainable
    np.testing.assert_allclose(model.candidate_stitch_locations[0, :, 0].numpy(), 0.5)


def test_patchwork_reference_follows_grid_gpw_rows():
    data = split()
    gpw = config()
    patchwork = replace(UKConfig(model="patchwork", exact_training_points="gpw_initial", exact_partition="grid",
        exact_grid_shape=(2, 1), exact_stitches_per_edge=6, device="cpu"))
    [(_, bound)] = bind_patchwork_reference([("Standard Patchwork", patchwork)], gpw)
    assert bound.exact_gpw_reference == {**gpw_reference(gpw)} and bound.exact_gpw_reference["partition"] == "grid"
    rows = initial_gpw_points(data["x"][data["train"]], bound.exact_gpw_reference, data)["indices"]
    model = build_model(gpw, data)
    assert torch.equal(data["x"][data["train"]][rows.flatten()], model.initial_support_locations.reshape(-1, 2))


@pytest.mark.parametrize("values", [{"gpw_grid_shape": None}, {"gpw_grid_shape": (2, 2)},
    {"gpw_partition": "kmeans"}, {"gpw_partition": "line"}])
def test_invalid_grid_settings_raise(values):
    with pytest.raises(ValueError):
        gpw_reference(config(**values))


def test_kmeans_reference_is_unchanged():
    assert gpw_reference(replace(UKConfig(), gpw_partition="kmeans")) == {
        "inducing_points": 1024, "patches": 8, "seed": 31, "threads": 2}
