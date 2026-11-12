from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.default_hyperparameters import oisst_defaults
from experiments.run_selection import select_runs
from experiments.ukhousing_benchmark import build_model
from pwgp.initialization import hierarchical_initialization, hierarchical_supports
from pwgp.models.standard_patchwork import StandardPatchworkGP
from pwgp.partitions import checkerboard_partition, pc_tree_partition, voronoi_partition


@pytest.fixture
def comparison():
    torch.set_num_threads(2)
    x = torch.cartesian_prod(*[torch.linspace(0.025, 0.975, 20, dtype=torch.float64)] * 2)
    y = torch.sin(8 * x[:, 0]) + torch.cos(6 * x[:, 1])
    split = {"x": x, "y": y, "train": np.arange(len(x))}
    _, runs = oisst_defaults(device="cpu")
    selected = select_runs(runs, models=["GPW", "Standard Patchwork", "Standard Patchwork (GPW regions)",
                                       "Standard Patchwork (PC tree)"],
        overrides={"GPW": {"inducing_points": 64, "patches": 4, "patch_neighbors": 1,
                           "q_rank": 4, "stitches_per_edge": 2},
                   "Standard Patchwork": {"exact_stitches_per_edge": 2, "exact_maximum_points": 2},
                   "Standard Patchwork (GPW regions)": {"exact_stitches_per_edge": 2}})
    return selected, split


def test_initialization_preserves_selected_rows_and_centers(comparison):
    _, split = comparison
    state = hierarchical_initialization(split["x"], 4, 16, seed=31)
    torch.testing.assert_close(state["supports"], split["x"][state["indices"]], rtol=0, atol=0)
    torch.testing.assert_close(state["supports"], hierarchical_supports(split["x"], 4, 16, seed=31), rtol=0, atol=0)
    assert state["indices"].unique().numel() == 64
    labels = (state["supports"].reshape(-1, 1, 2) - state["centers"][None]).square().sum(-1).argmin(1)
    torch.testing.assert_close(labels, torch.arange(4).repeat_interleave(16))


def test_both_baselines_use_exact_gpw_initial_locations_and_targets(comparison):
    runs, split = comparison
    gpw = build_model(runs[0][1], split)
    for _, config in runs[1:]:
        model = build_model(config, split)
        torch.testing.assert_close(model.train_x, gpw.initial_support_locations.reshape(-1, 2), rtol=0, atol=0)
        torch.testing.assert_close(model.source_training_indices, gpw.source_initial_indices.flatten())
        torch.testing.assert_close(model.train_y, split["y"][model.source_training_indices])
        assert len(model.train_x) == 64 < len(split["train"])
        if config.exact_partition == "grid":
            assert model.patches == 8
            assert torch.bincount(model.labels).max() > config.exact_maximum_points
        elif config.exact_partition == "gpw_voronoi":
            torch.testing.assert_close(model.region_centers, gpw.initial_cluster_centers)
            torch.testing.assert_close(model.labels, torch.arange(4).repeat_interleave(16))
        else:
            assert model.patches == 8
            torch.testing.assert_close(torch.bincount(model.labels), torch.full((8,), 8))
        before = model.train_x.clone()
        with torch.no_grad():
            gpw.support.raw.add_(0.1)
        torch.testing.assert_close(model.train_x, before, rtol=0, atol=0)


def test_shared_initialization_is_used_without_reclustering(comparison, monkeypatch):
    from experiments import ukhousing_benchmark

    runs, split = comparison
    reference = runs[1][1].exact_gpw_reference
    state = hierarchical_initialization(split["x"], 4, 16, seed=31)
    shared = {**split, "gpw_initialization": state, "gpw_initialization_reference": reference}

    def forbidden(*args, **kwargs):
        raise AssertionError("Shared comparison initialization was recomputed")

    monkeypatch.setattr(ukhousing_benchmark, "hierarchical_initialization", forbidden)
    for _, config in runs:
        model = build_model(config, shared)
        if config.model == "gpw":
            torch.testing.assert_close(model.initial_support_locations, state["supports"], rtol=0, atol=0)
        else:
            torch.testing.assert_close(model.train_x, state["supports"].reshape(-1, 2), rtol=0, atol=0)
    with pytest.raises(ValueError, match="source training rows"):
        build_model(runs[1][1], {**shared, "gpw_initialization": {**state, "supports": state["supports"] + 0.001}})


def test_baseline_only_selection_keeps_reference_and_overrides(comparison):
    _, defaults = oisst_defaults(device="cpu")
    selected = select_runs(defaults, models=["Standard Patchwork (GPW regions)"],
                           overrides={"GPW": {"inducing_points": 64, "patches": 4},
                                      "Standard Patchwork (GPW regions)": {"epochs": 10}})
    config = selected[0][1]
    assert config.epochs == 10
    assert config.exact_gpw_reference == {"inducing_points": 64, "patches": 4, "seed": 31, "threads": 2}
    assert dict(defaults)["Standard Patchwork (GPW regions)"].exact_gpw_reference is None
    with pytest.raises(ValueError, match="reference differs"):
        select_runs(defaults, models=["Standard Patchwork"], overrides={"Standard Patchwork": {
            "exact_gpw_reference": {"inducing_points": 1, "patches": 1, "seed": 1}}})


def test_checkerboard_has_fixed_count_and_boundary_stitches():
    x = np.array([[(i + 0.5) / 4, (j + 0.5) / 2] for i in range(4) for j in range(2)])
    labels, bounds, points, edges, polygons = checkerboard_partition(x, grid_shape=(4, 2), stitches_per_edge=5)
    np.testing.assert_array_equal(labels, np.arange(8))
    assert len(bounds) == len(polygons) == 8 and len(points) == 50
    assert len(set(edges)) == 10
    for point, (i, j) in zip(points, edges, strict=True):
        assert ((point >= bounds[i, 0]) & (point <= bounds[i, 1])).all()
        assert ((point >= bounds[j, 0]) & (point <= bounds[j, 1])).all()


def test_voronoi_stitches_on_shared_linear_edges():
    centers = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]])
    labels, _, points, edges, _ = voronoi_partition(centers, centers=centers, stitches_per_edge=3)
    np.testing.assert_array_equal(labels, np.arange(3))
    assert len(points) == 9 and len(set(edges)) == 3
    for point, (i, j) in zip(points, edges, strict=True):
        distances = ((centers - point)**2).sum(1)
        assert distances[i] == pytest.approx(distances[j], abs=1e-12)
        assert distances[i] <= distances.min() + 1e-12
        assert ((point > 0) & (point < 1)).all()


def test_pc_tree_balances_oblique_regions_and_preserves_all_points():
    rng = np.random.default_rng(17)
    axes = rng.uniform([-0.35, -0.04], [0.35, 0.04], size=(128, 2))
    angle = np.pi / 6
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    x = 0.5 + axes @ rotation.T
    labels, _, points, edges, polygons, tree = pc_tree_partition(x, patches=8, stitches_per_edge=5, seed=31)
    np.testing.assert_array_equal(np.bincount(labels), np.full(8, 16))
    assert tree.shape == (15, 6) and np.abs(tree[0, :2]).min() > 0.2
    assert len(points) == 5 * len(set(edges))
    areas = []
    for polygon in polygons:
        areas.append(0.5 * np.abs(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                                            - polygon[:, 1] * np.roll(polygon[:, 0], -1))))
    assert sum(areas) == pytest.approx(1)
    for point, (i, j) in zip(points, edges, strict=True):
        for patch in (i, j):
            vertices = polygons[patch]
            sides = np.roll(vertices, -1, axis=0) - vertices
            delta = point - vertices
            cross = sides[:, 0] * delta[:, 1] - sides[:, 1] * delta[:, 0]
            assert cross.min() >= -1e-12 and np.abs(cross).min() < 1e-12
    a = pc_tree_partition(x, patches=8, stitches_per_edge=5, seed=31)
    b = pc_tree_partition(x, patches=8, stitches_per_edge=5, seed=32)
    np.testing.assert_array_equal(a[2], points)
    np.testing.assert_array_equal(b[0], labels)
    assert not np.array_equal(b[2], points)
    tx = torch.tensor(x, dtype=torch.float64)
    model = StandardPatchworkGP(tx, tx[:, 0], partition="pc_tree", tree_patches=8, seed=31)
    torch.testing.assert_close(model.assign_regions(tx), model.labels)
    root = model.partition_tree[0]
    at_cut = (root[:2] * root[2])[None]
    if bool(((at_cut >= 0) & (at_cut < 1)).all()):
        assert model.assign_regions(at_cut).item() < 4


@pytest.mark.parametrize("patches,count,message", [
    (3, 24, "power of two"), (8, 30, "divide evenly"), (8, 4, "divide evenly")])
def test_pc_tree_rejects_invalid_balanced_budgets(patches, count, message):
    x = np.random.default_rng(19).uniform(0.05, 0.95, size=(count, 2))
    with pytest.raises(ValueError, match=message):
        pc_tree_partition(x, patches=patches, stitches_per_edge=2, seed=31)


def test_pc_tree_unsplittable_projections_raise():
    with pytest.raises(ValueError, match="zero input variance"):
        pc_tree_partition(np.full((8, 2), 0.5), patches=2, stitches_per_edge=2, seed=31)
    x = np.array([[0.1, 0.5], [0.5, 0.5], [0.5, 0.5], [0.9, 0.5]])
    with pytest.raises(ValueError, match="projection ties"):
        pc_tree_partition(x, patches=2, stitches_per_edge=2, seed=31)


@pytest.mark.parametrize("partition", ["grid", "voronoi", "pc_tree"])
def test_fixed_partition_joint_likelihood_gradients_and_prediction_match_dense(partition):
    torch.manual_seed(8)
    x = 0.05 + 0.9 * torch.rand(20, 2, dtype=torch.float64)
    y = torch.sin(8 * x[:, 0])
    model = StandardPatchworkGP(x, y, partition=partition, grid_shape=(2, 1),
        centers=np.array([[0.25, 0.4], [0.75, 0.6]]) if partition == "voronoi" else None,
        tree_patches=2, seed=31, stitches_per_edge=2)
    n, q = len(x), len(model.stitch_points)
    ky = (model.hyp.noise_std().square() + model.jitter) * torch.eye(n, dtype=x.dtype)
    g, b = model.jitter * torch.eye(q, dtype=x.dtype), x.new_zeros(n, q)
    for patch in range(model.patches):
        mask = (model.labels == patch).double()
        signs = model.coefficients[patch]
        ky = ky + model.kernel(x, x, patch) * mask[:, None] * mask[None]
        g = g + model.kernel(model.stitch_points, model.stitch_points, patch) * signs[:, None] * signs[None]
        b = b + model.kernel(x, model.stitch_points, patch) * mask[:, None] * signs[None]
    joint = torch.cat((torch.cat((ky, b), 1), torch.cat((b.T, g), 1)), 0)
    target = torch.cat((y, x.new_zeros(q)))
    expected = -torch.distributions.MultivariateNormal(torch.zeros_like(target), joint).log_prob(target)
    state = model.conditioned_state()
    torch.testing.assert_close(state["nll"], expected)
    actual_grad = torch.autograd.grad(state["nll"], tuple(model.parameters()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tuple(model.parameters()), retain_graph=True)
    for actual, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, reference)
    query = 0.05 + 0.9 * torch.rand(6, 2, dtype=x.dtype)
    means, variances = [], []
    for point, patch in zip(query, model.assign_regions(query), strict=True):
        cross = torch.cat((model.kernel(point[None], x, patch).flatten() * (model.labels == patch),
            model.kernel(point[None], model.stitch_points, patch).flatten() * model.coefficients[patch]))
        means.append(cross @ torch.linalg.solve(joint, target))
        variances.append(model.hyp.outputscales()[patch] - cross @ torch.linalg.solve(joint, cross))
    mean, variance = model.predict(query, observation_noise=False)
    torch.testing.assert_close(mean, torch.stack(means))
    torch.testing.assert_close(variance, torch.stack(variances))


def test_empty_cells_and_invalid_reference_raise_without_changing_budget(comparison):
    runs, split = comparison
    with pytest.raises(ValueError, match="no selected training points"):
        StandardPatchworkGP(torch.tensor([[0.1, 0.1], [0.2, 0.2]], dtype=torch.float64),
                            torch.ones(2, dtype=torch.float64), partition="grid", grid_shape=(4, 2))
    with pytest.raises(ValueError, match="explicit GPW initialization"):
        build_model(replace(runs[1][1], exact_gpw_reference=None), split)
    with pytest.raises(ValueError, match="observation count"):
        build_model(replace(runs[1][1], exact_gpw_reference={"patches": 4, "inducing_points": 4000,
                                                          "seed": 31, "threads": 2}), split)
