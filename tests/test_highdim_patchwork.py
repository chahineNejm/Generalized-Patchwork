import numpy as np
import pytest
import torch

from pwgp.models.standard_patchwork import StandardPatchworkGP
from pwgp.partitions import pc_tree_partition


@pytest.mark.parametrize("dimension", [3, 9, 21])
def test_pc_tree_uses_every_dimension_and_reproducible_shared_faces(dimension):
    x = np.random.default_rng(31).uniform(0.05, 0.95, size=(64, dimension))
    a = pc_tree_partition(x, patches=4, stitches_per_edge=3, seed=31)
    b = pc_tree_partition(x, patches=4, stitches_per_edge=3, seed=31)
    np.testing.assert_array_equal(np.bincount(a[0]), [16] * 4)
    np.testing.assert_array_equal(a[2], b[2])
    assert a[5].shape == (7, dimension + 4)
    assert ((a[2] > 0) & (a[2] < 1)).all()
    for point, (i, j) in zip(a[2], a[3], strict=True):
        for patch in (i, j):
            node = 0
            for _ in range(2):
                row = a[5][node]
                leaves_left = []
                todo = [int(row[dimension + 1])]
                while todo:
                    current = a[5][todo.pop()]
                    if current[-1] >= 0:
                        leaves_left.append(int(current[-1]))
                    else:
                        todo.extend(current[dimension + 1:dimension + 3].astype(int))
                margin = point @ row[:dimension] - row[dimension]
                if patch in leaves_left:
                    assert margin <= 1e-10
                    node = int(row[dimension + 1])
                else:
                    assert margin >= -1e-10
                    node = int(row[dimension + 2])
    tx = torch.tensor(x, dtype=torch.float64)
    model = StandardPatchworkGP(tx, tx[:, 0], partition="pc_tree", tree_patches=4,
                               stitches_per_edge=3, seed=31)
    torch.testing.assert_close(model.assign_regions(tx), model.labels)
    assert model.hyp.lengthscales().shape == (4, dimension)
    assert model.region_vertices.shape == (0, dimension)
    assert model.assign_regions(torch.stack((tx.new_zeros(dimension), tx.new_ones(dimension)))).shape == (2,)


@pytest.mark.parametrize("dimension", [3, 9])
def test_highdim_joint_values_gradients_and_predictions_match_dense(dimension):
    x = torch.tensor(np.random.default_rng(19).uniform(0.05, 0.95, (24, dimension)), dtype=torch.float64)
    y = torch.sin(x[:, 0] * 5) + x[:, -1]
    model = StandardPatchworkGP(x, y, partition="pc_tree", tree_patches=4, stitches_per_edge=2, seed=31)
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
    query = torch.tensor(np.random.default_rng(17).uniform(0, 1, (7, dimension)), dtype=x.dtype)
    means, variances = [], []
    for point, patch in zip(query, model.assign_regions(query), strict=True):
        cross = torch.cat((model.kernel(point[None], x, patch).flatten() * (model.labels == patch),
            model.kernel(point[None], model.stitch_points, patch).flatten() * model.coefficients[patch]))
        means.append(cross @ torch.linalg.solve(joint, target))
        variances.append(model.hyp.outputscales()[patch] - cross @ torch.linalg.solve(joint, cross))
    mean, variance = model.predict(query, observation_noise=False)
    torch.testing.assert_close(mean, torch.stack(means))
    torch.testing.assert_close(variance, torch.stack(variances))


@pytest.mark.parametrize("values,message", [
    (np.full((16, 3), 0.5), "zero input variance"),
    (np.ones((16, 3)), "strictly inside"),
    (np.full((15, 3), 0.5), "divide evenly")])
def test_highdim_invalid_geometry_raises(values, message):
    with pytest.raises(ValueError, match=message):
        pc_tree_partition(values, patches=4, stitches_per_edge=2, seed=31)
