"""Direct-observation Patchwork Kriging with fixed rectangular regions."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from pwgp.config import DEFAULT_HYPERPARAMETER_BOUNDS
from pwgp.linalg import cholesky_solve
from pwgp.parameters import LocalARDHyperparameters


def rectangular_partition(x: np.ndarray, *, maximum_points: int, stitches_per_edge: int):
    if x.ndim != 2 or x.shape[1] != 2 or maximum_points < 2 or stitches_per_edge < 1:
        raise ValueError("Expected 2D inputs, maximum_points >= 2 and positive stitch count")
    if not np.isfinite(x).all() or (x <= 0).any() or (x >= 1).any():
        raise ValueError("Partition training inputs must be strictly inside the unit square")
    leaves = []

    def split(indices, lower, upper):
        if len(indices) <= maximum_points:
            leaves.append((indices, lower, upper))
            return
        values = x[indices]
        dimension = int(np.ptp(values, axis=0).argmax())
        unique = np.unique(values[:, dimension])
        if len(unique) < 2:
            raise ValueError("Too many coincident observations for the declared patch size")
        middle = np.searchsorted(unique, np.median(values[:, dimension]))
        middle = max(1, min(middle, len(unique) - 1))
        cut = 0.5 * (unique[middle - 1] + unique[middle])
        left = values[:, dimension] < cut
        upper_left, lower_right = upper.copy(), lower.copy()
        upper_left[dimension] = lower_right[dimension] = cut
        split(indices[left], lower, upper_left)
        split(indices[~left], lower_right, upper)

    split(np.arange(len(x)), np.zeros(2), np.ones(2))
    labels = np.empty(len(x), dtype=np.int64)
    bounds = np.array([[lower, upper] for _, lower, upper in leaves])
    for patch, (indices, _, _) in enumerate(leaves):
        labels[indices] = patch
    points, edges = [], []
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            for dimension in range(2):
                boundary = None
                if np.isclose(bounds[i, 1, dimension], bounds[j, 0, dimension], atol=1e-12, rtol=0):
                    boundary = bounds[i, 1, dimension]
                elif np.isclose(bounds[j, 1, dimension], bounds[i, 0, dimension], atol=1e-12, rtol=0):
                    boundary = bounds[j, 1, dimension]
                other = 1 - dimension
                lo, hi = max(bounds[i, 0, other], bounds[j, 0, other]), min(bounds[i, 1, other], bounds[j, 1, other])
                if boundary is not None and hi - lo > 1e-12:
                    for value in np.linspace(lo, hi, stitches_per_edge + 2)[1:-1]:
                        point = np.empty(2)
                        point[dimension], point[other] = boundary, value
                        points.append(point)
                        edges.append((i, j))
    if len(leaves) > 1 and not edges:
        raise ValueError("No shared patch boundaries found")
    return labels, bounds, np.asarray(points).reshape(-1, 2), edges


class StandardPatchworkGP(nn.Module):
    """Condition directly on all noisy observations and zero stitch differences."""

    def __init__(self, x, y, *, maximum_points=512, stitches_per_edge=2,
                 lengthscale=0.1, outputscale=1.0, noise_std=0.1, jitter=1e-4):
        super().__init__()
        labels, bounds, points, edges = rectangular_partition(
            x.detach().cpu().numpy(), maximum_points=maximum_points,
            stitches_per_edge=stitches_per_edge,
        )
        self.patches = len(bounds)
        self.jitter = jitter
        self.register_buffer("train_x", x.clone())
        self.register_buffer("train_y", y.clone())
        self.register_buffer("labels", torch.as_tensor(labels, device=x.device))
        self.register_buffer("region_bounds", torch.as_tensor(bounds, dtype=x.dtype, device=x.device))
        self.register_buffer("stitch_points", torch.as_tensor(points, dtype=x.dtype, device=x.device))
        coefficients = x.new_zeros(self.patches, len(points))
        for q, (i, j) in enumerate(edges):
            coefficients[i, q], coefficients[j, q] = 1, -1
        self.register_buffer("coefficients", coefficients)
        self.hyp = LocalARDHyperparameters(
            self.patches, 2, lengthscale=lengthscale, outputscale=outputscale,
            noise_std=noise_std, bounds=DEFAULT_HYPERPARAMETER_BOUNDS,
            dtype=x.dtype, device=x.device, trainable=True,
        )

    def kernel(self, x, y, patch):
        delta = (x[:, None] - y[None]) / self.hyp.lengthscales()[patch]
        return self.hyp.outputscales()[patch] * torch.exp(-0.5 * delta.square().sum(-1))

    def conditioned_state(self):
        q = len(self.stitch_points)
        identity = torch.eye(q, dtype=self.train_x.dtype, device=self.train_x.device)
        g = self.jitter * identity
        reduction = torch.zeros_like(g)
        b = self.train_x.new_zeros(q, 1)
        base_nll = self.train_x.new_zeros(())
        blocks = []
        for patch in range(self.patches):
            indices = (self.labels == patch).nonzero().flatten()
            local_x, local_y = self.train_x[indices], self.train_y[indices, None]
            active = self.coefficients[patch].nonzero().flatten()
            signs = self.coefficients[patch, active]
            stitches = self.stitch_points[active]
            ky = self.kernel(local_x, local_x, patch)
            ky = ky + (self.hyp.noise_std().square() + self.jitter) * torch.eye(
                len(local_x), dtype=ky.dtype, device=ky.device)
            factor = torch.linalg.cholesky(ky)
            cross = self.kernel(local_x, stitches, patch) * signs
            solved = cholesky_solve(torch.cat((local_y, cross), dim=1), factor)
            inverse_y, inverse_cross = solved[:, :1], solved[:, 1:]
            base_nll = base_nll + 0.5 * (
                (local_y * inverse_y).sum() + 2 * factor.diagonal().log().sum()
                + len(local_x) * math.log(2 * math.pi))
            stitch_covariance = self.kernel(stitches, stitches, patch) * signs[:, None] * signs[None]
            positions = (active[:, None] * q + active[None]).flatten()
            g = g.flatten().index_add(0, positions, stitch_covariance.flatten()).reshape(q, q)
            reduction = reduction.flatten().index_add(
                0, positions, (cross.T @ inverse_cross).flatten()).reshape(q, q)
            b = b.index_add(0, active, cross.T @ inverse_y)
            blocks.append((indices, active, signs, factor, inverse_y, inverse_cross))
        t = 0.5 * ((g - reduction) + (g - reduction).T)
        factor_t = torch.linalg.cholesky(t)
        inverse_b = cholesky_solve(b, factor_t)
        nll = base_nll + 0.5 * (
            2 * factor_t.diagonal().log().sum() + (b * inverse_b).sum()
            + q * math.log(2 * math.pi))
        return {"nll": nll, "blocks": blocks, "factor_t": factor_t, "inverse_b": inverse_b}

    def loss(self):
        return self.conditioned_state()["nll"] / len(self.train_x)

    @torch.no_grad()
    def predict(self, x, *, observation_noise=True, state=None):
        state = self.conditioned_state() if state is None else state
        inside = ((x[:, None] >= self.region_bounds[None, :, 0])
                  & (x[:, None] < self.region_bounds[None, :, 1])).all(-1)
        if not bool((inside.sum(-1) == 1).all()):
            raise ValueError("Each query must belong to exactly one fixed patch")
        assigned = inside.long().argmax(-1)
        mean, variance = x.new_empty(len(x)), x.new_empty(len(x))
        for patch, block in enumerate(state["blocks"]):
            take = (assigned == patch).nonzero().flatten()
            if not len(take):
                continue
            indices, active, signs, factor, inverse_y, inverse_cross = block
            query = x[take]
            cross = self.kernel(query, self.train_x[indices], patch)
            d_local = self.kernel(query, self.stitch_points[active], patch) * signs - cross @ inverse_cross
            d = x.new_zeros(len(take), len(self.stitch_points))
            d[:, active] = d_local
            mean[take] = (cross @ inverse_y - d @ state["inverse_b"]).flatten()
            variance[take] = self.hyp.outputscales()[patch] - (
                cross * cholesky_solve(cross.T, factor).T).sum(-1)
            variance[take] -= (d * cholesky_solve(d.T, state["factor_t"]).T).sum(-1)
        if not bool(torch.isfinite(mean).all() & torch.isfinite(variance).all()) or bool((variance <= 0).any()):
            raise FloatingPointError("Invalid direct Patchwork latent variance")
        if observation_noise:
            variance = variance + self.hyp.noise_std().square()
        return mean, variance
