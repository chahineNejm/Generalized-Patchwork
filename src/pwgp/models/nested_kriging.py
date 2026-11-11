"""Two-layer Nested Kriging with shared kernels and correlated exact submodels."""

from __future__ import annotations

import math
import time
from dataclasses import asdict

import torch
from torch import nn
from tqdm.auto import tqdm

from pwgp.config import DEFAULT_HYPERPARAMETER_BOUNDS, HyperparameterBounds
from pwgp.kernels import ard_rbf
from pwgp.linalg import cholesky_solve
from pwgp.parameters import bounded_sigmoid, inverse_bounded_sigmoid
from pwgp.training import FitHistory


class NestedKriging(nn.Module):
    """Aggregate disjoint exact GP experts using Cov(M,M)^-1 Cov(M,f)."""

    kind = "nested_kriging"

    def __init__(
        self, x: torch.Tensor, y: torch.Tensor, groups: torch.Tensor, *,
        kernel: str = "matern52", lengthscale: float | torch.Tensor = 0.2,
        outputscale: float = 1.0, noise_std: float = 0.0, jitter: float = 0.0,
        train_hyperparameters: bool = True,
        bounds: HyperparameterBounds = DEFAULT_HYPERPARAMETER_BOUNDS,
    ) -> None:
        super().__init__()
        if x.ndim != 2 or min(x.shape) < 1 or x.dtype not in (torch.float32, torch.float64):
            raise ValueError("Expected nonempty floating x[N,D]")
        if y.shape != (len(x),) or y.dtype != x.dtype or y.device != x.device:
            raise ValueError("Expected y[N] with the same dtype/device as x")
        if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
            raise ValueError("Training inputs and targets must be finite")
        if groups.shape != y.shape or groups.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected integer groups[N]")
        if groups.device != x.device:
            raise ValueError("groups must have the same device as x")
        unique = groups.unique(sorted=True)
        if not torch.equal(unique, torch.arange(len(unique), device=x.device)):
            raise ValueError("Group labels must be contiguous from zero, with every group nonempty")
        if kernel not in ("rbf", "matern52"):
            raise ValueError("kernel must be 'rbf' or 'matern52'")
        if any(not math.isfinite(value) or value < 0 for value in (noise_std, jitter)):
            raise ValueError("noise_std and jitter must be finite and nonnegative")
        ls = torch.as_tensor(lengthscale, dtype=x.dtype, device=x.device).reshape(-1)
        if ls.numel() == 1:
            ls = ls.repeat(x.shape[1])
        if ls.shape != (x.shape[1],) or not bool(torch.isfinite(ls).all()):
            raise ValueError("Expected one finite lengthscale per input dimension")
        if not math.isfinite(outputscale):
            raise ValueError("outputscale must be finite")
        self.kernel_name, self.bounds, self.jitter = kernel, bounds, jitter
        self.patches = len(unique)
        self.register_buffer("train_x", x.detach().clone())
        self.register_buffer("train_y", y.detach().clone())
        self.register_buffer("groups", groups.detach().clone())
        self.register_buffer("noise_std", x.new_tensor(noise_std))
        options = {"dtype": x.dtype, "device": x.device}
        self.raw_lengthscales = nn.Parameter(
            inverse_bounded_sigmoid(ls, *bounds.lengthscale, **options),
            requires_grad=train_hyperparameters,
        )
        self.raw_outputscale = nn.Parameter(
            inverse_bounded_sigmoid(outputscale, *bounds.outputscale, **options),
            requires_grad=train_hyperparameters,
        )

    def lengthscales(self) -> torch.Tensor:
        return bounded_sigmoid(self.raw_lengthscales, self.bounds.lengthscale)

    def outputscale(self) -> torch.Tensor:
        return bounded_sigmoid(self.raw_outputscale, self.bounds.outputscale)

    def kernel(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if self.kernel_name == "rbf":
            return ard_rbf(x, z, self.lengthscales(), self.outputscale())
        distance = torch.linalg.vector_norm(
            (x[:, None] - z[None]) / self.lengthscales(), dim=-1,
        )
        radius = math.sqrt(5) * distance
        return self.outputscale() * (1 + radius + radius.square() / 3) * torch.exp(-radius)

    def conditioned_state(self, *, exclude: int | None = None) -> list[dict]:
        state = []
        for group in range(self.patches):
            indices = (self.groups == group).nonzero().flatten()
            if exclude is not None:
                indices = indices[indices != exclude]
            if not len(indices):
                raise ValueError("Each expert must remain nonempty after leave-one-out exclusion")
            x = self.train_x[indices]
            covariance = self.kernel(x, x) + (self.noise_std.square() + self.jitter) * torch.eye(
                len(x), dtype=x.dtype, device=x.device,
            )
            state.append({"x": x, "y": self.train_y[indices],
                          "cholesky": torch.linalg.cholesky(covariance)})
        return state

    def components(self, x: torch.Tensor, *, state: list[dict] | None = None) -> dict:
        """Return exact expert cross-covariances, NK coefficients and latent moments."""
        if x.ndim != 2 or x.shape[1] != self.train_x.shape[1] or not len(x):
            raise ValueError("Expected nonempty query x[B,D] matching training dimensions")
        if x.dtype != self.train_x.dtype or x.device != self.train_x.device:
            raise ValueError("Query dtype/device must match training inputs")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("Query inputs must be finite")
        state = self.conditioned_state() if state is None else state
        local_weights, local_means, cross = [], [], []
        for expert in state:
            k = self.kernel(expert["x"], x)
            weights = cholesky_solve(k, expert["cholesky"]).T
            local_weights.append(weights)
            local_means.append(weights @ expert["y"])
            cross.append((weights * k.T).sum(-1))
        k_m = torch.stack(cross, dim=-1)
        entries = {}
        for i, expert in enumerate(state):
            entries[i, i] = cross[i]
            for j in range(i):
                # Cross-expert predictors share the same prior, not independent GP priors.
                block = self.kernel(expert["x"], state[j]["x"])
                entries[i, j] = entries[j, i] = torch.einsum(
                    "bi,ij,bj->b", local_weights[i], block, local_weights[j],
                )
        covariance = torch.stack([
            torch.stack([entries[i, j] for j in range(len(state))], dim=-1)
            for i in range(len(state))
        ], dim=-2)
        beta = cholesky_solve(k_m[..., None], torch.linalg.cholesky(covariance)).squeeze(-1)
        means = torch.stack(local_means, dim=-1)
        mean = (beta * means).sum(-1)
        variance = self.outputscale() - (beta * k_m).sum(-1)
        if not bool(torch.isfinite(mean).all() & torch.isfinite(variance).all()):
            raise FloatingPointError("Nested Kriging predictions must be finite")
        if bool((variance < 0).any()):
            raise FloatingPointError("Nested Kriging latent variance must be nonnegative")
        return {"local_mean": means, "covariance": covariance, "cross_covariance": k_m,
                "weights": beta, "mean": mean, "variance": variance}

    def latent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        parts = self.components(x)
        return parts["mean"], parts["variance"]

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, *, observation_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        mean, variance = self.latent(x)
        return mean, variance + self.noise_std.square() if observation_noise else variance

    def leave_one_out(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if indices.ndim != 1 or not len(indices) or indices.dtype != torch.int64:
            raise ValueError("Expected nonempty int64 leave-one-out indices")
        if bool(((indices < 0) | (indices >= len(self.train_x))).any()):
            raise ValueError("Leave-one-out indices must be valid training rows")
        means, variances = [], []
        for index in indices.tolist():
            parts = self.components(self.train_x[index:index + 1],
                                    state=self.conditioned_state(exclude=index))
            means.append(parts["mean"].squeeze(0))
            variances.append(parts["variance"].squeeze(0))
        return torch.stack(means), torch.stack(variances)

    @torch.no_grad()
    def calibrate_outputscale(self) -> torch.Tensor:
        """Apply the paper's noiseless LOO variance calibration, equation (18)."""
        if self.noise_std.item() != 0 or self.jitter != 0:
            raise ValueError("Paper variance calibration requires noise_std=0 and jitter=0")
        indices = torch.arange(len(self.train_x), device=self.train_x.device)
        mean, variance = self.leave_one_out(indices)
        if bool((variance <= 0).any()):
            raise FloatingPointError("LOO variance calibration requires strictly positive variances")
        value = self.outputscale() * ((self.train_y - mean).square() / variance).mean()
        self.raw_outputscale.copy_(inverse_bounded_sigmoid(
            value, *self.bounds.outputscale, dtype=value.dtype, device=value.device,
        ))
        return value

    def fit(
        self, *, epochs: int, learning_rate: float, loo_batch_size: int,
        seed: int = 0, calibrate_variance: bool = True, on_epoch=None,
    ) -> FitHistory:
        """Fit lengthscales by LOO MSE using Adam, not the paper's SPSA optimizer."""
        if epochs < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Require positive epochs and finite positive learning_rate")
        if not 1 <= loo_batch_size <= len(self.train_x):
            raise ValueError("Require 1 <= loo_batch_size <= training count")
        if not self.raw_lengthscales.requires_grad:
            raise ValueError("Lengthscale fitting requires train_hyperparameters=True")
        if bool((torch.bincount(self.groups.long()) < 2).any()):
            raise ValueError("LOO fitting requires at least two observations in every expert")
        if calibrate_variance and (self.noise_std.item() != 0 or self.jitter != 0):
            raise ValueError("Paper variance calibration requires noise_std=0 and jitter=0")
        self.train()
        optimizer = torch.optim.Adam([self.raw_lengthscales], lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        history, started = FitHistory(), time.perf_counter()
        for epoch in (bar := tqdm(range(epochs), desc="train:nested-kriging", unit="epoch")):
            indices = torch.randperm(len(self.train_x), generator=generator)[:loo_batch_size]
            indices = indices.to(self.train_x.device)
            self.zero_grad(set_to_none=True)
            mean, _ = self.leave_one_out(indices)
            loss = (mean - self.train_y[indices]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nested Kriging LOO loss must be finite")
            loss.backward()
            if not bool(torch.isfinite(self.raw_lengthscales.grad).all()):
                raise FloatingPointError("Nested Kriging lengthscale gradients must be finite")
            optimizer.step()
            history.losses.append(float(loss.detach()))
            history.elapsed_seconds = time.perf_counter() - started
            bar.set_postfix(loss=history.losses[-1])
            if on_epoch is not None:
                on_epoch(epoch + 1, self, history)
        if calibrate_variance:
            self.calibrate_outputscale()
        history.elapsed_seconds = time.perf_counter() - started
        return history

    def configuration(self) -> dict:
        return {"model_class": type(self).__name__, "kernel": self.kernel_name,
                "lengthscale": self.lengthscales().detach().cpu().tolist(),
                "outputscale": self.outputscale().item(), "noise_std": self.noise_std.item(),
                "jitter": self.jitter, "bounds": asdict(self.bounds), "experts": self.patches,
                "group_sizes": torch.bincount(self.groups.long()).cpu().tolist(),
                "aggregation": "two-layer shared-prior NK; unconstrained coefficients"}
