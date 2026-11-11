"""Full-covariance stochastic variational Gaussian process."""

from __future__ import annotations

import math

import torch
from torch import nn

from pwgp.config import (
    DEFAULT_HYPERPARAMETER_BOUNDS,
    DEFAULT_NUMERICAL_CONFIG,
    HyperparameterBounds,
    NumericalConfig,
)
from pwgp.kernels import ard_rbf
from pwgp.linalg import cholesky_solve
from pwgp.parameters import GlobalARDHyperparameters, UnitCubeParameter
from pwgp.variational import DenseGaussian


class SVGP(nn.Module):
    kind = "svgp"

    def __init__(
        self,
        supports: torch.Tensor,
        *,
        train_supports: bool = True,
        train_hyperparameters: bool = True,
        lengthscale: float | torch.Tensor = 0.2,
        outputscale: float = 1.0,
        noise_std: float = 0.05,
        numerical: NumericalConfig = DEFAULT_NUMERICAL_CONFIG,
        bounds: HyperparameterBounds = DEFAULT_HYPERPARAMETER_BOUNDS,
    ) -> None:
        super().__init__()
        if supports.ndim != 2:
            raise ValueError("supports must have shape [M,D]")
        self.numerical = numerical
        self.support = UnitCubeParameter(supports, trainable=train_supports)
        self.hyperparameters = GlobalARDHyperparameters(
            supports.shape[-1],
            lengthscale=lengthscale,
            outputscale=outputscale,
            noise_std=noise_std,
            bounds=bounds,
            dtype=supports.dtype,
            device=supports.device,
            trainable=train_hyperparameters,
        )
        self.q = DenseGaussian(
            supports.shape[0],
            min_cholesky=numerical.min_variational_cholesky,
            dtype=supports.dtype,
            device=supports.device,
        )
        with torch.no_grad():
            self.q.initialize_from_prior(self.prior()["cholesky"])

    @property
    def hyp(self) -> GlobalARDHyperparameters:
        return self.hyperparameters

    def supports(self) -> torch.Tensor:
        return self.support()

    def prior(self) -> dict[str, torch.Tensor]:
        supports = self.supports()
        covariance = ard_rbf(
            supports,
            supports,
            self.hyp.lengthscales(),
            self.hyp.outputscales(),
        )
        identity = torch.eye(len(supports), dtype=supports.dtype, device=supports.device)
        covariance = covariance + self.numerical.prior_jitter * identity
        return {"covariance": covariance, "cholesky": torch.linalg.cholesky(covariance)}

    def latent(
        self, x: torch.Tensor, prior: dict[str, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior = prior or self.prior()
        supports = self.supports()
        cross_covariance = ard_rbf(x, supports, self.hyp.lengthscales(), self.hyp.outputscales())
        weights = cholesky_solve(cross_covariance.T, prior["cholesky"]).T
        mean = weights @ self.q.mean
        conditional_variance = self.hyp.outputscales() - (weights * cross_covariance).sum(-1)
        propagated = weights @ self.q.cholesky()
        variance = conditional_variance + propagated.square().sum(-1)
        if not bool(torch.isfinite(mean).all() & torch.isfinite(variance).all()):
            raise FloatingPointError("SVGP predictive mean and variance must be finite")
        if bool((variance <= self.numerical.variance_floor).any()):
            raise FloatingPointError("SVGP latent variance is below the configured floor")
        return mean, variance

    def kl(self, prior: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        prior = prior or self.prior()
        return self.q.kl_to_prior(prior["cholesky"])

    def elbo_batch(
        self, x: torch.Tensor, y: torch.Tensor, n_total: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prior = self.prior()
        mean, latent_variance = self.latent(x, prior)
        noise_variance = self.hyp.noise_std().square()
        expected_log_likelihood = -0.5 * (
            torch.log(2 * math.pi * noise_variance)
            + ((y - mean).square() + latent_variance) / noise_variance
        )
        kl = self.kl(prior)
        elbo = (n_total / x.shape[0]) * expected_log_likelihood.sum() - kl
        return elbo, mean, latent_variance, kl

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, *, observation_noise: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        mean, variance = self.latent(x)
        if observation_noise:
            variance = variance + self.hyp.noise_std().square()
        return mean, variance
