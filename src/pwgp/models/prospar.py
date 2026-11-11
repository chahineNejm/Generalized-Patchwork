"""ProSpar-GP from the generative model in Li and Mak, arXiv:2311.08752v1."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional

from pwgp.config import (
    DEFAULT_HYPERPARAMETER_BOUNDS,
    DEFAULT_NUMERICAL_CONFIG,
    HyperparameterBounds,
    NumericalConfig,
)
from pwgp.kernels import ard_rbf
from pwgp.linalg import cholesky_solve
from pwgp.parameters import LocalARDHyperparameters, UnitCubeParameter, inverse_softplus
from pwgp.variational import DenseGaussian


def _positive_raw(value, shape, reference, name):
    value = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if value.ndim == 0:
        value = value.expand(shape).clone()
    if value.shape != shape or not bool(torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise ValueError(f"{name} must be finite, positive and scalar or shape {shape}")
    return nn.Parameter(inverse_softplus(value))


def _positive(raw, name):
    value = functional.softplus(raw)
    if not bool(torch.isfinite(raw).all() & torch.isfinite(value).all()) or bool((value <= 0).any()):
        raise FloatingPointError(f"{name} must remain finite and positive")
    return value


class _ExpertSupports(nn.ModuleList):
    def __init__(self, supports, trainable):
        super().__init__([UnitCubeParameter(z, trainable=trainable) for z in supports])
        self.trainable = trainable

    def forward(self):
        return tuple(points() for points in self)


class _ProSparHyperparameters(LocalARDHyperparameters):
    def __init__(self, supports, temperatures, exponent, inducing_nuggets, **options):
        reference = supports[0]
        super().__init__(len(supports), reference.shape[1], dtype=reference.dtype,
                         device=reference.device, **options)
        self.raw_temperatures = _positive_raw(temperatures, (len(supports),), reference, "temperatures")
        self.raw_exponent = _positive_raw(exponent, (), reference, "exponent")
        if isinstance(inducing_nuggets, (int, float)) or (
            isinstance(inducing_nuggets, torch.Tensor) and inducing_nuggets.ndim == 0
        ):
            inducing_nuggets = [inducing_nuggets] * len(supports)
        if len(inducing_nuggets) != len(supports):
            raise ValueError("inducing_nuggets requires one scalar/vector per expert")
        self.raw_nuggets = nn.ParameterList([
            _positive_raw(value, (len(z),), reference, f"expert {j} nuggets")
            for j, (value, z) in enumerate(zip(inducing_nuggets, supports, strict=True))
        ])
        if not options["trainable"]:
            self.requires_grad_(False)

    def temperatures(self):
        return _positive(self.raw_temperatures, "temperatures")

    def exponent(self):
        return _positive(self.raw_exponent, "exponent")

    def nuggets(self):
        return tuple(_positive(value, "inducing nuggets") for value in self.raw_nuggets)


class ProSparGP(nn.Module):
    """Independent full-covariance q(U_j), with paper-derived fusion and ELBO.

    Required prior_jitter leaves the extra fixed-jitter decision to the caller.
    Kernel/noise bounds and variational Cholesky constraints reuse SVGP's settings.
    Temperatures, exponent and inducing nuggets have no initial-value defaults.
    """

    kind = "prospar"

    def __init__(
        self, supports: Sequence[torch.Tensor] | torch.Tensor, *,
        temperatures: float | torch.Tensor, exponent: float,
        inducing_nuggets: float | Sequence[torch.Tensor], prior_jitter: float,
        train_supports: bool = True, train_hyperparameters: bool = True,
        lengthscale: float | torch.Tensor = 0.2, outputscale: float | torch.Tensor = 1.0,
        noise_std: float = 0.05, numerical: NumericalConfig = DEFAULT_NUMERICAL_CONFIG,
        bounds: HyperparameterBounds = DEFAULT_HYPERPARAMETER_BOUNDS,
    ) -> None:
        super().__init__()
        if isinstance(supports, torch.Tensor) and supports.ndim != 3:
            raise ValueError("Tensor supports must have shape [experts,points,dimensions]")
        supports = tuple(supports)
        if not supports:
            raise ValueError("ProSpar requires at least one expert")
        reference = supports[0]
        if reference.ndim != 2 or reference.shape[1] < 1:
            raise ValueError("Each expert requires supports[M_j,D]")
        if reference.dtype not in (torch.float32, torch.float64):
            raise ValueError("ProSpar supports must be float32 or float64")
        for z in supports:
            if z.ndim != 2 or len(z) < 1 or z.shape[1] != reference.shape[1]:
                raise ValueError("Experts require nonempty supports with the same input dimension")
            if z.dtype != reference.dtype or z.device != reference.device:
                raise ValueError("Expert supports must share dtype and device")
        if not math.isfinite(prior_jitter) or prior_jitter < 0:
            raise ValueError("prior_jitter must be explicitly finite and nonnegative")
        if not math.isfinite(numerical.variance_floor) or numerical.variance_floor < 0:
            raise ValueError("variance_floor must be finite and nonnegative")
        if not math.isfinite(numerical.min_variational_cholesky) or numerical.min_variational_cholesky <= 0:
            raise ValueError("min_variational_cholesky must be finite and positive")
        self.patches, self.input_dim = len(supports), reference.shape[1]
        self.inducing_counts = tuple(len(z) for z in supports)
        self.prior_jitter, self.numerical = prior_jitter, numerical
        self.support = _ExpertSupports(supports, train_supports)
        self.hyperparameters = _ProSparHyperparameters(
            supports, temperatures, exponent, inducing_nuggets, lengthscale=lengthscale,
            outputscale=outputscale, noise_std=noise_std, bounds=bounds,
            trainable=train_hyperparameters,
        )
        self.q = nn.ModuleList([
            DenseGaussian(len(z), min_cholesky=numerical.min_variational_cholesky,
                          dtype=z.dtype, device=z.device) for z in supports
        ])
        with torch.no_grad():
            for q, prior in zip(self.q, self.prior(), strict=True):
                q.initialize_from_prior(prior["cholesky"])

    @property
    def hyp(self):
        return self.hyperparameters

    def supports(self):
        return self.support()

    def prior(self):
        priors = []
        for j, (z, nugget) in enumerate(zip(self.supports(), self.hyp.nuggets(), strict=True)):
            covariance = ard_rbf(z, z, self.hyp.lengthscales()[j], self.hyp.outputscales()[j])
            covariance = covariance + torch.diag(nugget + self.prior_jitter)
            priors.append({"covariance": covariance, "cholesky": torch.linalg.cholesky(covariance)})
        return priors

    def conditional_terms(self, x, prior=None):
        if x.ndim != 2 or x.shape[1] != self.input_dim or len(x) < 1:
            raise ValueError("ProSpar query inputs must be nonempty x[B,D]")
        reference = self.q[0].mean
        if x.dtype != reference.dtype or x.device != reference.device:
            raise ValueError("ProSpar queries must share the model dtype/device")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("ProSpar query inputs must be finite")
        if prior is None:
            prior = self.prior()
        if len(prior) != self.patches:
            raise ValueError("ProSpar requires one prior per expert")
        projections, residuals = [], []
        for j, (z, state) in enumerate(zip(self.supports(), prior, strict=True)):
            cross = ard_rbf(x, z, self.hyp.lengthscales()[j], self.hyp.outputscales()[j])
            projection = cholesky_solve(cross.T, state["cholesky"]).T
            projections.append(projection)
            residuals.append(self.hyp.outputscales()[j] - (cross * projection).sum(-1))
        local_residual = torch.stack(residuals, dim=1)
        if not bool(torch.isfinite(local_residual).all()) or bool((local_residual <= 0).any()):
            raise FloatingPointError("ProSpar local conditional residuals must be finite and positive")
        logits = -self.hyp.temperatures() * local_residual.pow(self.hyp.exponent())
        if not bool(torch.isfinite(logits).all()):
            raise FloatingPointError("ProSpar softmax logits must be finite")
        log_alpha = torch.log_softmax(logits, dim=1)
        log_precision = log_alpha - local_residual.log()
        residual = torch.exp(-torch.logsumexp(log_precision, dim=1))
        return {"projections": projections, "local_residual": local_residual,
                "alpha": log_alpha.exp(), "weights": torch.softmax(log_precision, dim=1),
                "residual": residual}

    def moments(self, x, prior=None):
        terms = self.conditional_terms(x, prior)
        local_mean, local_variance = [], []
        for projection, q in zip(terms["projections"], self.q, strict=True):
            local_mean.append(projection @ q.mean)
            local_variance.append((projection @ q.cholesky()).square().sum(-1))
        mean = (terms["weights"] * torch.stack(local_mean, dim=1)).sum(-1)
        propagated = (terms["weights"].square() * torch.stack(local_variance, dim=1)).sum(-1)
        variance = terms["residual"] + propagated
        if not bool(torch.isfinite(mean).all() & torch.isfinite(variance).all()):
            raise FloatingPointError("ProSpar predictive mean and variance must be finite")
        if bool((variance <= self.numerical.variance_floor).any()):
            raise FloatingPointError("ProSpar latent variance is below the configured validity threshold")
        return mean, variance, propagated, terms

    def latent(self, x, prior=None):
        mean, variance, _, _ = self.moments(x, prior)
        return mean, variance

    def kl(self, prior=None):
        if prior is None:
            prior = self.prior()
        return sum(q.kl_to_prior(state["cholesky"]) for q, state in zip(self.q, prior, strict=True))

    def elbo_batch(self, x, y, n_total):
        if y.shape != (len(x),) or y.dtype != x.dtype or y.device != x.device:
            raise ValueError("Expected ProSpar y[B] with the same dtype/device as x")
        if not bool(torch.isfinite(y).all()) or not isinstance(n_total, int) or n_total < len(x):
            raise ValueError("ProSpar requires finite targets and integer n_total >= batch size")
        prior = self.prior()
        mean, variance, propagated, terms = self.moments(x, prior)
        # E_q(U) log p(y|U), after integrating the conditional f residual.
        conditional = terms["residual"] + self.hyp.noise_std().square()
        expected = -0.5 * (torch.log(2 * math.pi * conditional)
                           + ((y - mean).square() + propagated) / conditional)
        kl = self.kl(prior)
        elbo = (n_total / len(x)) * expected.sum() - kl
        if not bool(torch.isfinite(elbo)):
            raise FloatingPointError("ProSpar ELBO must be finite")
        return elbo, mean, variance, kl

    @torch.no_grad()
    def predict(self, x, *, observation_noise=True):
        self.eval()
        mean, variance = self.latent(x)
        if observation_noise:
            variance = variance + self.hyp.noise_std().square()
        return mean, variance
