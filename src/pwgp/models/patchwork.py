"""Generalized stitched Patchwork GP with Nested-Kriging fusion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from pwgp.config import (
    DEFAULT_HYPERPARAMETER_BOUNDS,
    DEFAULT_NUMERICAL_CONFIG,
    HyperparameterBounds,
    NumericalConfig,
)
from pwgp.kernels import local_shared, local_xz, local_z_shared, local_zz
from pwgp.linalg import cholesky_solve
from pwgp.parameters import LocalARDHyperparameters, UnitCubeParameter
from pwgp.variational import BlockLowRankPrecisionGaussian


def stitch_coefficients(
    edges: Sequence[tuple[int, int]],
    stitches_per_edge: int,
    patches: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for left, right in edges:
        if left == right or min(left, right) < 0 or max(left, right) >= patches:
            raise ValueError(f"Invalid patch edge {(left, right)}")
        row = torch.zeros(patches, dtype=dtype, device=device)
        row[left], row[right] = 1.0, -1.0
        rows.extend(row.clone() for _ in range(stitches_per_edge))
    if not rows:
        return torch.zeros(0, patches, dtype=dtype, device=device)
    return torch.stack(rows)


def agreement_weights(
    local_residual: torch.Tensor,
    *,
    reference: int = 0,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Compute regularized unit-sum coefficients for local agreement."""
    patches = local_residual.shape[-1]
    if local_residual.shape[-2] != patches or not 0 <= reference < patches:
        raise ValueError("Invalid local covariance shape or reference index")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    identity = torch.eye(patches, dtype=local_residual.dtype, device=local_residual.device)
    if patches == 1:
        return torch.ones(
            *local_residual.shape[:-2],
            1,
            dtype=local_residual.dtype,
            device=local_residual.device,
        )
    others = [index for index in range(patches) if index != reference]
    contrasts = identity[reference][None, :] - identity[others]
    residual = 0.5 * (local_residual + local_residual.transpose(-2, -1))
    delta_covariance = contrasts @ residual @ contrasts.T
    delta_covariance = 0.5 * (delta_covariance + delta_covariance.transpose(-2, -1))
    delta_covariance = delta_covariance + epsilon * torch.eye(
        patches - 1,
        dtype=local_residual.dtype,
        device=local_residual.device,
    )
    rhs = contrasts @ residual[..., :, reference, None]
    cholesky = torch.linalg.cholesky(delta_covariance)
    coefficients = cholesky_solve(rhs, cholesky).squeeze(-1)
    return identity[reference] - coefficients @ contrasts


class GeneralizedPatchworkGP(nn.Module):
    """Stitched predictors with agreement conditional on inducing values."""

    kind = "patchwork"

    def __init__(
        self,
        supports: torch.Tensor,
        stitch_points: torch.Tensor,
        edges: Sequence[tuple[int, int]],
        *,
        q_rank: int = 8,
        train_supports: bool = True,
        train_stitches: bool = True,
        train_hyperparameters: bool = True,
        lengthscale: float | torch.Tensor = 0.2,
        outputscale: float | torch.Tensor = 1.0,
        noise_std: float = 0.05,
        numerical: NumericalConfig = DEFAULT_NUMERICAL_CONFIG,
        bounds: HyperparameterBounds = DEFAULT_HYPERPARAMETER_BOUNDS,
    ) -> None:
        super().__init__()
        if supports.ndim != 3:
            raise ValueError("supports must have shape [patches,supports_per_patch,input_dim]")
        self.patches, self.supports_per_patch, input_dim = supports.shape
        self.edges = list(edges)
        expected = (len(self.edges), stitch_points.shape[1], input_dim)
        if stitch_points.ndim != 3 or tuple(stitch_points.shape) != expected:
            raise ValueError(f"stitch_points must have shape {expected}")
        self.stitches_per_edge = stitch_points.shape[1]
        self.numerical = numerical
        self.support = UnitCubeParameter(supports, trainable=train_supports)
        self.stitch = UnitCubeParameter(
            stitch_points,
            trainable=train_stitches and stitch_points.numel() > 0,
        )
        self.hyperparameters = LocalARDHyperparameters(
            self.patches,
            input_dim,
            lengthscale=lengthscale,
            outputscale=outputscale,
            noise_std=noise_std,
            bounds=bounds,
            dtype=supports.dtype,
            device=supports.device,
            trainable=train_hyperparameters,
        )
        self.register_buffer(
            "stitch_coefficients",
            stitch_coefficients(
                self.edges,
                self.stitches_per_edge,
                self.patches,
                dtype=supports.dtype,
                device=supports.device,
            ),
        )
        self.q = BlockLowRankPrecisionGaussian(
            self.patches,
            self.supports_per_patch,
            q_rank,
            min_cholesky=numerical.min_variational_cholesky,
            dtype=supports.dtype,
            device=supports.device,
        )
        with torch.no_grad():
            self.q.initialize_from_prior(self.prior())

    @property
    def hyp(self) -> LocalARDHyperparameters:
        return self.hyperparameters

    def supports(self) -> torch.Tensor:
        return self.support()

    def stitches(self) -> torch.Tensor:
        return self.stitch()

    def prior(self) -> dict[str, Any]:
        supports = self.supports()
        lengthscales = self.hyp.lengthscales()
        outputscales = self.hyp.outputscales()
        stitches = self.stitches().reshape(-1, supports.shape[-1])
        coefficients = self.stitch_coefficients
        constraints = len(stitches)
        local_identity = torch.eye(
            self.supports_per_patch, dtype=supports.dtype, device=supports.device
        )
        local_covariance = local_zz(supports, lengthscales, outputscales)
        local_covariance = local_covariance + self.numerical.prior_jitter * local_identity[None]
        local_cholesky = torch.linalg.cholesky(local_covariance)

        if constraints:
            support_constraint = (
                local_z_shared(supports, stitches, lengthscales, outputscales)
                * coefficients.T[:, None, :]
            )
            stitch_covariances = local_shared(stitches, stitches, lengthscales, outputscales)
            transposed = coefficients.T
            constraint_covariance = (
                stitch_covariances * transposed[:, :, None] * transposed[:, None, :]
            ).sum(0)
            constraint_identity = torch.eye(
                constraints, dtype=supports.dtype, device=supports.device
            )
            constraint_covariance = (
                0.5 * (constraint_covariance + constraint_covariance.T)
                + self.numerical.prior_jitter * constraint_identity
            )
            raw_constraint_cholesky = torch.linalg.cholesky(constraint_covariance)
            w = cholesky_solve(support_constraint, local_cholesky)
            schur = constraint_covariance - torch.einsum("pmq,pmr->qr", support_constraint, w)
            schur = 0.5 * (schur + schur.T)
            constraint_cholesky = torch.linalg.cholesky(schur)
            logdet = (
                2 * torch.log(torch.diagonal(local_cholesky, dim1=-2, dim2=-1)).sum()
                + 2 * torch.log(torch.diagonal(constraint_cholesky)).sum()
                - 2 * torch.log(torch.diagonal(raw_constraint_cholesky)).sum()
            )
        else:
            support_constraint = supports.new_zeros(self.patches, self.supports_per_patch, 0)
            w = support_constraint
            constraint_covariance = supports.new_zeros(0, 0)
            raw_constraint_cholesky = supports.new_zeros(0, 0)
            constraint_cholesky = supports.new_zeros(0, 0)
            logdet = 2 * torch.log(torch.diagonal(local_cholesky, dim1=-2, dim2=-1)).sum()
        return {
            "supports": supports,
            "stitches": stitches,
            "coefficients": coefficients,
            "lengthscales": lengthscales,
            "outputscales": outputscales,
            "local_covariance": local_covariance,
            "local_cholesky": local_cholesky,
            "support_constraint": support_constraint,
            "constraint_covariance": constraint_covariance,
            "raw_constraint_cholesky": raw_constraint_cholesky,
            "w": w,
            "constraint_cholesky": constraint_cholesky,
            "logdet": logdet,
        }

    def local_query_terms(self, x: torch.Tensor, prior: dict[str, Any]) -> dict[str, Any]:
        phi = local_xz(x, prior["supports"], prior["lengthscales"], prior["outputscales"])
        phi0 = cholesky_solve(phi.permute(1, 2, 0), prior["local_cholesky"]).permute(2, 0, 1)
        constraints = prior["constraint_cholesky"].shape[0]
        if constraints:
            query_stitch = local_shared(
                x, prior["stitches"], prior["lengthscales"], prior["outputscales"]
            ).permute(1, 0, 2)
            h = query_stitch * prior["coefficients"].T[None, :, :]
            residual_cross = h - torch.einsum("bpm,pmq->bpq", phi0, prior["support_constraint"])
            rhs = residual_cross.reshape(-1, constraints).T
            solved_cross = cholesky_solve(rhs, prior["constraint_cholesky"]).T.reshape_as(
                residual_cross
            )
        else:
            h = x.new_zeros(x.shape[0], self.patches, 0)
            residual_cross = h
            solved_cross = h
        independent_variance = prior["outputscales"][None, :] - (phi * phi0).sum(-1)
        residual = torch.diag_embed(independent_variance) - (
            residual_cross @ solved_cross.transpose(-2, -1)
        )
        residual = 0.5 * (residual + residual.transpose(-2, -1))
        return {
            "phi": phi,
            "phi0": phi0,
            "h": h,
            "residual_cross": residual_cross,
            "solved_cross": solved_cross,
            "residual": residual,
        }

    def local_prediction_terms(
        self, x: torch.Tensor, prior: dict[str, Any], q_state: dict[str, torch.Tensor]
    ) -> dict[str, Any]:
        """Compute A S A^T + R without materializing the full local query map A."""
        terms = self.local_query_terms(x, prior)
        local_phi = torch.einsum("bpm,pmn->bpn", terms["phi0"], q_state["local"])
        local_w = q_state["local"].transpose(-2, -1) @ prior["w"]
        cross = torch.einsum("bpm,pmq->bpq", local_phi, local_w)
        h = terms["solved_cross"]
        mixed = cross @ h.transpose(-2, -1)
        middle = torch.einsum("pmq,pmr->qr", local_w, local_w)
        covariance = torch.diag_embed(local_phi.square().sum(-1))
        covariance = covariance - mixed - mixed.transpose(-2, -1) + (h @ middle) @ h.transpose(-2, -1)
        correction = torch.einsum("bpm,pmr->bpr", terms["phi0"], q_state["correction"])
        w_correction = torch.einsum("pmq,pmr->qr", prior["w"], q_state["correction"])
        correction = correction - h @ w_correction
        covariance = covariance - correction @ correction.transpose(-2, -1)
        covariance = 0.5 * (covariance + covariance.transpose(-2, -1))
        terms["inducing_covariance"] = covariance
        terms["predictive_covariance"] = covariance + terms["residual"]
        return terms

    def fusion_weights(self, x: torch.Tensor, prior: dict[str, Any] | None = None) -> torch.Tensor:
        prior = prior or self.prior()
        terms = self.local_query_terms(x, prior)
        return agreement_weights(terms["residual"], epsilon=self.numerical.weight_epsilon)

    def regression_weights(
        self,
        x: torch.Tensor,
        prior: dict[str, Any],
        *,
        return_fusion: bool = False,
        q_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        terms = self.local_query_terms(x, prior)
        fusion = agreement_weights(terms["residual"], epsilon=self.numerical.weight_epsilon)
        projected = torch.einsum("bp,bpq->bq", fusion, terms["solved_cross"])
        weights = fusion[:, :, None] * terms["phi0"] - torch.einsum(
            "bq,pmq->bpm", projected, prior["w"]
        )
        return (weights, fusion) if return_fusion else weights

    def latent(
        self,
        x: torch.Tensor,
        prior: dict[str, Any] | None = None,
        q_state: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prior = prior or self.prior()
        q_state = q_state or self.q.state()
        terms = self.local_query_terms(x, prior)
        fusion = agreement_weights(terms["residual"], epsilon=self.numerical.weight_epsilon)
        projected = torch.einsum("bp,bpq->bq", fusion, terms["solved_cross"])
        weights = fusion[:, :, None] * terms["phi0"] - torch.einsum(
            "bq,pmq->bpm", projected, prior["w"]
        )
        mean = self.q.mean_from_weights(weights)
        if not bool(torch.isfinite(mean).all()):
            raise FloatingPointError("Patchwork latent mean must be finite")
        residual = torch.einsum("bi,bij,bj->b", fusion, terms["residual"], fusion)
        variance = self.q.variance_from_weights(weights, q_state) + residual
        if not bool(torch.isfinite(variance).all()) or bool((variance <= self.numerical.variance_floor).any()):
            raise FloatingPointError("Patchwork latent variance is below the configured floor")
        return mean, variance

    def kl(
        self,
        prior: dict[str, Any] | None = None,
        q_state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        prior = prior or self.prior()
        return self.q.kl(prior, q_state)

    def elbo_batch(
        self, x: torch.Tensor, y: torch.Tensor, n_total: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prior = self.prior()
        q_state = self.q.state()
        mean, latent_variance = self.latent(x, prior, q_state)
        noise_variance = self.hyp.noise_std().square()
        expected_log_likelihood = -0.5 * (
            torch.log(2 * math.pi * noise_variance)
            + ((y - mean).square() + latent_variance) / noise_variance
        )
        kl = self.kl(prior, q_state)
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
