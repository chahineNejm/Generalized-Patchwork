"""Variational Gaussian families for SVGP and generalized Patchwork GP."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional

from pwgp.parameters import inverse_softplus


class DenseGaussian(nn.Module):
    """A full-covariance Gaussian q(U) = N(m, LL^T)."""

    def __init__(
        self,
        size: int,
        *,
        min_cholesky: float = 1e-3,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.size = int(size)
        self.min_cholesky = min_cholesky
        self.mean = nn.Parameter(torch.zeros(size, dtype=dtype, device=device))
        self.raw_cholesky = nn.Parameter(torch.zeros(size, size, dtype=dtype, device=device))

    def cholesky(self) -> torch.Tensor:
        lower = torch.tril(self.raw_cholesky, diagonal=-1)
        diagonal = functional.softplus(torch.diagonal(self.raw_cholesky))
        return lower + torch.diag(diagonal + self.min_cholesky)

    @torch.no_grad()
    def initialize_from_prior(self, prior_cholesky: torch.Tensor) -> None:
        self.mean.zero_()
        diagonal = torch.diagonal(prior_cholesky) - self.min_cholesky
        if bool((diagonal <= 0).any()):
            raise ValueError("Prior Cholesky diagonal is below min_cholesky")
        raw = torch.tril(prior_cholesky, diagonal=-1) + torch.diag(
            inverse_softplus(diagonal)
        )
        self.raw_cholesky.copy_(raw)

    def kl_to_prior(self, prior_cholesky: torch.Tensor) -> torch.Tensor:
        q_cholesky = self.cholesky()
        whitened_q = torch.linalg.solve_triangular(
            prior_cholesky, q_cholesky, upper=False
        )
        whitened_mean = torch.linalg.solve_triangular(
            prior_cholesky, self.mean[:, None], upper=False
        ).squeeze(-1)
        logdet_prior = 2 * torch.log(torch.diagonal(prior_cholesky)).sum()
        logdet_q = 2 * torch.log(torch.diagonal(q_cholesky)).sum()
        return 0.5 * (
            whitened_q.square().sum()
            + whitened_mean.square().sum()
            - self.size
            + logdet_prior
            - logdet_q
        )


class BlockLowRankPrecisionGaussian(nn.Module):
    """q(U)=N(m,(D+VV^T)^-1) with block-diagonal positive D."""

    def __init__(
        self,
        blocks: int,
        block_size: int,
        rank: int,
        *,
        min_cholesky: float = 1e-3,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.blocks = int(blocks)
        self.block_size = int(block_size)
        self.rank = int(rank)
        self.min_cholesky = min_cholesky
        total = self.blocks * self.block_size
        if not 0 <= self.rank <= total:
            raise ValueError("rank must lie between zero and the number of supports")
        self.mean = nn.Parameter(
            torch.zeros(blocks, block_size, dtype=dtype, device=device)
        )
        self.raw_inverse_d_cholesky = nn.Parameter(
            torch.zeros(blocks, block_size, block_size, dtype=dtype, device=device)
        )
        if rank:
            self.v = nn.Parameter(1e-3 * torch.randn(total, rank, dtype=dtype, device=device))
        else:
            self.register_buffer("v", torch.zeros(total, 0, dtype=dtype, device=device))

    def inverse_d_cholesky(self) -> torch.Tensor:
        lower = torch.tril(self.raw_inverse_d_cholesky, diagonal=-1)
        diagonal = functional.softplus(
            torch.diagonal(self.raw_inverse_d_cholesky, dim1=-2, dim2=-1)
        ) + self.min_cholesky
        return lower + torch.diag_embed(diagonal)

    @torch.no_grad()
    def initialize_from_prior(self, prior: dict[str, Any]) -> None:
        self.mean.zero_()
        local_cholesky = prior["local_cholesky"]
        diagonal = torch.diagonal(
            local_cholesky, dim1=-2, dim2=-1
        ) - self.min_cholesky
        if bool((diagonal <= 0).any()):
            raise ValueError("Local prior Cholesky diagonal is below min_cholesky")
        self.raw_inverse_d_cholesky.copy_(
            torch.tril(local_cholesky, diagonal=-1)
            + torch.diag_embed(inverse_softplus(diagonal))
        )
        constraints = prior["constraint_cholesky"].shape[0]
        if not self.rank or not constraints:
            return
        w = prior["w"].reshape(self.blocks * self.block_size, constraints)
        correction = torch.linalg.solve_triangular(
            prior["constraint_cholesky"], w.T, upper=False
        ).T
        if self.rank >= constraints:
            self.v[:, :constraints].copy_(correction)
        else:
            u, singular_values, _ = torch.linalg.svd(correction, full_matrices=False)
            self.v.copy_(u[:, : self.rank] * singular_values[: self.rank])

    def state(self) -> dict[str, torch.Tensor]:
        """Represent covariance as blockdiag(LL^T) - FF^T."""
        local = self.inverse_d_cholesky()
        logdet = 2 * torch.log(torch.diagonal(local, dim1=-2, dim2=-1)).sum()
        if not self.rank:
            return {
                "local": local,
                "correction": self.v.reshape(self.blocks, self.block_size, 0),
                "logdet": logdet,
            }
        v_blocks = self.v.reshape(self.blocks, self.block_size, self.rank)
        projected = (local.transpose(-2, -1) @ v_blocks).reshape(-1, self.rank)
        scale = projected.detach().abs().amax().clamp_min(1.0)
        identity = torch.eye(self.rank, dtype=local.dtype, device=local.device)
        _, triangular = torch.linalg.qr(
            torch.cat((identity / scale, projected / scale), dim=0), mode="reduced"
        )
        triangular = triangular * scale
        uncorrected = (local @ projected.reshape_as(v_blocks)).reshape(-1, self.rank)
        correction = torch.linalg.solve_triangular(
            triangular.T, uncorrected.T, upper=False
        ).T
        logdet = logdet - 2 * torch.log(torch.diagonal(triangular).abs()).sum()
        return {
            "local": local,
            "correction": correction.reshape(self.blocks, self.block_size, self.rank),
            "logdet": logdet,
        }

    def mean_from_weights(self, weights: torch.Tensor) -> torch.Tensor:
        return (weights * self.mean[None]).sum(dim=(1, 2))

    def variance_from_weights(
        self, weights: torch.Tensor, state: dict[str, torch.Tensor] | None = None
    ) -> torch.Tensor:
        state = state or self.state()
        weighted_local = torch.einsum("bpm,pmn->bpn", weights, state["local"])
        variance = weighted_local.square().sum(dim=(1, 2))
        if self.rank:
            weighted_correction = torch.einsum(
                "bpm,pmr->br", weights, state["correction"]
            )
            variance = variance - weighted_correction.square().sum(-1)
        return variance

    def kl(
        self,
        prior: dict[str, Any],
        state: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        state = state or self.state()
        local = state["local"]
        correction = state["correction"]
        prior_local = prior["local_cholesky"]
        w = prior["w"]
        constraint_cholesky = prior["constraint_cholesky"]
        constraints = constraint_cholesky.shape[0]
        total = self.blocks * self.block_size

        solved_local = torch.linalg.solve_triangular(prior_local, local, upper=False)
        trace = solved_local.square().sum()
        if constraints:
            w_local = torch.einsum("pmq,pmn->qpn", w, local).reshape(constraints, total)
            trace = trace + torch.linalg.solve_triangular(
                constraint_cholesky, w_local, upper=False
            ).square().sum()
        if self.rank:
            solved_correction = torch.linalg.solve_triangular(
                prior_local, correction, upper=False
            )
            trace = trace - solved_correction.square().sum()
            if constraints:
                w_correction = torch.einsum("pmq,pmr->qr", w, correction)
                trace = trace - torch.linalg.solve_triangular(
                    constraint_cholesky, w_correction, upper=False
                ).square().sum()

        solved_mean = torch.linalg.solve_triangular(
            prior_local, self.mean[..., None], upper=False
        )
        quadratic = solved_mean.square().sum()
        if constraints:
            w_mean = torch.einsum("pmq,pm->q", w, self.mean)
            quadratic = quadratic + torch.linalg.solve_triangular(
                constraint_cholesky, w_mean[:, None], upper=False
            ).square().sum()
        return 0.5 * (
            trace
            + quadratic
            - total
            + prior["logdet"]
            - state["logdet"]
        )

