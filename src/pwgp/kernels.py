"""ARD radial-basis-function kernels used by both model families."""

from __future__ import annotations

import torch


def ard_rbf(
    x: torch.Tensor,
    z: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
) -> torch.Tensor:
    """Return an ARD RBF covariance matrix for two point sets."""
    lengthscale = torch.as_tensor(lengthscale, dtype=x.dtype, device=x.device)
    outputscale = torch.as_tensor(outputscale, dtype=x.dtype, device=x.device)
    delta = (x[:, None, :] - z[None, :, :]) / lengthscale
    distance2 = delta.square().sum(-1)
    return outputscale * torch.exp(-0.5 * distance2)


def local_xz(
    x: torch.Tensor,
    supports: torch.Tensor,
    lengthscales: torch.Tensor,
    outputscales: torch.Tensor,
) -> torch.Tensor:
    """Evaluate every local kernel: [B,D] x [P,M,D] -> [B,P,M]."""
    delta = (
        x[:, None, None, :] - supports[None, :, :, :]
    ) / lengthscales[None, :, None, :]
    return outputscales[None, :, None] * torch.exp(-0.5 * delta.square().sum(-1))


def local_zz(
    supports: torch.Tensor,
    lengthscales: torch.Tensor,
    outputscales: torch.Tensor,
) -> torch.Tensor:
    """Evaluate within-patch covariance blocks [P,M,M]."""
    delta = (
        supports[:, :, None, :] - supports[:, None, :, :]
    ) / lengthscales[:, None, None, :]
    return outputscales[:, None, None] * torch.exp(-0.5 * delta.square().sum(-1))


def local_shared(
    a: torch.Tensor,
    b: torch.Tensor,
    lengthscales: torch.Tensor,
    outputscales: torch.Tensor,
) -> torch.Tensor:
    """Evaluate each local kernel between shared point sets [P,|A|,|B|]."""
    delta = (
        a[None, :, None, :] - b[None, None, :, :]
    ) / lengthscales[:, None, None, :]
    return outputscales[:, None, None] * torch.exp(-0.5 * delta.square().sum(-1))


def local_z_shared(
    supports: torch.Tensor,
    points: torch.Tensor,
    lengthscales: torch.Tensor,
    outputscales: torch.Tensor,
) -> torch.Tensor:
    """Evaluate local support-to-shared-point covariances [P,M,Q]."""
    delta = (
        supports[:, :, None, :] - points[None, None, :, :]
    ) / lengthscales[:, None, None, :]
    return outputscales[:, None, None] * torch.exp(-0.5 * delta.square().sum(-1))

