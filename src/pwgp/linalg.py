"""Differentiable solves using explicit triangular factors."""

import torch


def cholesky_solve(rhs: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    """Solve (L L^T) X = B without the batched POTRS kernel."""
    intermediate = torch.linalg.solve_triangular(factor, rhs, upper=False)
    return torch.linalg.solve_triangular(factor.transpose(-2, -1), intermediate, upper=True)
