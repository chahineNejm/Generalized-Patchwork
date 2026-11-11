"""Constrained trainable parameters and ARD hyperparameters."""

from __future__ import annotations

import torch
from torch import nn

from pwgp.config import DEFAULT_HYPERPARAMETER_BOUNDS, HyperparameterBounds


def inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Parameter values must be finite")
    if bool((value <= 0).any()):
        raise ValueError("inverse_softplus requires strictly positive values")
    return value + torch.log(-torch.expm1(-value))


def inverse_bounded_sigmoid(
    value: torch.Tensor | float,
    lower: float,
    upper: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=dtype, device=device)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Parameter values must be finite")
    if bool(((value <= lower) | (value >= upper)).any()):
        raise ValueError(f"Value must be strictly inside ({lower}, {upper})")
    probability = (value - lower) / (upper - lower)
    return torch.log(probability) - torch.log1p(-probability)


def bounded_sigmoid(raw: torch.Tensor, bounds: tuple[float, float]) -> torch.Tensor:
    lower, upper = bounds
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("Raw hyperparameters must be finite")
    return lower + (upper - lower) * torch.sigmoid(raw)


class UnitCubeParameter(nn.Module):
    """A point tensor constrained to the open unit cube."""

    def __init__(self, initial: torch.Tensor, *, trainable: bool = True) -> None:
        super().__init__()
        if not bool(torch.isfinite(initial).all()):
            raise ValueError("Initial point coordinates must be finite")
        if bool(((initial <= 0) | (initial >= 1)).any()):
            raise ValueError("Trainable points must lie strictly inside the unit cube")
        raw = torch.log(initial) - torch.log1p(-initial)
        if trainable:
            self.raw = nn.Parameter(raw.clone())
        else:
            self.register_buffer("raw", raw.clone())
        self.trainable = trainable

    def forward(self) -> torch.Tensor:
        if not bool(torch.isfinite(self.raw).all()):
            raise FloatingPointError("Raw point coordinates must be finite")
        return torch.sigmoid(self.raw)


class GlobalARDHyperparameters(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        lengthscale: float | torch.Tensor = 0.2,
        outputscale: float = 1.0,
        noise_std: float = 0.05,
        bounds: HyperparameterBounds = DEFAULT_HYPERPARAMETER_BOUNDS,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        device = device or torch.device("cpu")
        ls = torch.as_tensor(lengthscale, dtype=dtype, device=device).reshape(-1)
        if ls.numel() == 1:
            ls = ls.repeat(input_dim)
        if ls.numel() != input_dim:
            raise ValueError(f"Expected {input_dim} lengthscales, received {ls.numel()}")
        self.bounds = bounds
        self.raw_lengthscales = nn.Parameter(
            inverse_bounded_sigmoid(ls, *bounds.lengthscale, dtype=dtype, device=device),
            requires_grad=trainable,
        )
        self.raw_outputscale = nn.Parameter(
            inverse_bounded_sigmoid(
                outputscale, *bounds.outputscale, dtype=dtype, device=device
            ),
            requires_grad=trainable,
        )
        self.raw_noise = nn.Parameter(
            inverse_bounded_sigmoid(noise_std, *bounds.noise_std, dtype=dtype, device=device),
            requires_grad=trainable,
        )

    def lengthscales(self) -> torch.Tensor:
        return bounded_sigmoid(self.raw_lengthscales, self.bounds.lengthscale)

    def outputscales(self) -> torch.Tensor:
        return bounded_sigmoid(self.raw_outputscale, self.bounds.outputscale)

    def noise_std(self) -> torch.Tensor:
        return bounded_sigmoid(self.raw_noise, self.bounds.noise_std)


class LocalARDHyperparameters(nn.Module):
    def __init__(
        self,
        patches: int,
        input_dim: int,
        *,
        lengthscale: float | torch.Tensor = 0.2,
        outputscale: float | torch.Tensor = 1.0,
        noise_std: float = 0.05,
        bounds: HyperparameterBounds = DEFAULT_HYPERPARAMETER_BOUNDS,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        device = device or torch.device("cpu")
        ls = torch.as_tensor(lengthscale, dtype=dtype, device=device)
        if ls.ndim == 0:
            ls = ls.repeat(patches, input_dim)
        elif ls.ndim == 1 and ls.numel() in (1, input_dim):
            ls = ls.reshape(1, -1).repeat(patches, input_dim if ls.numel() == 1 else 1)
        if tuple(ls.shape) != (patches, input_dim):
            raise ValueError(f"Expected lengthscale shape {(patches, input_dim)}, got {ls.shape}")
        os = torch.as_tensor(outputscale, dtype=dtype, device=device)
        if os.ndim == 0:
            os = os.repeat(patches)
        if tuple(os.shape) != (patches,):
            raise ValueError(f"Expected outputscale shape {(patches,)}, got {os.shape}")
        self.bounds = bounds
        self.raw_lengthscales = nn.Parameter(
            inverse_bounded_sigmoid(ls, *bounds.lengthscale, dtype=dtype, device=device),
            requires_grad=trainable,
        )
        self.raw_outputscales = nn.Parameter(
            inverse_bounded_sigmoid(os, *bounds.outputscale, dtype=dtype, device=device),
            requires_grad=trainable,
        )
        self.raw_noise = nn.Parameter(
            inverse_bounded_sigmoid(noise_std, *bounds.noise_std, dtype=dtype, device=device),
            requires_grad=trainable,
        )

    def lengthscales(self) -> torch.Tensor:
        return bounded_sigmoid(self.raw_lengthscales, self.bounds.lengthscale)

    def outputscales(self) -> torch.Tensor:
        return bounded_sigmoid(self.raw_outputscales, self.bounds.outputscale)

    def noise_std(self) -> torch.Tensor:
        return bounded_sigmoid(self.raw_noise, self.bounds.noise_std)

