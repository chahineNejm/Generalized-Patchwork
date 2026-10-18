"""Numerical and optimization configuration objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NumericalConfig:
    prior_jitter: float = 1e-4
    min_variational_cholesky: float = 1e-3
    variance_floor: float = 1e-12


@dataclass(frozen=True)
class HyperparameterBounds:
    lengthscale: tuple[float, float] = (0.02, 3.0)
    outputscale: tuple[float, float] = (0.02, 30.0)
    noise_std: tuple[float, float] = (1e-4, 1.0)


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate_q: float = 2e-3
    learning_rate_supports: float = 2e-2
    learning_rate_stitches: float = 5e-2
    learning_rate_hyperparameters: float = 1e-2
    warmup_fraction: float = 0.5
    alternate_patchwork_updates: bool = True
    gradient_clip: float = 10.0
    weight_decay: float = 0.0


DEFAULT_NUMERICAL_CONFIG = NumericalConfig()
DEFAULT_HYPERPARAMETER_BOUNDS = HyperparameterBounds()
DEFAULT_OPTIMIZER_CONFIG = OptimizerConfig()

