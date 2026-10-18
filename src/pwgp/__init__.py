"""Structured Gaussian-process models for scalable regression."""

from pwgp.models.patchwork import GeneralizedPatchworkGP
from pwgp.models.svgp import SVGP
from pwgp.training import FitHistory, fit, predict_batched

__all__ = [
    "SVGP",
    "FitHistory",
    "GeneralizedPatchworkGP",
    "fit",
    "predict_batched",
]

__version__ = "0.1.0"

