"""Structured Gaussian-process models for scalable regression."""

from pwgp.models.nested_kriging import NestedKriging
from pwgp.models.patchwork import GeneralizedPatchworkGP
from pwgp.models.prospar import ProSparGP
from pwgp.models.svgp import SVGP
from pwgp.training import FitHistory, fit, predict_batched

__all__ = [
    "SVGP",
    "FitHistory",
    "GeneralizedPatchworkGP",
    "NestedKriging",
    "ProSparGP",
    "fit",
    "predict_batched",
]

__version__ = "0.1.0"

