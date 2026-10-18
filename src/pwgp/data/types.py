from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RegressionDataset:
    name: str
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    target_name: str
    source_files: tuple[Path, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.x.ndim != 2:
            raise ValueError("x must have shape [N,D]")
        if self.y.ndim != 1 or len(self.y) != len(self.x):
            raise ValueError("y must have shape [N]")
        if len(self.feature_names) != self.x.shape[1]:
            raise ValueError("feature_names must match the input dimension")

    def finite(self) -> RegressionDataset:
        valid = np.isfinite(self.y) & np.isfinite(self.x).all(axis=1)
        return RegressionDataset(
            name=self.name,
            x=self.x[valid],
            y=self.y[valid],
            feature_names=self.feature_names,
            target_name=self.target_name,
            source_files=self.source_files,
            metadata={**self.metadata, "removed_nonfinite": int((~valid).sum())},
        )

