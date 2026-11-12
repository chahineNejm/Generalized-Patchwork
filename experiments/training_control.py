"""Explicit training-loss stopping; never restore or substitute model states."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyStoppingConfig:
    patience: int
    relative_min_improvement: float

    def __post_init__(self):
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("Early-stopping patience must be a positive integer")
        if not math.isfinite(self.relative_min_improvement) or self.relative_min_improvement < 0:
            raise ValueError("Early-stopping relative improvement must be finite and nonnegative")


class TrainingLossStopper:
    def __init__(self, config, *, warmup_epochs):
        if type(warmup_epochs) is not int or warmup_epochs < 0:
            raise ValueError("Early-stopping warm-up must be a nonnegative integer")
        self.config, self.warmup_epochs = config, warmup_epochs
        self.best_loss, self.best_epoch = None, None
        self.bad_epochs, self.last_epoch, self.stopped = 0, 0, False

    def update(self, epoch, loss):
        if epoch != self.last_epoch + 1 or self.stopped:
            raise ValueError("Early-stopping updates must be consecutive and stop after triggering")
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss cannot be used for early stopping")
        self.last_epoch = epoch
        if epoch <= self.warmup_epochs:
            return False
        improvement = None if self.best_loss is None else self.best_loss - loss
        if self.best_loss is None or (improvement > 0 and
                improvement >= self.config.relative_min_improvement * abs(self.best_loss)):
            self.best_loss, self.best_epoch, self.bad_epochs = loss, epoch, 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.config.patience
        return self.stopped
