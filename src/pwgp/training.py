"""Shared minibatch optimization and batched prediction."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from pwgp.config import DEFAULT_OPTIMIZER_CONFIG, OptimizerConfig


@dataclass
class FitHistory:
    losses: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    peak_cuda_megabytes: float | None = None


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def fit(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    seed: int = 0,
    optimizer: OptimizerConfig = DEFAULT_OPTIMIZER_CONFIG,
    on_epoch: Callable[[int, torch.nn.Module, FitHistory], None] | None = None,
) -> FitHistory:
    """Optimize the training objective, alternating Patchwork q and structure updates."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("Expected x [N,D] and y [N]")
    device = _device_of(model)
    q_parameters = list(model.q.parameters())
    structure_groups: list[dict[str, object]] = []
    if model.support.trainable:
        structure_groups.append(
            {"params": list(model.support.parameters()), "lr": optimizer.learning_rate_supports}
        )
    if hasattr(model, "stitch") and model.stitch.trainable:
        structure_groups.append(
            {"params": [model.stitch.raw], "lr": optimizer.learning_rate_stitches}
        )
    hyperparameters = [parameter for parameter in model.hyp.parameters() if parameter.requires_grad]
    if not 0.0 <= optimizer.warmup_fraction <= 1.0:
        raise ValueError("warmup_fraction must lie between zero and one")
    warmup_epochs = (
        0 if optimizer.warmup_fraction == 0.0 else max(1, round(optimizer.warmup_fraction * epochs))
    )
    groups: list[dict[str, object]] = [
        {
            "params": q_parameters,
            "lr": optimizer.learning_rate_q,
            "weight_decay": optimizer.weight_decay,
            "name": "q",
        }
    ]
    groups.extend({**group, "name": "structure"} for group in structure_groups)
    if hyperparameters:
        groups.append(
            {
                "params": hyperparameters,
                "lr": 0.0 if warmup_epochs else optimizer.learning_rate_hyperparameters,
                "name": "hyperparameters",
            }
        )
    adam = torch.optim.Adam(groups)
    structure_parameters = [
        parameter
        for group in structure_groups
        for parameter in group["params"]  # type: ignore[index]
    ] + hyperparameters
    alternating = (
        model.kind == "patchwork"
        and optimizer.alternate_patchwork_updates
        and bool(structure_parameters)
    )
    phases = ("q", "structure") if alternating else ("joint",)
    all_parameters = q_parameters + structure_parameters
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    history = FitHistory()
    started = time.perf_counter()
    progress = tqdm(range(epochs), desc=f"train:{model.kind}", unit="epoch")
    for epoch in progress:
        if epoch == warmup_epochs:
            for group in adam.param_groups:
                if group.get("name") == "hyperparameters":
                    group["lr"] = optimizer.learning_rate_hyperparameters
        model.train()
        accumulated = 0.0
        seen = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=device.type == "cuda")
            batch_y = batch_y.to(device, non_blocking=device.type == "cuda")
            for phase in phases:
                if alternating:
                    for parameter in q_parameters:
                        parameter.requires_grad_(phase == "q")
                    for parameter in structure_parameters:
                        parameter.requires_grad_(phase == "structure")
                adam.zero_grad(set_to_none=True)
                elbo, _, _, kl = model.elbo_batch(batch_x, batch_y, len(x))
                loss = -elbo / len(x)
                if not torch.isfinite(loss) or not torch.isfinite(kl):
                    raise FloatingPointError(
                        f"Non-finite objective at epoch {epoch}, phase {phase}"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in all_parameters if parameter.requires_grad],
                    optimizer.gradient_clip,
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"Non-finite gradient at epoch {epoch}, phase {phase}")
                adam.step()
            accumulated += float(loss.detach()) * len(batch_x)
            seen += len(batch_x)
        history.losses.append(accumulated / seen)
        progress.set_postfix(loss=f"{history.losses[-1]:.5f}")
        history.elapsed_seconds = time.perf_counter() - started
        if on_epoch is not None:
            on_epoch(epoch + 1, model, history)
    for parameter in all_parameters:
        parameter.requires_grad_(True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        history.peak_cuda_megabytes = torch.cuda.max_memory_allocated(device) / 2**20
    history.elapsed_seconds = time.perf_counter() - started
    return history


@torch.no_grad()
def predict_batched(
    model: torch.nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int = 2048,
    observation_noise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict without moving the entire query set onto the model device."""
    model.eval()
    device = _device_of(model)
    means: list[torch.Tensor] = []
    variances: list[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        mean, variance = model.predict(
            x[start : start + batch_size].to(device),
            observation_noise=observation_noise,
        )
        means.append(mean.cpu())
        variances.append(variance.cpu())
    return torch.cat(means), torch.cat(variances)
