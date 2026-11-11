"""Two-regime visual diagnostic for SVGP and generalized Patchwork GP."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch

from pwgp import SVGP, GeneralizedPatchworkGP, fit
from pwgp.config import NumericalConfig, OptimizerConfig


@dataclass
class DynamicRegimeResult:
    figure: plt.Figure
    svgp: SVGP
    patchwork: GeneralizedPatchworkGP
    metrics: dict[str, float]


def _truth(x: torch.Tensor) -> torch.Tensor:
    left = 0.15 + 0.9 * (x - 0.5)
    right = 0.15 + 0.65 * torch.sin(28 * torch.pi * (x - 0.5)) + 0.15 * (x - 0.5)
    return torch.where(x < 0.5, left, right)


def run_dynamic_regime_test(
    *,
    epochs: int = 100,
    observations: int = 280,
    supports_per_patch: int = 24,
    q_rank: int = 12,
    warmup_fraction: float = 0.3,
    weight_epsilon: float = 1e-8,
    noise_std: float = 0.035,
    seed: int = 19,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> DynamicRegimeResult:
    """Fit a smooth-to-oscillatory curve and expose the learned GPW weights."""
    torch.manual_seed(seed)
    compute_device = torch.device(device)
    x = 0.02 + 0.96 * torch.rand(observations, 1, dtype=dtype)
    y = _truth(x[:, 0]) + noise_std * torch.randn(observations, dtype=dtype)
    order = torch.argsort(x[:, 0])
    x, y = x[order].to(compute_device), y[order].to(compute_device)

    total_supports = 2 * supports_per_patch
    svgp_initial = torch.linspace(0.03, 0.97, total_supports, dtype=dtype)[:, None]
    patchwork_initial = torch.stack(
        (
            torch.linspace(0.03, 0.47, supports_per_patch, dtype=dtype)[:, None],
            torch.linspace(0.53, 0.97, supports_per_patch, dtype=dtype)[:, None],
        )
    )
    stitch_initial = torch.tensor([[[0.5]]], dtype=dtype)
    svgp_initial = svgp_initial.to(compute_device)
    patchwork_initial = patchwork_initial.to(compute_device)
    stitch_initial = stitch_initial.to(compute_device)

    svgp = SVGP(
        svgp_initial.clone(),
        lengthscale=0.12,
        outputscale=1.0,
        noise_std=noise_std,
        train_supports=True,
        train_hyperparameters=True,
    ).to(compute_device)
    patchwork = GeneralizedPatchworkGP(
        patchwork_initial.clone(),
        stitch_initial.clone(),
        [(0, 1)],
        q_rank=q_rank,
        lengthscale=torch.tensor([[0.35], [0.025]], dtype=dtype, device=compute_device),
        outputscale=torch.tensor([1.0, 1.0], dtype=dtype, device=compute_device),
        noise_std=noise_std,
        train_supports=True,
        train_stitches=True,
        train_hyperparameters=True,
        numerical=NumericalConfig(weight_epsilon=weight_epsilon),
    ).to(compute_device)
    optimizer = OptimizerConfig(
        learning_rate_q=5e-3,
        learning_rate_supports=2e-3,
        learning_rate_stitches=2e-3,
        learning_rate_hyperparameters=5e-3,
        warmup_fraction=warmup_fraction,
        alternate_patchwork_updates=True,
        gradient_clip=10.0,
        weight_decay=0.0,
    )
    svgp_history = fit(
        svgp,
        x,
        y,
        epochs=epochs,
        batch_size=observations,
        seed=seed,
        optimizer=optimizer,
    )
    patchwork_history = fit(
        patchwork,
        x,
        y,
        epochs=epochs,
        batch_size=observations,
        seed=seed,
        optimizer=optimizer,
    )

    grid = torch.linspace(0.01, 0.99, 700, dtype=dtype, device=compute_device)[:, None]
    truth = _truth(grid[:, 0])
    svgp_mean, svgp_variance = svgp.predict(grid, observation_noise=False)
    pw_mean, pw_variance = patchwork.predict(grid, observation_noise=False)
    fusion = patchwork.fusion_weights(grid)
    left = grid[:, 0] < 0.5
    metrics = {
        "svgp_rmse": float(torch.mean((svgp_mean - truth).square()).sqrt().cpu()),
        "svgp_left_rmse": float(torch.mean((svgp_mean[left] - truth[left]).square()).sqrt().cpu()),
        "svgp_right_rmse": float(torch.mean((svgp_mean[~left] - truth[~left]).square()).sqrt().cpu()),
        "patchwork_rmse": float(torch.mean((pw_mean - truth).square()).sqrt().cpu()),
        "patchwork_left_rmse": float(torch.mean((pw_mean[left] - truth[left]).square()).sqrt().cpu()),
        "patchwork_right_rmse": float(torch.mean((pw_mean[~left] - truth[~left]).square()).sqrt().cpu()),
        "svgp_initial_loss": svgp_history.losses[0],
        "svgp_final_loss": svgp_history.losses[-1],
        "patchwork_initial_loss": patchwork_history.losses[0],
        "patchwork_final_loss": patchwork_history.losses[-1],
        "weight_min": float(fusion.min().detach().cpu()),
        "weight_max": float(fusion.max().detach().cpu()),
    }

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    views = (
        (axes[0, 0], "SVGP", svgp_mean, svgp_variance),
        (axes[0, 1], "Generalized Patchwork", pw_mean, pw_variance),
    )
    for axis, title, mean, variance in views:
        deviation = variance.sqrt()
        axis.scatter(x[:, 0].cpu(), y.cpu(), s=10, color="black", alpha=0.4, label="data")
        axis.plot(grid[:, 0].cpu(), truth.cpu(), color="black", linestyle="--", label="truth")
        axis.plot(grid[:, 0].cpu(), mean.cpu(), color="#146C94", label="mean")
        axis.fill_between(
            grid[:, 0].cpu(),
            (mean - 1.96 * deviation).cpu(),
            (mean + 1.96 * deviation).cpu(),
            color="#19A7CE",
            alpha=0.22,
            label="95% latent interval",
        )
        axis.axvline(0.5, color="#555555", linestyle=":", label="regime change")
        axis.set(title=title, xlabel="x", ylabel="y")
        axis.legend(fontsize=8)

    axes[1, 0].plot(svgp_history.losses, label="SVGP")
    axes[1, 0].plot(patchwork_history.losses, label="Patchwork")
    axes[1, 0].axvline(
        round(warmup_fraction * epochs), color="#555555", linestyle=":", label="warm-up end"
    )
    axes[1, 0].set(title="Optimization", xlabel="epoch", ylabel="negative ELBO / N")
    axes[1, 0].legend()

    fusion_cpu = fusion.detach().cpu()
    axes[1, 1].plot(grid[:, 0].cpu(), fusion_cpu[:, 0], label="local model 1")
    axes[1, 1].plot(grid[:, 0].cpu(), fusion_cpu[:, 1], label="local model 2")
    learned_stitch = float(patchwork.stitches().detach().cpu().reshape(-1)[0])
    axes[1, 1].axvline(learned_stitch, color="#C84B31", label="learned stitch")
    lengthscales = patchwork.hyp.lengthscales().detach().cpu().reshape(-1)
    axes[1, 1].set(
        title=f"GPW weights | learned lengthscales {lengthscales[0]:.3f}, {lengthscales[1]:.3f}",
        xlabel="x",
        ylabel="weight",
    )
    axes[1, 1].legend()
    return DynamicRegimeResult(figure, svgp, patchwork, metrics)


if __name__ == "__main__":
    result = run_dynamic_regime_test()
    print(result.metrics)
    plt.show()
