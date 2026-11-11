"""SVGP diagnostic on the smooth-to-rapid reference curve."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pwgp import SVGP, fit
from pwgp.config import OptimizerConfig


@dataclass
class ReferenceLikeSVGPResult:
    figure: plt.Figure
    model: SVGP
    metrics: dict[str, float]


def _truth(x: torch.Tensor) -> torch.Tensor:
    display_x = 2.0 * x
    smooth = torch.cos(2.0 * torch.pi * display_x)
    rapid = torch.cos(20.0 * torch.pi * (display_x - 1.0))
    return torch.where(display_x < 1.0, smooth, rapid)


def run_reference_like_svgp_test(
    *,
    epochs: int = 500,
    warmup_epochs: int = 30,
    observations: int = 360,
    supports: int = 72,
    noise_std: float = 0.03,
    seed: int = 31,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> ReferenceLikeSVGPResult:
    """Train a budget-matched global SVGP on the reference curve."""
    torch.manual_seed(seed)
    compute_device = torch.device(device)
    x = 0.01 + 0.98 * torch.rand(observations, 1, dtype=dtype)
    y = _truth(x[:, 0]) + noise_std * torch.randn(observations, dtype=dtype)
    order = torch.argsort(x[:, 0])
    x, y = x[order].to(compute_device), y[order].to(compute_device)

    initial_supports = torch.linspace(0.015, 0.985, supports, dtype=dtype)[:, None]
    model = SVGP(
        initial_supports.to(compute_device),
        lengthscale=0.05,
        outputscale=1.0,
        noise_std=noise_std,
        train_supports=True,
        train_hyperparameters=True,
    ).to(compute_device)
    optimizer = OptimizerConfig(
        learning_rate_q=5e-3,
        learning_rate_supports=2e-3,
        learning_rate_stitches=2e-3,
        learning_rate_hyperparameters=5e-3,
        warmup_fraction=warmup_epochs / epochs,
        alternate_patchwork_updates=True,
        gradient_clip=10.0,
        weight_decay=0.0,
    )
    history = fit(
        model,
        x,
        y,
        epochs=epochs,
        batch_size=observations,
        seed=seed,
        optimizer=optimizer,
    )

    grid = torch.linspace(0.005, 0.995, 1200, dtype=dtype, device=compute_device)[:, None]
    truth = _truth(grid[:, 0])
    mean, variance = model.predict(grid, observation_noise=False)
    display_grid = 2.0 * grid[:, 0]
    display_x = 2.0 * x[:, 0]
    learned_supports = 2.0 * model.supports().detach().cpu().squeeze(-1)
    split = display_grid < 1.0
    metrics = {
        "rmse": float(torch.mean((mean - truth).square()).sqrt().cpu()),
        "left_rmse": float(torch.mean((mean[split] - truth[split]).square()).sqrt().cpu()),
        "right_rmse": float(torch.mean((mean[~split] - truth[~split]).square()).sqrt().cpu()),
        "final_loss": history.losses[-1],
        "lengthscale": float(model.hyp.lengthscales().detach().cpu().reshape(-1)[0]),
    }

    figure, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    deviation = variance.sqrt()
    axes[0, 0].scatter(display_x.cpu(), y.cpu(), s=9, color="#2B8C57", alpha=0.45, label="data")
    axes[0, 0].plot(display_grid.cpu(), truth.cpu(), color="#146C94", label="true function")
    axes[0, 0].plot(display_grid.cpu(), mean.cpu(), color="#F28E2B", label="SVGP mean")
    axes[0, 0].fill_between(
        display_grid.cpu(),
        (mean - 1.96 * deviation).cpu(),
        (mean + 1.96 * deviation).cpu(),
        color="#F28E2B",
        alpha=0.18,
        label="95% latent interval",
    )
    axes[0, 0].axvline(1.0, color="#555555", linestyle=":")
    axes[0, 0].set(title="Global SVGP", xlabel="x", ylabel="f(x)")
    axes[0, 0].legend(fontsize=8, ncol=2)

    axes[0, 1].plot(display_grid.cpu(), deviation.cpu(), color="#7A5195")
    axes[0, 1].axvline(1.0, color="#555555", linestyle=":", label="regime change")
    axes[0, 1].set(title="Latent standard deviation", xlabel="x", ylabel="standard deviation")
    axes[0, 1].legend()

    axes[1, 0].plot(history.losses, color="#146C94")
    axes[1, 0].axvline(warmup_epochs, color="#555555", linestyle=":", label="warm-up end")
    axes[1, 0].set(title="Optimization", xlabel="epoch", ylabel="negative ELBO / N")
    axes[1, 0].legend()

    axes[1, 1].scatter(
        learned_supports,
        torch.zeros(supports),
        marker="|",
        s=260,
        linewidths=1.8,
        color="#146C94",
        label="inducing points",
    )
    axes[1, 1].axvline(1.0, color="#555555", linestyle=":", label="regime change")
    axes[1, 1].set(
        title="Learned inducing positions",
        xlabel="x",
        xlim=(0.0, 2.0),
        ylim=(-0.6, 0.6),
        yticks=[],
    )
    axes[1, 1].legend()
    return ReferenceLikeSVGPResult(figure, model, metrics)


if __name__ == "__main__":
    result = run_reference_like_svgp_test()
    output = Path("results/reference_like_svgp_test.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.figure.savefig(output, dpi=180)
    print(result.metrics)
    print(output.resolve())
