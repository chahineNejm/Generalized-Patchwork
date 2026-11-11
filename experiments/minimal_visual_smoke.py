"""Small visual verification of SVGP and generalized Patchwork GP."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch

from pwgp import SVGP, GeneralizedPatchworkGP, fit
from pwgp.config import OptimizerConfig


@dataclass
class SmokeResult:
    figure: plt.Figure
    svgp: SVGP
    patchwork: GeneralizedPatchworkGP
    svgp_losses: list[float]
    patchwork_losses: list[float]


def run_visual_smoke(
    *,
    epochs: int = 120,
    observations: int = 80,
    noise_std: float = 0.08,
    seed: int = 7,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> SmokeResult:
    """Fit both models to one synthetic curve and return a diagnostic figure."""
    torch.manual_seed(seed)
    compute_device = torch.device(device)
    x = 0.04 + 0.92 * torch.rand(observations, 1, dtype=dtype)
    latent = torch.sin(2 * torch.pi * x[:, 0]) + 0.35 * torch.cos(6 * torch.pi * x[:, 0])
    y = latent + noise_std * torch.randn(observations, dtype=dtype)
    order = torch.argsort(x[:, 0])
    x, y = x[order].to(compute_device), y[order].to(compute_device)

    svgp_initial = torch.linspace(0.08, 0.92, 12, dtype=dtype)[:, None].to(compute_device)
    patchwork_initial = torch.stack(
        (
            torch.linspace(0.06, 0.46, 6, dtype=dtype)[:, None],
            torch.linspace(0.54, 0.94, 6, dtype=dtype)[:, None],
        )
    ).to(compute_device)
    stitch_initial = torch.tensor([[[0.5]]], dtype=dtype, device=compute_device)

    svgp = SVGP(
        svgp_initial.clone(),
        lengthscale=0.18,
        outputscale=1.0,
        noise_std=noise_std,
        train_supports=True,
        train_hyperparameters=True,
    ).to(compute_device)
    patchwork = GeneralizedPatchworkGP(
        patchwork_initial.clone(),
        stitch_initial.clone(),
        [(0, 1)],
        q_rank=4,
        lengthscale=torch.tensor([[0.18], [0.18]], dtype=dtype, device=compute_device),
        outputscale=torch.tensor([1.0, 1.0], dtype=dtype, device=compute_device),
        noise_std=noise_std,
        train_supports=True,
        train_stitches=True,
        train_hyperparameters=True,
    ).to(compute_device)
    svgp_optimizer = OptimizerConfig(
        learning_rate_q=2e-3,
        learning_rate_supports=2e-2,
        learning_rate_stitches=5e-3,
        learning_rate_hyperparameters=2e-3,
        warmup_fraction=0.2,
        alternate_patchwork_updates=True,
        gradient_clip=10.0,
        weight_decay=0.0,
    )
    patchwork_optimizer = OptimizerConfig(
        learning_rate_q=2e-3,
        learning_rate_supports=5e-2,
        learning_rate_stitches=5e-3,
        learning_rate_hyperparameters=1e-2,
        warmup_fraction=0.0,
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
        optimizer=svgp_optimizer,
    )
    patchwork_history = fit(
        patchwork,
        x,
        y,
        epochs=epochs,
        batch_size=observations,
        seed=seed,
        optimizer=patchwork_optimizer,
    )

    grid = torch.linspace(0.02, 0.98, 300, dtype=dtype, device=compute_device)[:, None]
    truth = torch.sin(2 * torch.pi * grid[:, 0]) + 0.35 * torch.cos(6 * torch.pi * grid[:, 0])
    svgp_mean, svgp_variance = svgp.predict(grid, observation_noise=False)
    pw_mean, pw_variance = patchwork.predict(grid, observation_noise=False)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    axes[0].plot(svgp_history.losses, label="SVGP")
    axes[0].plot(patchwork_history.losses, label="Patchwork")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Negative ELBO / N")
    axes[0].legend()

    views = (
        (axes[1], "SVGP", svgp_mean, svgp_variance, svgp_initial, svgp.supports()),
        (
            axes[2],
            "Generalized Patchwork",
            pw_mean,
            pw_variance,
            patchwork_initial.reshape(-1, 1),
            patchwork.supports().reshape(-1, 1),
        ),
    )
    for axis, title, mean, variance, initial, learned in views:
        standard_deviation = variance.sqrt()
        axis.scatter(x[:, 0].cpu(), y.cpu(), s=12, color="black", alpha=0.55, label="data")
        axis.plot(grid[:, 0].cpu(), truth.cpu(), color="black", linestyle="--", label="truth")
        axis.plot(grid[:, 0].cpu(), mean.cpu(), color="#146C94", label="mean")
        axis.fill_between(
            grid[:, 0].cpu(),
            (mean - 1.96 * standard_deviation).cpu(),
            (mean + 1.96 * standard_deviation).cpu(),
            color="#19A7CE",
            alpha=0.22,
            label="95% latent interval",
        )
        floor = float(y.min().cpu()) - 0.35
        axis.scatter(
            initial[:, 0].cpu(),
            torch.full((len(initial),), floor),
            marker="|",
            s=120,
            label="initial supports",
        )
        axis.scatter(
            learned[:, 0].detach().cpu(),
            torch.full((len(learned),), floor - 0.12),
            marker="|",
            s=120,
            label="learned supports",
        )
        axis.set(title=title, xlabel="x", ylabel="y")
    stitch = float(patchwork.stitches().detach().cpu().reshape(-1)[0])
    axes[2].axvline(stitch, color="#C84B31", linewidth=1.5, label="learned stitch")
    axes[1].legend(fontsize=8)
    axes[2].legend(fontsize=8)
    return SmokeResult(
        figure=figure,
        svgp=svgp,
        patchwork=patchwork,
        svgp_losses=svgp_history.losses,
        patchwork_losses=patchwork_history.losses,
    )


if __name__ == "__main__":
    result = run_visual_smoke()
    plt.show()

