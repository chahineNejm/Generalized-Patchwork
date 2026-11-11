"""Isolate residual propagation on one fitted GPW with a central observation gap."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pwgp import GeneralizedPatchworkGP, fit
from pwgp.config import NumericalConfig, OptimizerConfig


def signal(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(2 * torch.pi * x[:, 0]) * torch.cos(2 * torch.pi * x[:, 1])


def run(
    *, epochs: int = 100, training_side: int = 24, query_side: int = 61,
    gap_radius: float = 0.25, supports_per_patch: int = 12, q_rank: int = 8,
    lengthscale: float = 0.12, outputscale: float = 1.0, noise_std: float = 0.08,
    learning_rate_q: float = 0.02, weight_epsilon: float = 0.01, seed: int = 31,
) -> Path:
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    dtype = torch.float64
    axis = torch.linspace(0.02, 0.98, training_side, dtype=dtype)
    x = torch.cartesian_prod(axis, axis)
    x = x[(x - 0.5).square().sum(-1) >= gap_radius**2]
    y = signal(x) + noise_std * torch.randn(len(x), dtype=dtype)
    patch_id = (x[:, 0] >= 0.5).long() + 2 * (x[:, 1] >= 0.5).long()
    supports = []
    for patch in range(4):
        candidates = x[patch_id == patch]
        indices = torch.linspace(0, len(candidates) - 1, supports_per_patch).round().long()
        if len(indices.unique()) != supports_per_patch:
            raise ValueError("Not enough distinct training locations for supports")
        supports.append(candidates[indices])
    supports = torch.stack(supports)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    stitches = torch.tensor(
        [[[0.5, 0.10], [0.5, 0.20]], [[0.10, 0.5], [0.20, 0.5]],
         [[0.80, 0.5], [0.90, 0.5]], [[0.5, 0.80], [0.5, 0.90]]], dtype=dtype,
    )
    numerical = NumericalConfig(weight_epsilon=weight_epsilon)
    optimizer = OptimizerConfig(learning_rate_q=learning_rate_q, warmup_fraction=0.0)
    model = GeneralizedPatchworkGP(
        supports, stitches, edges, q_rank=q_rank, train_supports=False,
        train_stitches=False, train_hyperparameters=False, lengthscale=lengthscale,
        outputscale=outputscale, noise_std=noise_std, numerical=numerical,
    )
    history = fit(model, x, y, epochs=epochs, batch_size=len(x), seed=seed, optimizer=optimizer)
    axis = torch.linspace(0, 1, query_side, dtype=dtype)
    query = torch.cartesian_prod(axis, axis)
    means, current, residuals = [], [], []
    with torch.no_grad():
        prior, state = model.prior(), model.q.state()
        for batch in query.split(512):
            weights, alpha = model.regression_weights(batch, prior, return_fusion=True)
            local_residual = model.local_query_terms(batch, prior)["residual"]
            residual = torch.einsum("bi,bij,bj->b", alpha, local_residual, alpha)
            if not torch.isfinite(residual).all() or (residual < 0).any():
                raise FloatingPointError("Invalid aggregated residual; no clipping applied")
            torch.testing.assert_close(alpha.sum(-1), torch.ones(len(batch), dtype=dtype))
            means.append(model.q.mean_from_weights(weights))
            current.append(model.q.variance_from_weights(weights, state))
            residuals.append(residual)
        mean, current, residual = map(torch.cat, (means, current, residuals))
        reference_mean, reference_variance = model.predict(query, observation_noise=False)
        torch.testing.assert_close(mean, reference_mean)
        torch.testing.assert_close(current + residual, reference_variance)
    corrected = current + residual
    truth = signal(query)
    test_y = truth + noise_std * torch.randn(len(query), dtype=dtype)
    gap = (query - 0.5).square().sum(-1) < gap_radius**2
    rows = []
    for region, mask in [("unobserved_gap", gap), ("observed_region", ~gap)]:
        for name, variance in [("current", current), ("with_residual", corrected)]:
            observation_variance = variance[mask] + noise_std**2
            errors = test_y[mask] - mean[mask]
            rows.append({
                "region": region, "variance": name, "points": int(mask.sum()),
                "mean_latent_sd": float(variance[mask].sqrt().mean()),
                "NLPD": float((0.5 * (torch.log(2 * torch.pi * observation_variance)
                                     + errors.square() / observation_variance)).mean()),
                "observation_coverage_95": float(
                    (errors.abs() <= 1.959963984540054 * observation_variance.sqrt()).double().mean()),
                "latent_coverage_95": float(
                    ((truth[mask] - mean[mask]).abs() <= 1.959963984540054
                     * variance[mask].sqrt()).double().mean()),
                "RMSE_latent": float((truth[mask] - mean[mask]).square().mean().sqrt()),
            })
    output = Path("results/residual_2d") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "epochs": epochs, "training_side": training_side, "query_side": query_side,
        "gap_radius": gap_radius, "supports_per_patch": supports_per_patch,
        "patches": 4, "q_rank": q_rank, "lengthscale": lengthscale,
        "outputscale": outputscale, "noise_std": noise_std, "seed": seed,
        "numerical": asdict(numerical), "optimizer": asdict(optimizer),
        "device": "cpu", "dtype": "float64", "torch": torch.__version__,
        "comparison": "Same model trained with full covariance; residual excluded/included at evaluation",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(asdict(history)), encoding="utf-8")
    torch.save(model.state_dict(), output / "model.pt")
    np.savez_compressed(output / "predictions.npz", query=query.numpy(), mean=mean.numpy(),
                        truth=truth.numpy(), test_y=test_y.numpy(), gap=gap.numpy(),
                        current_variance=current.numpy(), residual=residual.numpy(),
                        corrected_variance=corrected.numpy(), train_x=x.numpy(), train_y=y.numpy())
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    fields = [truth, mean, (truth - mean).abs(), current.sqrt(), corrected.sqrt(), residual.sqrt()]
    titles = ["True function / observations", "Same fitted mean", "Absolute mean error",
              "Current latent SD: wSw", "Latent SD: wSw + alpha R alpha", "Added residual SD"]
    sd_max = float(corrected.sqrt().max())
    for index, (ax, field, title) in enumerate(zip(axes.flat, fields, titles, strict=True)):
        limits = {"vmin": -1, "vmax": 1} if index < 2 else {"vmin": 0}
        if index >= 3:
            limits["vmax"] = sd_max
        im = ax.imshow(field.reshape(query_side, query_side).T.numpy(), origin="lower",
                       extent=(0, 1, 0, 1), cmap="coolwarm" if index < 2 else "viridis", **limits)
        ax.add_patch(plt.Circle((0.5, 0.5), gap_radius, fill=False, color="red", linestyle="--"))
        if index == 0:
            ax.scatter(*x.numpy().T, s=2, c="black", alpha=0.5)
        if index in (3, 4):
            ax.scatter(*supports.reshape(-1, 2).numpy().T, s=10, c="white", edgecolors="black")
        ax.set(title=title, xlabel="x", ylabel="y")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3), constrained_layout=True)
    ax.plot(range(1, epochs + 1), history.losses)
    ax.set(xlabel="Epoch", ylabel="Negative ELBO / observation")
    fig.savefig(output / "training.png", dpi=160)
    plt.close(fig)
    print(json.dumps(rows, indent=2))
    print("Results:", output.resolve())
    return output


if __name__ == "__main__":
    run()
