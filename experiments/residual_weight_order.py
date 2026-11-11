"""Compare residual-before-weights and residual-after-weights on one fitted GPW."""

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

from experiments.residual_2d_diagnostic import signal
from pwgp import GeneralizedPatchworkGP, fit
from pwgp.config import NumericalConfig, OptimizerConfig
from pwgp.models.patchwork import agreement_weights


def local_posterior_terms(model, x, prior, state):
    """Return local means, inducing-propagated covariance, and conditional residual."""
    terms = model.local_query_terms(x, prior)
    eye = torch.eye(model.patches, dtype=x.dtype, device=x.device)
    a = torch.einsum("ij,bim->bijm", eye, terms["phi0"])
    a = a - torch.einsum("biq,pmq->bipm", terms["solved_cross"], prior["w"])
    projected = torch.einsum("bipm,pmn->bipn", a, state["local"])
    covariance = torch.einsum("bipn,bjpn->bij", projected, projected)
    correction = torch.einsum("bipm,pmr->bir", a, state["correction"])
    covariance = covariance - correction @ correction.transpose(-1, -2)
    means = torch.einsum("bipm,pm->bi", a, model.q.mean)
    return a, means, covariance, terms["residual"]


def run(
    *, epochs=100, training_side=24, query_side=61, gap_radius=0.25,
    supports_per_patch=16, q_rank=8, lengthscale=0.12, outputscale=1.0,
    noise_std=0.08, learning_rate_q=0.02, weight_epsilon=0.01, seed=31,
):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    dtype = torch.float64
    axis = torch.linspace(0.02, 0.98, training_side, dtype=dtype)
    x = torch.cartesian_prod(axis, axis)
    x = x[(x - 0.5).square().sum(-1) >= gap_radius**2]
    y = signal(x) + noise_std * torch.randn(len(x), dtype=dtype)
    patch_id = (3 * x[:, 0]).long()
    supports = []
    for patch in range(3):
        candidates = x[patch_id == patch]
        indices = torch.linspace(0, len(candidates) - 1, supports_per_patch).round().long()
        if len(indices.unique()) != supports_per_patch:
            raise ValueError("Not enough distinct support candidates")
        supports.append(candidates[indices])
    supports = torch.stack(supports)
    stitches = torch.tensor(
        [[[1 / 3, 0.1], [1 / 3, 0.9]], [[2 / 3, 0.1], [2 / 3, 0.9]]], dtype=dtype,
    )
    numerical = NumericalConfig(weight_epsilon=weight_epsilon)
    optimizer = OptimizerConfig(learning_rate_q=learning_rate_q, warmup_fraction=0.0)
    model = GeneralizedPatchworkGP(
        supports, stitches, [(0, 1), (1, 2)], q_rank=q_rank,
        train_supports=False, train_stitches=False, train_hyperparameters=False,
        lengthscale=lengthscale, outputscale=outputscale, noise_std=noise_std,
        numerical=numerical,
    )
    history = fit(model, x, y, epochs=epochs, batch_size=len(x), seed=seed, optimizer=optimizer)
    axis = torch.linspace(0, 1, query_side, dtype=dtype)
    query = torch.cartesian_prod(axis, axis)
    values = {name: {key: [] for key in ("mean", "variance", "alpha")}
              for name in ("R_after_weights", "R_before_weights")}
    with torch.no_grad():
        prior, state = model.prior(), model.q.state()
        for batch in query.split(512):
            a, means, inducing, residual = local_posterior_terms(model, batch, prior, state)
            total = inducing + residual
            original_alpha = model.fusion_weights(batch, prior)
            original_w = torch.einsum("bi,bipm->bpm", original_alpha, a)
            torch.testing.assert_close(original_w, model.regression_weights(batch, prior))
            for name, weight_covariance in [("R_after_weights", inducing),
                                            ("R_before_weights", total)]:
                alpha = agreement_weights(weight_covariance, epsilon=weight_epsilon)
                variance = torch.einsum("bi,bij,bj->b", alpha, total, alpha)
                w = torch.einsum("bi,bipm->bpm", alpha, a)
                expected = model.q.variance_from_weights(w, state) + torch.einsum(
                    "bi,bij,bj->b", alpha, residual, alpha)
                torch.testing.assert_close(variance, expected)
                torch.testing.assert_close(alpha.sum(-1), torch.ones(len(batch), dtype=dtype))
                if not torch.isfinite(variance).all() or (variance <= 0).any():
                    raise FloatingPointError(f"Invalid variance: {name}")
                values[name]["mean"].append((alpha * means).sum(-1))
                values[name]["variance"].append(variance)
                values[name]["alpha"].append(alpha)
    values = {name: {key: torch.cat(parts) for key, parts in fields.items()}
              for name, fields in values.items()}
    truth = signal(query)
    test_y = truth + noise_std * torch.randn(len(query), dtype=dtype)
    gap = (query - 0.5).square().sum(-1) < gap_radius**2
    rows = []
    for name, fields in values.items():
        for region, mask in [("gap", gap), ("observed_region", ~gap)]:
            variance = fields["variance"][mask] + noise_std**2
            error = test_y[mask] - fields["mean"][mask]
            rows.append({"method": name, "region": region,
                         "NLPD": float((0.5 * (torch.log(2 * torch.pi * variance)
                                               + error.square() / variance)).mean()),
                         "latent_RMSE": float((truth[mask] - fields["mean"][mask]).square().mean().sqrt()),
                         "mean_latent_SD": float(fields["variance"][mask].sqrt().mean()),
                         "observation_coverage_95": float(
                             (error.abs() <= 1.959963984540054 * variance.sqrt()).double().mean()),
                         "mean_alpha": fields["alpha"][mask].mean(0).tolist()})
    output = Path("results/residual_weight_order") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    config = {"epochs": epochs, "training_side": training_side, "query_side": query_side,
              "gap_radius": gap_radius, "supports_per_patch": supports_per_patch, "patches": 3,
              "q_rank": q_rank, "lengthscale": lengthscale, "outputscale": outputscale,
              "noise_std": noise_std, "seed": seed, "numerical": asdict(numerical),
              "optimizer": asdict(optimizer), "device": "cpu", "dtype": "float64",
              "torch": torch.__version__, "comparison": "Two post-hoc weight rules on one fitted q",
              "training_weight_rule": "Production rule uses C_U+R, including residual variance in training",
              "evaluation_weight_rules": "agreement_weights(C_U) vs agreement_weights(C_U+R)",
              "final_covariance_both": "C_U+R"}
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output / "history.json").write_text(json.dumps(asdict(history)), encoding="utf-8")
    torch.save(model.state_dict(), output / "model.pt")
    arrays = {f"{name}_{key}": value.numpy() for name, fields in values.items()
              for key, value in fields.items()}
    np.savez_compressed(output / "predictions.npz", query=query.numpy(), truth=truth.numpy(),
                        gap=gap.numpy(), train_x=x.numpy(), train_y=y.numpy(),
                        test_y=test_y.numpy(), **arrays)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    sd_max = max(float(fields["variance"].sqrt().max()) for fields in values.values())
    alpha_min = min(float(fields["alpha"].min()) for fields in values.values())
    alpha_max = max(float(fields["alpha"].max()) for fields in values.values())
    for row, (name, fields) in enumerate(values.items()):
        maps = [fields["mean"], fields["variance"].sqrt(), *fields["alpha"].T]
        titles = ["Mean", "Full latent SD", "Patch 1 weight", "Patch 2 weight", "Patch 3 weight"]
        for col, (field, title) in enumerate(zip(maps, titles, strict=True)):
            limits = (-1, 1) if col == 0 else (0, sd_max) if col == 1 else (alpha_min, alpha_max)
            ax = axes[row, col]
            im = ax.imshow(field.reshape(query_side, query_side).T.numpy(), origin="lower",
                           extent=(0, 1, 0, 1), vmin=limits[0], vmax=limits[1],
                           cmap="coolwarm" if col == 0 else "viridis")
            ax.add_patch(plt.Circle((0.5, 0.5), gap_radius, fill=False, color="red", linestyle="--"))
            if col == 0:
                ax.scatter(*x.numpy().T, c="black", s=1, alpha=0.3)
            ax.set(title=title, xlabel="x", ylabel=name if col == 0 else "y")
            fig.colorbar(im, ax=ax, shrink=0.75)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(rows, indent=2))
    print("Results:", output.resolve())
    return output


if __name__ == "__main__":
    run()
