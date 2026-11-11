"""Two-frequency sine regression with a fixed stitch at the phase boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pwgp import GeneralizedPatchworkGP, fit
from pwgp.config import HyperparameterBounds, NumericalConfig, OptimizerConfig

NORMAL_95 = 1.959963984540054


@dataclass(frozen=True)
class TwoPhaseSineConfig:
    """Complete configuration for the fixed-boundary diagnostic."""

    epochs: int = 500
    warmup_epochs: int = 30
    observations_per_phase: int = 128
    query_points: int = 1201
    supports_per_patch: int = 32
    q_rank: int = 16
    left_frequency: float = 1.0
    right_frequency: float = 5.0
    noise_std: float = 0.05
    left_lengthscale: float = 0.16
    right_lengthscale: float = 0.035
    outputscale: float = 1.0
    learning_rate_q: float = 5e-3
    learning_rate_hyperparameters: float = 3e-3
    weight_epsilon: float = 1e-2
    boundary_half_width: float = 0.1
    seed: int = 31
    torch_threads: int = 2
    device: str = "cpu"

    def validate(self) -> None:
        bounds = HyperparameterBounds()
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must lie in [0, epochs)")
        if self.observations_per_phase < 2:
            raise ValueError("observations_per_phase must be at least two")
        if self.query_points < 3 or self.query_points % 2 == 0:
            raise ValueError("query_points must be odd and at least three so x=0 is included")
        if self.supports_per_patch < 2:
            raise ValueError("supports_per_patch must be at least two")
        if not 0 <= self.q_rank <= 2 * self.supports_per_patch:
            raise ValueError("q_rank must lie between zero and the total number of supports")
        if self.left_frequency <= 0 or self.right_frequency <= 0:
            raise ValueError("frequencies must be positive")
        if not bounds.noise_std[0] < self.noise_std < bounds.noise_std[1]:
            raise ValueError(f"noise_std must lie strictly inside {bounds.noise_std}")
        for name, value in (
            ("left_lengthscale", self.left_lengthscale),
            ("right_lengthscale", self.right_lengthscale),
        ):
            if not bounds.lengthscale[0] < value < bounds.lengthscale[1]:
                raise ValueError(f"{name} must lie strictly inside {bounds.lengthscale}")
        if not bounds.outputscale[0] < self.outputscale < bounds.outputscale[1]:
            raise ValueError(f"outputscale must lie strictly inside {bounds.outputscale}")
        if self.learning_rate_q <= 0 or self.learning_rate_hyperparameters <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_epsilon <= 0:
            raise ValueError("weight_epsilon must be positive")
        if not 0 < self.boundary_half_width <= 1:
            raise ValueError("boundary_half_width must lie in (0, 1]")
        if self.torch_threads < 1:
            raise ValueError("torch_threads must be positive")
        if torch.device(self.device).type != "cpu":
            raise ValueError("this deterministic diagnostic currently requires device='cpu'")


def normalize_input(physical_x: torch.Tensor) -> torch.Tensor:
    """Map the physical domain [-1,1] to the model domain [0,1]."""
    return 0.5 * (physical_x + 1.0)


def physical_input(normalized_x: torch.Tensor) -> torch.Tensor:
    """Map normalized model inputs back to the physical domain [-1,1]."""
    return 2.0 * normalized_x - 1.0


def two_phase_truth(
    physical_x: torch.Tensor,
    *,
    left_frequency: float = 1.0,
    right_frequency: float = 5.0,
) -> torch.Tensor:
    """Continuous piecewise sine with a frequency change at x=0."""
    left = torch.sin(2.0 * torch.pi * left_frequency * physical_x)
    right = torch.sin(2.0 * torch.pi * right_frequency * physical_x)
    return torch.where(physical_x < 0.0, left, right)


def make_training_data(
    config: TwoPhaseSineConfig,
    *,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate balanced observations in the interiors of both phases."""
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    left_z = 0.01 + 0.48 * torch.rand(
        config.observations_per_phase, generator=generator, dtype=dtype
    )
    right_z = 0.51 + 0.48 * torch.rand(
        config.observations_per_phase, generator=generator, dtype=dtype
    )
    normalized = torch.cat((left_z, right_z))[:, None]
    order = torch.argsort(normalized[:, 0])
    normalized = normalized[order]
    physical = physical_input(normalized[:, 0])
    truth = two_phase_truth(
        physical,
        left_frequency=config.left_frequency,
        right_frequency=config.right_frequency,
    )
    observations = truth + config.noise_std * torch.randn(
        len(truth), generator=generator, dtype=dtype
    )
    return normalized, physical, truth, observations


def build_model(
    config: TwoPhaseSineConfig,
    *,
    dtype: torch.dtype = torch.float64,
) -> GeneralizedPatchworkGP:
    """Build two local models with fixed supports and a fixed boundary stitch."""
    device = torch.device(config.device)
    supports = torch.stack(
        (
            torch.linspace(0.015, 0.485, config.supports_per_patch, dtype=dtype),
            torch.linspace(0.515, 0.985, config.supports_per_patch, dtype=dtype),
        )
    )[..., None].to(device)
    stitches = torch.tensor([[[0.5]]], dtype=dtype, device=device)
    lengthscales = torch.tensor(
        [[config.left_lengthscale], [config.right_lengthscale]],
        dtype=dtype,
        device=device,
    )
    outputscales = torch.full(
        (2,), config.outputscale, dtype=dtype, device=device
    )
    model = GeneralizedPatchworkGP(
        supports,
        stitches,
        [(0, 1)],
        q_rank=config.q_rank,
        train_supports=False,
        train_stitches=False,
        train_hyperparameters=True,
        lengthscale=lengthscales,
        outputscale=outputscales,
        noise_std=config.noise_std,
        numerical=NumericalConfig(weight_epsilon=config.weight_epsilon),
    ).to(device)
    model.hyp.raw_noise.requires_grad_(False)
    return model


@torch.no_grad()
def prediction_decomposition(
    model: GeneralizedPatchworkGP,
    normalized_x: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return and verify mean, fusion weights, and both latent-variance terms."""
    prior = model.prior()
    q_state = model.q.state()
    terms = model.local_prediction_terms(normalized_x, prior, q_state)
    weights, fusion = model.regression_weights(
        normalized_x,
        prior,
        return_fusion=True,
        q_state=q_state,
    )
    mean = model.q.mean_from_weights(weights)
    inducing_variance = model.q.variance_from_weights(weights, q_state)
    residual_variance = torch.einsum(
        "bi,bij,bj->b", fusion, terms["residual"], fusion
    )
    latent_variance = inducing_variance + residual_variance
    reference_mean, reference_variance = model.predict(
        normalized_x, observation_noise=False
    )
    torch.testing.assert_close(mean, reference_mean, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(
        latent_variance, reference_variance, rtol=1e-9, atol=1e-10
    )
    torch.testing.assert_close(
        fusion.sum(-1),
        torch.ones(len(normalized_x), dtype=normalized_x.dtype, device=normalized_x.device),
        rtol=0,
        atol=1e-10,
    )
    for name, value in (
        ("inducing_variance", inducing_variance),
        ("residual_variance", residual_variance),
        ("latent_variance", latent_variance),
    ):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains a non-finite value")
    if bool((inducing_variance < 0).any()):
        raise FloatingPointError("w^T S w contains a negative value")
    if bool((residual_variance < 0).any()):
        raise FloatingPointError("alpha^T R alpha contains a negative value")
    if bool((latent_variance <= model.numerical.variance_floor).any()):
        raise FloatingPointError("latent variance is below the configured floor")
    return {
        "mean": mean,
        "fusion": fusion,
        "inducing_variance": inducing_variance,
        "residual_variance": residual_variance,
        "latent_variance": latent_variance,
        "reference_variance": reference_variance,
    }


def _normal_nlpd(
    observations: torch.Tensor,
    mean: torch.Tensor,
    variance: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        torch.log(2.0 * torch.pi * variance)
        + (observations - mean).square() / variance
    ).mean()


def _region_metrics(
    mask: torch.Tensor,
    *,
    truth: torch.Tensor,
    test_observations: torch.Tensor,
    mean: torch.Tensor,
    inducing_variance: torch.Tensor,
    residual_variance: torch.Tensor,
    noise_variance: torch.Tensor,
) -> dict[str, Any]:
    if not bool(mask.any()):
        raise ValueError("metric region is empty")
    truth_error = truth[mask] - mean[mask]
    observation_error = test_observations[mask] - mean[mask]
    inducing = inducing_variance[mask]
    residual = residual_variance[mask]
    full_latent = inducing + residual
    no_r_observation = inducing + noise_variance
    full_observation = full_latent + noise_variance

    def uncertainty_metrics(
        latent: torch.Tensor, observation: torch.Tensor
    ) -> dict[str, float]:
        return {
            "mean_latent_sd": float(latent.sqrt().mean()),
            "mean_observation_sd": float(observation.sqrt().mean()),
            "latent_coverage_95": float(
                (truth_error.abs() <= NORMAL_95 * latent.sqrt()).double().mean()
            ),
            "observation_coverage_95": float(
                (observation_error.abs() <= NORMAL_95 * observation.sqrt()).double().mean()
            ),
            "observation_nlpd": float(
                _normal_nlpd(
                    test_observations[mask], mean[mask], observation
                )
            ),
        }

    return {
        "points": int(mask.sum()),
        "latent_rmse": float(truth_error.square().mean().sqrt()),
        "latent_mae": float(truth_error.abs().mean()),
        "with_R": uncertainty_metrics(full_latent, full_observation),
        "without_R": uncertainty_metrics(inducing, no_r_observation),
        "mean_inducing_variance": float(inducing.mean()),
        "mean_residual_variance": float(residual.mean()),
        "mean_residual_fraction": float((residual / full_latent).mean()),
    }


def compute_metrics(
    config: TwoPhaseSineConfig,
    *,
    physical_x: torch.Tensor,
    truth: torch.Tensor,
    test_observations: torch.Tensor,
    decomposition: dict[str, torch.Tensor],
    noise_variance: torch.Tensor,
    stitch_physical: float,
    losses: list[float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Score fit quality and compare uncertainty with and without residual R."""
    regions = {
        "overall": torch.ones_like(physical_x, dtype=torch.bool),
        "left_phase": physical_x < 0,
        "right_phase": physical_x >= 0,
        "boundary": physical_x.abs() <= config.boundary_half_width,
    }
    values = {
        name: _region_metrics(
            mask,
            truth=truth,
            test_observations=test_observations,
            mean=decomposition["mean"],
            inducing_variance=decomposition["inducing_variance"],
            residual_variance=decomposition["residual_variance"],
            noise_variance=noise_variance,
        )
        for name, mask in regions.items()
    }
    variance_error = (
        decomposition["latent_variance"] - decomposition["reference_variance"]
    ).abs()
    fusion_error = (decomposition["fusion"].sum(-1) - 1.0).abs()
    return {
        "regions": values,
        "diagnostics": {
            "stitch_physical": stitch_physical,
            "variance_decomposition_max_abs_error": float(variance_error.max()),
            "fusion_sum_to_one_max_abs_error": float(fusion_error.max()),
            "fusion_weight_min": float(decomposition["fusion"].min()),
            "fusion_weight_max": float(decomposition["fusion"].max()),
            "final_negative_training_objective_per_observation": losses[-1],
            "training_elapsed_seconds": elapsed_seconds,
        },
    }


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def plot_regression(
    output: Path,
    *,
    config: TwoPhaseSineConfig,
    train_physical: torch.Tensor,
    train_observations: torch.Tensor,
    query_physical: torch.Tensor,
    truth: torch.Tensor,
    decomposition: dict[str, torch.Tensor],
    noise_variance: torch.Tensor,
    supports: torch.Tensor,
) -> None:
    """Save the fit, variance split, fusion weights, and uncertainty/error check."""
    x = _as_numpy(query_physical)
    mean = _as_numpy(decomposition["mean"])
    truth_np = _as_numpy(truth)
    inducing = _as_numpy(decomposition["inducing_variance"])
    residual = _as_numpy(decomposition["residual_variance"])
    latent = inducing + residual
    latent_sd = np.sqrt(latent)
    observation_sd = np.sqrt(latent + float(noise_variance))
    fusion = _as_numpy(decomposition["fusion"])
    residual_fraction = residual / latent

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fit_axis, variance_axis, weight_axis, error_axis = axes.flat
    fit_axis.fill_between(
        x,
        mean - NORMAL_95 * observation_sd,
        mean + NORMAL_95 * observation_sd,
        color="#8DA0CB",
        alpha=0.18,
        label="95% observation interval",
    )
    fit_axis.fill_between(
        x,
        mean - NORMAL_95 * latent_sd,
        mean + NORMAL_95 * latent_sd,
        color="#FC8D62",
        alpha=0.28,
        label="95% latent interval (includes R)",
    )
    fit_axis.scatter(
        _as_numpy(train_physical),
        _as_numpy(train_observations),
        s=10,
        color="#4D4D4D",
        alpha=0.42,
        label="noisy training data",
        zorder=2,
    )
    fit_axis.plot(x, truth_np, color="#1B9E77", linewidth=1.8, label="truth")
    fit_axis.plot(x, mean, color="#D95F02", linewidth=1.5, label="posterior mean")
    support_physical = _as_numpy(physical_input(supports.squeeze(-1)))
    for patch, color in enumerate(("#7570B3", "#E7298A")):
        fit_axis.scatter(
            support_physical[patch],
            np.full(config.supports_per_patch, 0.025),
            transform=fit_axis.get_xaxis_transform(),
            marker="|",
            s=65,
            color=color,
            alpha=0.75,
            label=f"patch {patch + 1} supports",
        )
    fit_axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.1, label="fixed stitch")
    fit_axis.set(
        title="Two-frequency regression",
        xlabel="physical input x",
        ylabel="f(x)",
        xlim=(-1.0, 1.0),
    )
    fit_axis.legend(fontsize=8, ncol=2, loc="upper left")

    variance_axis.plot(x, latent, color="#D95F02", linewidth=1.7, label="total: wᵀSw + αᵀRα")
    variance_axis.plot(x, inducing, color="#7570B3", linewidth=1.4, label="inducing: wᵀSw")
    variance_axis.plot(x, residual, color="#1B9E77", linewidth=1.4, label="residual: αᵀRα")
    variance_axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    variance_axis.set(
        title="Latent variance decomposition",
        xlabel="physical input x",
        ylabel="variance",
        xlim=(-1.0, 1.0),
    )
    variance_axis.legend(fontsize=8)

    weight_axis.plot(x, fusion[:, 0], color="#7570B3", label="left-model weight")
    weight_axis.plot(x, fusion[:, 1], color="#E7298A", label="right-model weight")
    weight_axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    weight_axis.set(
        title="Posterior-adaptive fusion",
        xlabel="physical input x",
        ylabel="fusion weight",
        xlim=(-1.0, 1.0),
    )
    fraction_axis = weight_axis.twinx()
    fraction_axis.plot(
        x,
        residual_fraction,
        color="#1B9E77",
        linestyle=":",
        linewidth=1.5,
        label="R fraction of latent variance",
    )
    fraction_axis.set_ylabel("residual variance fraction")
    lines = weight_axis.lines[:2] + fraction_axis.lines
    weight_axis.legend(lines, [line.get_label() for line in lines], fontsize=8)

    absolute_error = np.abs(truth_np - mean)
    error_axis.plot(x, absolute_error, color="#4D4D4D", linewidth=1.3, label="absolute mean error")
    error_axis.plot(
        x,
        NORMAL_95 * latent_sd,
        color="#D95F02",
        linewidth=1.5,
        label="95% latent half-width (includes R)",
    )
    error_axis.plot(
        x,
        NORMAL_95 * np.sqrt(inducing),
        color="#7570B3",
        linestyle="--",
        linewidth=1.2,
        label="95% half-width without R",
    )
    error_axis.axvline(0.0, color="#111111", linestyle="--", linewidth=1.0)
    error_axis.set(
        title="Regression error versus stated uncertainty",
        xlabel="physical input x",
        ylabel="absolute magnitude",
        xlim=(-1.0, 1.0),
    )
    error_axis.legend(fontsize=8)
    figure.suptitle(
        "Fixed stitch at x=0: sin(2πx) for x<0, sin(10πx) for x≥0",
        fontsize=13,
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_training(output: Path, losses: list[float]) -> None:
    """Save the posterior-adaptive training-objective trace."""
    figure, axis = plt.subplots(figsize=(7, 3.4), constrained_layout=True)
    axis.plot(np.arange(1, len(losses) + 1), losses, color="#D95F02", linewidth=1.3)
    axis.set(
        title="Training trace",
        xlabel="epoch",
        ylabel="negative training objective / observation",
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _tensor_digest(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(_as_numpy(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(command: list[str]) -> str:
    result = subprocess.run(
        ["git", *command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def run_experiment(
    config: TwoPhaseSineConfig,
    *,
    output_root: Path = Path("results/two_phase_sine"),
) -> Path:
    """Train, score, plot, and persist one complete two-phase experiment."""
    config.validate()
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    dtype = torch.float64
    output = output_root / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True, exist_ok=False)
    script_path = Path(__file__).resolve()
    repository = script_path.parents[1]
    numerical = NumericalConfig(weight_epsilon=config.weight_epsilon)
    optimizer = OptimizerConfig(
        learning_rate_q=config.learning_rate_q,
        learning_rate_supports=2e-3,
        learning_rate_stitches=2e-3,
        learning_rate_hyperparameters=config.learning_rate_hyperparameters,
        warmup_fraction=config.warmup_epochs / config.epochs,
        alternate_patchwork_updates=True,
        gradient_clip=10.0,
        weight_decay=0.0,
    )
    run_config = {
        "experiment": asdict(config),
        "model": {
            "class": "pwgp.models.patchwork.GeneralizedPatchworkGP",
            "package": "generalized-patchwork-gp",
            "package_version": version("generalized-patchwork-gp"),
            "patches": 2,
            "edges": [[0, 1]],
            "fixed_stitch_normalized": 0.5,
            "fixed_stitch_physical": 0.0,
            "train_supports": False,
            "train_stitches": False,
            "train_noise": False,
            "train_lengthscales": True,
            "train_outputscales": True,
            "variational_family": "q(U)=N(m,(D+VV^T)^-1)",
            "latent_variance": "w^T S w + alpha^T R alpha",
        },
        "optimizer": asdict(optimizer),
        "numerical": asdict(numerical),
        "coordinates": {
            "physical_domain": [-1.0, 1.0],
            "normalization": "z=(x+1)/2",
        },
    }
    _write_json(output / "config.json", run_config)
    _write_json(output / "status.json", {"status": "running"})
    try:
        train_z, train_x, train_truth, train_y = make_training_data(config, dtype=dtype)
        model = build_model(config, dtype=dtype)
        initial_stitch = model.stitches().detach().clone()
        history = fit(
            model,
            train_z.to(config.device),
            train_y.to(config.device),
            epochs=config.epochs,
            batch_size=len(train_z),
            seed=config.seed,
            optimizer=optimizer,
        )
        torch.testing.assert_close(model.stitches(), initial_stitch, rtol=0, atol=0)

        query_x = torch.linspace(-1.0, 1.0, config.query_points, dtype=dtype)
        query_z = normalize_input(query_x)[:, None].to(config.device)
        query_truth = two_phase_truth(
            query_x,
            left_frequency=config.left_frequency,
            right_frequency=config.right_frequency,
        ).to(config.device)
        decomposition = prediction_decomposition(model, query_z)
        test_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
        test_noise = config.noise_std * torch.randn(
            config.query_points, generator=test_generator, dtype=dtype
        )
        test_y = query_truth.cpu() + test_noise
        noise_variance = model.hyp.noise_std().square().detach()
        stitch_physical = float(physical_input(model.stitches().reshape(-1))[0])
        metrics = compute_metrics(
            config,
            physical_x=query_x.to(config.device),
            truth=query_truth,
            test_observations=test_y.to(config.device),
            decomposition=decomposition,
            noise_variance=noise_variance,
            stitch_physical=stitch_physical,
            losses=history.losses,
            elapsed_seconds=history.elapsed_seconds,
        )
        learned = {
            "lengthscales_normalized": _as_numpy(model.hyp.lengthscales()).tolist(),
            "outputscales": _as_numpy(model.hyp.outputscales()).tolist(),
            "noise_std": float(model.hyp.noise_std()),
            "supports_normalized": _as_numpy(model.supports()).tolist(),
            "stitches_normalized": _as_numpy(model.stitches()).tolist(),
        }
        dataset = {
            "generator": "balanced synthetic two-frequency sine",
            "training_points": len(train_z),
            "test_points": config.query_points,
            "training_input_sha256": _tensor_digest(train_z),
            "training_truth_sha256": _tensor_digest(train_truth),
            "training_observation_sha256": _tensor_digest(train_y),
            "test_input_sha256": _tensor_digest(query_x),
            "test_truth_sha256": _tensor_digest(query_truth),
            "test_observation_sha256": _tensor_digest(test_y),
            "training_seed": config.seed,
            "independent_test_noise_seed": config.seed + 1,
        }
        provenance = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": config.device,
            "dtype": str(dtype),
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "git_head": _git(["rev-parse", "HEAD"]),
            "git_branch": _git(["branch", "--show-current"]),
            "git_status": _git(["status", "--short"]),
            "runner": str(script_path.relative_to(repository)),
            "runner_sha256": _file_digest(script_path),
            "patchwork_source_sha256": _file_digest(
                repository / "src/pwgp/models/patchwork.py"
            ),
        }
        np.savez_compressed(
            output / "predictions.npz",
            train_normalized=_as_numpy(train_z),
            train_physical=_as_numpy(train_x),
            train_truth=_as_numpy(train_truth),
            train_observations=_as_numpy(train_y),
            query_normalized=_as_numpy(query_z),
            query_physical=_as_numpy(query_x),
            query_truth=_as_numpy(query_truth),
            independent_test_observations=_as_numpy(test_y),
            posterior_mean=_as_numpy(decomposition["mean"]),
            inducing_variance=_as_numpy(decomposition["inducing_variance"]),
            residual_variance=_as_numpy(decomposition["residual_variance"]),
            latent_variance=_as_numpy(decomposition["latent_variance"]),
            fusion_weights=_as_numpy(decomposition["fusion"]),
        )
        torch.save(model.state_dict(), output / "model.pt")
        _write_json(output / "dataset.json", dataset)
        _write_json(output / "metrics.json", metrics)
        _write_json(output / "learned_parameters.json", learned)
        _write_json(output / "history.json", asdict(history))
        _write_json(output / "provenance.json", provenance)
        plot_regression(
            output / "regression.png",
            config=config,
            train_physical=train_x,
            train_observations=train_y,
            query_physical=query_x,
            truth=query_truth,
            decomposition=decomposition,
            noise_variance=noise_variance,
            supports=model.supports(),
        )
        plot_training(output / "training.png", history.losses)
        artifacts = sorted(path.name for path in output.iterdir() if path.name != "status.json")
        _write_json(
            output / "status.json",
            {
                "status": "complete",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "artifacts": artifacts,
            },
        )
    except BaseException as error:
        _write_json(
            output / "status.json",
            {
                "status": "failed",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps(metrics, indent=2))
    print(f"Results: {output.resolve()}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TwoPhaseSineConfig.epochs)
    parser.add_argument("--warmup-epochs", type=int, default=TwoPhaseSineConfig.warmup_epochs)
    parser.add_argument(
        "--observations-per-phase",
        type=int,
        default=TwoPhaseSineConfig.observations_per_phase,
    )
    parser.add_argument("--query-points", type=int, default=TwoPhaseSineConfig.query_points)
    parser.add_argument(
        "--supports-per-patch", type=int, default=TwoPhaseSineConfig.supports_per_patch
    )
    parser.add_argument("--q-rank", type=int, default=TwoPhaseSineConfig.q_rank)
    parser.add_argument("--seed", type=int, default=TwoPhaseSineConfig.seed)
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/two_phase_sine")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TwoPhaseSineConfig(
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        observations_per_phase=args.observations_per_phase,
        query_points=args.query_points,
        supports_per_patch=args.supports_per_patch,
        q_rank=args.q_rank,
        seed=args.seed,
    )
    run_experiment(config, output_root=args.output_root)


if __name__ == "__main__":
    main()
