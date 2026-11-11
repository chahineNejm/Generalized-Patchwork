"""Sweep GPW weight regularization on the two-regime diagnostic."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.dynamic_regime_test import run_dynamic_regime_test


def run_weight_epsilon_sweep(
    *,
    epsilons: tuple[float, ...] = (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
    epochs: int = 500,
    warmup_epochs: int = 30,
    output_directory: str = "results/weight_epsilon_sweep",
) -> list[dict[str, float]]:
    """Run identical fits across explicit GPW weight epsilons."""
    if not 0 < warmup_epochs < epochs:
        raise ValueError("warmup_epochs must lie strictly between zero and epochs")
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    for epsilon in epsilons:
        result = run_dynamic_regime_test(
            epochs=epochs,
            warmup_fraction=warmup_epochs / epochs,
            weight_epsilon=epsilon,
        )
        row = {"weight_epsilon": epsilon, **result.metrics}
        lengthscales = result.patchwork.hyp.lengthscales().detach().cpu().reshape(-1)
        row["lengthscale_1"] = float(lengthscales[0])
        row["lengthscale_2"] = float(lengthscales[1])
        row["stitch"] = float(result.patchwork.stitches().detach().cpu().reshape(-1)[0])
        rows.append(row)
        result.figure.savefig(
            directory / f"epsilon_{epsilon:.0e}.png", dpi=140, bbox_inches="tight"
        )
        plt.close(result.figure)

    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    values = [row["weight_epsilon"] for row in rows]
    axes[0].semilogx(values, [row["patchwork_rmse"] for row in rows], marker="o")
    axes[0].set(xlabel="weight epsilon", ylabel="GPW RMSE", title="Prediction error")
    axes[1].semilogx(values, [row["weight_min"] for row in rows], marker="o", label="minimum")
    axes[1].semilogx(values, [row["weight_max"] for row in rows], marker="o", label="maximum")
    axes[1].set(xlabel="weight epsilon", ylabel="weight", title="GPW weight range")
    axes[1].legend()
    figure.savefig(directory / "summary.png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return rows


if __name__ == "__main__":
    for values in run_weight_epsilon_sweep():
        print(values)
