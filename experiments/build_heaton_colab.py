"""Build the GitHub-backed Heaton Colab launcher."""

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from pathlib import Path

from experiments.colab_source import github_bootstrap, source_metadata


def cell(kind: str, source: str) -> dict:
    source = textwrap.dedent(source).strip() + "\n"
    if kind == "code":
        last = ast.parse(source).body[-1]
        result_expression = isinstance(last, ast.Expr)
        if result_expression:
            lines = source.splitlines(keepends=True)
            lines[last.lineno - 1] = "_GPW_CELL_RESULT = " + lines[last.lineno - 1]
            source = "".join(lines)
        source = (
            'if "_GPW_STAGE_FAILED" not in globals():\n'
            '    _GPW_STAGE_FAILED = False\n'
            'if _GPW_STAGE_FAILED:\n'
            '    raise RuntimeError("An earlier cell failed; restart the runtime before continuing")\n'
            '_GPW_STAGE_FAILED = True\n'
            + source + '_GPW_STAGE_FAILED = False\n'
            + ('_GPW_CELL_RESULT\n' if result_expression else '')
        )
    result = {
        "cell_type": kind,
        "id": hashlib.sha256(source.encode()).hexdigest()[:12],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


def build_notebook(root: Path) -> Path:
    unpack = github_bootstrap("/content/gpw_heaton")
    cells = [
        cell(
            "markdown",
            """
        # GPW and SVGP: Heaton MODIS benchmark

        Upload this notebook to Google Colab, select a GPU runtime, then Run all.
        Restart the runtime first if you previously ran an older notebook copy.
        Code loads from the latest main of chahineNejm/Generalized-Patchwork.
        The resolved commit and source archive are saved with the results.

        Runs, in order: **GPW 8 x 60**, **SVGP 480**, **GPW 4 x 120**.
        Each uses **100 total epochs**, with kernel/noise parameters frozen for
        the first **20 epochs**. Inducing/stitch positions train from the start.
        Training uses float64 and the complete official MODIS split:
        105,569 training observations / 42,740 held-out observations.

        GPW agreement weights now use the full local posterior covariance
        C(x) = A(x) S A(x)^T + R(x,x), including stitched residual correlations.
        Latent variance is w(x)^T S w(x) + alpha(x)^T R(x,x) alpha(x).
        Both terms enter training and prediction. Measurement variance adds the
        learned observation noise once; SVGP is unchanged.
        This is posterior-adaptive aggregation: weights depend on S and receive
        gradients. Its expected-log-likelihood minus KL objective is not claimed
        to be an ELBO for a fixed generative model.

        Cholesky solves use two explicit triangular solves, not the batched
        POTRS kernel. GPU setup checks representative batched solves and their
        gradients before any benchmark training.

        Each run displays tqdm, live loss/position diagnostics every 10 epochs,
        final temperature/error/uncertainty maps, and its metrics. GPW also
        displays fusion-weight maps. A downloadable archive contains all results,
        model states, source snapshot, exact settings, and provenance.

        No CPU fallback, retries, automatic batch reduction, kernel substitution,
        or automatic hyperparameter adjustment is implemented.
        Existing model bounds are preserved and displayed before training.
        This uses the current ARD RBF models, not a kernel-matched reproduction
        of the original competition's exponential-covariance implementations.
        """,
        ),
        cell(
            "markdown",
            """
        ## 1. GitHub source and dependencies
        Uses Colab's installed CUDA-enabled PyTorch. Other dependencies are installed
        from the repository package declaration. No local GPU driver installation.
        """,
        ),
        cell("code", unpack),
        cell("code", '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--editable", "/content/gpw_heaton[visual,benchmark]"], check=True)'''),
        cell("markdown", "## 2. Fixed configurations and GPU validation"),
        cell(
            "code",
            """
        import json, shutil
        from dataclasses import asdict, replace
        from datetime import UTC, datetime
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import Image, Markdown, display
        from experiments.heaton_benchmark import HeatonConfig, run_heaton
        from pwgp.config import HyperparameterBounds, NumericalConfig
        from pwgp.linalg import cholesky_solve
        from pwgp import GeneralizedPatchworkGP

        if not torch.cuda.is_available():
            raise RuntimeError("A Colab GPU runtime is required. Select Runtime > Change runtime type > GPU, then Run all.")
        DEVICE = "cuda:0"
        print({"GPU": torch.cuda.get_device_name(0), "VRAM_GB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
               "PyTorch": torch.__version__, "CUDA": torch.version.cuda, "dtype": "float64",
               "compute_capability": torch.cuda.get_device_capability(0),
               "compiled_architectures": torch.cuda.get_arch_list(),
               "cholesky_solver": "two_triangular_solves"}, flush=True)

        def check_batched_solve(batch_count, size, columns):
            print(f"GPU check: {batch_count} systems, size {size}, {columns} RHS columns", flush=True)
            matrix = (torch.eye(size, dtype=torch.float64, device=DEVICE) + 0.01).expand(
                batch_count, size, size).clone().requires_grad_(True)
            rhs = torch.full((batch_count, size, columns), 0.25, dtype=torch.float64,
                             device=DEVICE, requires_grad=True)
            solution = cholesky_solve(rhs, torch.linalg.cholesky(matrix))
            torch.testing.assert_close(matrix @ solution, rhs, rtol=1e-10, atol=1e-10)
            solution.square().mean().backward()
            torch.cuda.synchronize()
            assert torch.isfinite(matrix.grad).all(), "Non-finite matrix gradients in GPU check"
            assert torch.isfinite(rhs.grad).all(), "Non-finite RHS gradients in GPU check"

        GPU_PROBES = [(8, 60, 96), (4, 120, 48), (2048, 7, 1), (1, 480, 16)]
        for shape in GPU_PROBES:
            check_batched_solve(*shape)
        print("GPU batched solve and backward checks passed.", flush=True)

        probe_generator = torch.Generator(device=DEVICE).manual_seed(31)
        probe_supports = 0.1 + 0.8 * torch.rand(3, 4, 2, dtype=torch.float64,
                                              device=DEVICE, generator=probe_generator)
        probe_stitches = 0.1 + 0.8 * torch.rand(2, 2, 2, dtype=torch.float64,
                                              device=DEVICE, generator=probe_generator)
        probe_model = GeneralizedPatchworkGP(probe_supports, probe_stitches,
                                             [(0, 1), (1, 2)], q_rank=3,
                                             numerical=NumericalConfig(weight_epsilon=0.01))
        probe_x = probe_supports.reshape(-1, 2)[:7]
        probe_objective, probe_mean, probe_variance, _ = probe_model.elbo_batch(
            probe_x, torch.sin(probe_x[:, 0]), len(probe_x))
        (-probe_objective).backward()
        torch.cuda.synchronize()
        assert torch.isfinite(probe_variance).all() and (probe_variance > 0).all()
        for parameter in probe_model.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        del probe_model, probe_objective, probe_mean, probe_variance
        print("Full-covariance GPW weights, residual variance and gradients passed.", flush=True)

        from experiments.default_hyperparameters import heaton_defaults
        COMMON = replace(heaton_defaults(device=DEVICE))  # Add explicit overrides here.
        RUNS = [
            ("GPW 8 x 60", replace(COMMON, model="patchwork", patches=8)),
            ("SVGP 480", replace(COMMON, model="svgp")),
            ("GPW 4 x 120", replace(COMMON, model="patchwork", patches=4)),
        ]
        RUN_ROOT = ROOT / "results/heaton" / datetime.now(UTC).strftime("colab_%Y%m%dT%H%M%S%fZ")
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        (RUN_ROOT / "source_snapshot.zip").write_bytes(SOURCE_ZIP)
        (RUN_ROOT / "source_provenance.json").write_text(json.dumps(SOURCE_PROVENANCE, indent=2))
        display(pd.DataFrame([{"run": name, **asdict(config)} for name, config in RUNS]))
        print("Existing parameter bounds:", asdict(HyperparameterBounds()))
        print("Numerical settings:", asdict(NumericalConfig(weight_epsilon=COMMON.weight_epsilon)))
        print("Results:", RUN_ROOT)
        """,
        ),
        cell(
            "markdown",
            """
        ## 3. Run and report each model
        Training-only preprocessing: normalized longitude/latitude, OLS spatial
        trend, and standardized residuals. Test targets are used only for final
        scoring. Intervals include observation noise; latent uncertainty is shown
        separately. Reported uncertainty conditions on fitted trend/hyperparameters.
        Runtime includes the saved progress diagnostics.
        """,
        ),
        cell(
            "code",
            """
        summaries = []
        completed = []
        for name, config in RUNS:
            display(Markdown(f"### {name}"))
            live = display(Markdown("Starting the official MODIS run..."), display_id=True)

            def show_progress(path, epoch, handle=live):
                handle.update(Image(filename=str(path / "progress.png")))

            result = run_heaton(
                config, run_directory=str(RUN_ROOT / name.lower().replace(" ", "_")),
                capture_git=False, on_snapshot=show_progress,
            )
            metrics = json.loads((result / "metrics.json").read_text())
            summaries.append({"model": name, **metrics})
            completed.append(result)
            display(pd.DataFrame([summaries[-1]]).set_index("model").T)
            display(Image(filename=str(result / "maps.png")))
            if config.model == "patchwork":
                display(Image(filename=str(result / "weights.png")))
            display(Markdown(f"Saved complete run: `{result}`"))

        comparison = pd.DataFrame(summaries).set_index("model")
        comparison.to_csv(RUN_ROOT / "comparison.csv")
        display(Markdown("## Completed comparison"))
        display(comparison)
        """,
        ),
        cell(
            "markdown",
            """
        ## 4. Observation-noise effect on NLPD (no retraining)
        Reuses completed runs' saved predictions in Celsius. The existing benchmark
        already includes the learned noise variance beta^-1 (multiplier 1).
        Multiplier 0 removes it; other multipliers are post-hoc sensitivity checks,
        not refitted models or test-selected parameters. GPW's conditional residual
        is already included in the saved latent variance and stays unchanged in
        this noise-only diagnostic. Lower NLPD is better.
        """,
        ),
        cell(
            "code",
            """
        import json
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display

        NOISE_VARIANCE_MULTIPLIERS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

        def noise_nlpd_sensitivity(run_paths, multipliers):
            multipliers = np.asarray(multipliers, dtype=float)
            if multipliers.ndim != 1 or not len(multipliers) or not np.isfinite(multipliers).all() or (multipliers < 0).any():
                raise ValueError("Noise variance multipliers must be finite, nonnegative, and nonempty")
            if not run_paths:
                raise ValueError("No completed runs: execute after at least one run finishes")
            rows = []
            for path in run_paths:
                metrics = json.loads((path / "metrics.json").read_text())
                noise_variance = metrics["noise_std_celsius"] ** 2
                with np.load(path / "predictions.npz") as saved:
                    test = saved["test_mask"]
                    error_squared = (saved["truth"][test] - saved["mean"][test]) ** 2
                    latent = saved["latent_variance"][test]
                    observation = saved["observation_variance"][test]
                np.testing.assert_allclose(observation, latent + noise_variance)
                baseline = np.mean(0.5 * (np.log(2 * np.pi * observation) + error_squared / observation))
                for multiplier in multipliers:
                    variance = latent + multiplier * noise_variance
                    if not np.isfinite(variance).all() or (variance <= 0).any() or not np.isfinite(error_squared).all():
                        raise ValueError(f"Invalid test predictions/variance: {path.name}, multiplier {multiplier}")
                    nlpd = np.mean(0.5 * (np.log(2 * np.pi * variance) + error_squared / variance))
                    rows.append({"model": path.name, "noise_variance_multiplier": multiplier,
                                 "NLPD": float(nlpd), "delta_vs_saved_NLPD": float(nlpd - baseline),
                                 "noise_variance_C2": noise_variance})
            return pd.DataFrame(rows)

        noise_sensitivity = noise_nlpd_sensitivity(completed, NOISE_VARIANCE_MULTIPLIERS)
        display(noise_sensitivity)
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, frame in noise_sensitivity.groupby("model", sort=False):
            ax.plot(frame["noise_variance_multiplier"], frame["NLPD"], marker="o", label=name)
        ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Saved benchmark noise")
        ax.set(xlabel="Multiplier of learned noise variance beta^-1", ylabel="Test NLPD")
        ax.legend()
        fig.tight_layout()
        plt.show()
        noise_sensitivity.to_csv(RUN_ROOT / "noise_nlpd_sensitivity.csv", index=False)
        fig.savefig(RUN_ROOT / "noise_nlpd_sensitivity.png", dpi=160)
        """,
        ),
        cell("markdown", "## 5. Download all results"),
        cell(
            "code",
            """
        from google.colab import files

        archive = shutil.make_archive(str(RUN_ROOT), "zip", root_dir=RUN_ROOT)
        print("Results archive:", archive)
        files.download(archive)
        """,
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": "GPW_Heaton_Colab.ipynb", "provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "gpw_source": source_metadata(),
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = root / "notebooks/GPW_Heaton_Colab.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
