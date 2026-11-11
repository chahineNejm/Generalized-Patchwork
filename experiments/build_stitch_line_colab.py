"""Build a GitHub-backed GPU Colab for the periodic heat learning-rate study."""

import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata


def build_notebook(root: Path) -> Path:
    unpack = github_bootstrap("/content/gpw_periodic_heat")
    cells = [
        cell("markdown", """
        # GPW: Periodic Heat and Movable Stitch Lines

        Select a GPU runtime, then Run all. Restart first if an older notebook
        already loaded these model classes. Loads the latest main from GitHub;
        the resolved commit is saved. No model source is embedded.

        Edit STITCH_LEARNING_RATES in the settings cell to test one or several
        values. Each value runs all four starts from scratch, with identical data
        and seeds. Default 0.003 reproduces the local configuration.

        Heat stays horizontal on the left and vertical on the right. Initial
        stitches form a vertical, offset vertical, diagonal or horizontal line.
        Each submodel starts on its own side; supports are free to cross later.
        Each stitch then moves independently; there is no line constraint. Kernels are two
        ARD RBFs, and uncertainty includes the full stitched residual R.

        This is a posterior-adaptive likelihood-minus-KL objective, not a
        fixed-model ELBO guarantee. The learned stitches need not recover the material
        boundary. Heating is spatially repeating; the exterior is insulated.
        Finite-element heat simulation runs on CPU; GP training uses float64 GPU.
        No CPU fallback, solver retries or automatic parameter changes.
        Live views include each kernel's x/y lengthscales, output variance and
        shared noise SD, from epoch 0 onward. Dashed markers show kernel warm-up.
        """),
        cell("markdown", "## 1. GitHub source and dependencies"),
        cell("code", unpack),
        cell("code", '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--editable", "/content/gpw_periodic_heat[heat]"], check=True)'''),
        cell("markdown", "## 2. Settings"),
        cell("code", """
        import gc, json, shutil, zipfile
        from dataclasses import asdict, replace
        from datetime import UTC, datetime
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import Image, Markdown, display
        from google.colab import files
        from experiments.periodic_heat_preview import PreviewConfig
        from experiments.stitch_line_preview import LayoutConfig, preview
        from experiments.stitch_line_heat import RunConfig, build_model, run
        from pwgp.linalg import cholesky_solve

        STITCH_LEARNING_RATES = [0.003]  # e.g. [0.003, 0.01, 0.03]
        DEVICE = "cuda:0"
        from experiments.default_hyperparameters import periodic_heat_defaults
        CONFIG = replace(periodic_heat_defaults(device=DEVICE))  # Explicit overrides here.
        LAYOUTS = CONFIG.layouts
        HEAT = LAYOUTS.heat
        if not STITCH_LEARNING_RATES or any(not np.isfinite(v) or v <= 0 for v in STITCH_LEARNING_RATES):
            raise ValueError("STITCH_LEARNING_RATES must contain finite positive values")
        if torch.device(DEVICE).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime. This notebook does not fall back to CPU")
        print({"GPU": torch.cuda.get_device_name(DEVICE), "torch": torch.__version__,
               "CUDA": torch.version.cuda, "dtype": "float64"})
        RUN_ROOT = ROOT / "results/periodic_heat_colab" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        (RUN_ROOT / "source_snapshot.zip").write_bytes(SOURCE_ZIP)
        (RUN_ROOT / "source_provenance.json").write_text(json.dumps(SOURCE_PROVENANCE, indent=2))
        display(pd.DataFrame([{"stitch_learning_rate": v, "layouts": len(LAYOUTS.line_angles),
                              "epochs": CONFIG.epochs, "warmup": CONFIG.warmup_epochs}
                              for v in STITCH_LEARNING_RATES]))
        """),
        cell("markdown", "## 3. GPU checks and initial layouts"),
        cell("code", """
        for batch, size in [(2, 28), (500, 1), (1, 56)]:
            matrix = (torch.eye(size, device=DEVICE, dtype=torch.float64) + 0.01).repeat(batch, 1, 1)
            matrix.requires_grad_(True)
            rhs = torch.ones(batch, size, 1, device=DEVICE, dtype=torch.float64)
            answer = cholesky_solve(rhs, torch.linalg.cholesky(matrix))
            torch.testing.assert_close(matrix @ answer, rhs)
            answer.square().mean().backward()
            assert torch.isfinite(matrix.grad).all()
        del matrix, rhs, answer
        probe_heat = replace(HEAT, supports_per_patch=8, stitches=4, q_rank=2)
        probe_config = replace(CONFIG, layouts=replace(LAYOUTS, heat=probe_heat))
        for index in range(len(LAYOUTS.line_angles)):
            probe = build_model(probe_config, index)
            x = 0.05 + 0.9 * torch.rand(16, 2, device=DEVICE, dtype=torch.float64)
            loss = -probe.elbo_batch(x, torch.sin(8 * x[:, 0]), len(x))[0]
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
            del probe, x, loss
        torch.cuda.synchronize(DEVICE)
        gc.collect()
        torch.cuda.empty_cache()
        print("GPU factorization, solves, GPW and independent stitch gradients passed")
        starting = preview(LAYOUTS, directory=RUN_ROOT / "initial_layouts")
        display(Image(filename=str(starting / "initial_layouts.png")))
        """),
        cell("markdown", "## 4. Train, visualize and save each run"),
        cell("code", """
        summaries = []
        for rate_index, rate in enumerate(STITCH_LEARNING_RATES):
            config = replace(CONFIG, learning_rate_line=rate)
            directory = RUN_ROOT / f"rate_{rate_index}_{rate:g}"
            display(Markdown(f"## Stitch learning rate: {rate:g}"))
            live = display(Markdown("Starting..."), display_id=True)

            def show_progress(folder, epoch, handle=live):
                handle.update(Image(filename=str(folder / "progress.png")))

            def completed(folder, metrics, rate=rate, rate_index=rate_index):
                summaries.append({"stitch_learning_rate": rate, **metrics})
                pd.DataFrame(summaries).to_csv(RUN_ROOT / "comparison.csv", index=False)
                display(pd.DataFrame([summaries[-1]]).T)
                display(Image(filename=str(folder / "kernel_evolution.png")))
                parent = folder.parent
                shared = [parent / name for name in ("data.npz", "config.json", "provenance.json", "metrics.json")]
                archive = RUN_ROOT / f"rate_{rate_index}_{folder.name}.zip"
                with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
                    bundle.writestr("source_snapshot.zip", SOURCE_ZIP)
                    bundle.writestr("source_provenance.json", json.dumps(SOURCE_PROVENANCE, indent=2))
                    for path in shared + sorted(p for p in folder.rglob("*") if p.is_file()):
                        bundle.write(path, path.relative_to(parent))
                print("Saved:", archive)
                files.download(str(archive))

            result = run(config, directory=directory, capture_git=False,
                         on_snapshot=show_progress, on_complete=completed)
            for name in ("points.png", "fits.png", "training.png"):
                display(Image(filename=str(result / name)))
        display(pd.DataFrame(summaries)[["stitch_learning_rate", "layout", "rmse_standardized",
                "nlpd_standardized", "coverage_95", "mean_stitch_displacement"]])
        """),
        cell("markdown", "## 5. Download the complete comparison"),
        cell("code", """
        archive = shutil.make_archive(str(RUN_ROOT), "zip", root_dir=RUN_ROOT)
        files.download(archive)
        """),
    ]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU",
        "colab": {"name": "GPW_Periodic_Heat_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}, "gpw_source": source_metadata()},
        "nbformat": 4, "nbformat_minor": 5}
    target = root / "notebooks/GPW_Periodic_Heat_Colab.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
