"""Build an embedded-source GPU Colab for the periodic heat learning-rate study."""

import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

from experiments.build_heaton_colab import cell


def build_notebook(root: Path) -> Path:
    paths = [Path("pyproject.toml"), Path("README.md")]
    paths += [Path("experiments") / name for name in ("stitch_line_heat.py", "stitch_line_preview.py",
              "periodic_heat_preview.py", "two_phase_sine.py")]
    paths += sorted(p.relative_to(root) for p in (root / "src/pwgp").rglob("*.py"))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            info = zipfile.ZipInfo(path.as_posix(), date_time=(2026, 10, 29, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, (root / path).read_bytes())
    payload = stream.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    literals = "\n".join(f'    "{encoded[i:i + 100]}"' for i in range(0, len(encoded), 100))
    unpack = f'''import base64, hashlib, importlib, io, os, sys, zipfile
from pathlib import Path

ROOT = Path("/content/gpw_periodic_heat")
os.environ["MPLCONFIGDIR"] = str(ROOT / "results/.matplotlib")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
SOURCE_B64 = (
{literals}
)
SOURCE_ZIP = base64.b64decode(SOURCE_B64)
assert hashlib.sha256(SOURCE_ZIP).hexdigest() == "{digest}", "Embedded source checksum mismatch"
ROOT.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(SOURCE_ZIP)) as bundle:
    for name in bundle.namelist():
        if not (ROOT / name).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Unsafe archive member")
    bundle.extractall(ROOT)
os.chdir(ROOT)
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
importlib.invalidate_caches()
print("Source verified:", "{digest}")
'''
    cells = [
        cell("markdown", """
        # GPW: Periodic Heat and Movable Stitch Lines

        Select a GPU runtime, then Run all. Restart first if an older notebook
        already loaded these model classes. No GitHub or separate files needed.

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
        cell("markdown", "## 1. Embedded source and dependencies"),
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
        HEAT = PreviewConfig(
            cells=80, diffusivity=(0.6, 0.025), angle_pairs=((0, 90),),
            periods=(4, 4), baseline=1.0, amplitude=0.8, time=0.015,
            time_steps=40, supports_per_patch=28, support_margin=0.02,
            stitches=10, q_rank=12, lengthscale=0.12, outputscale=1.0,
            noise_std=0.03, weight_epsilon=0.01, seed=31,
        )
        LAYOUTS = LayoutConfig(
            heat=HEAT, line_centers=((0.5, 0.5), (0.56, 0.5), (0.5, 0.5), (0.5, 0.5)),
            line_angles=(90, 90, 45, 0),
            line_names=("Vertical", "Vertical, shifted", "Diagonal", "Horizontal"),
            line_margin=0.05, candidate_points=10000,
        )
        CONFIG = RunConfig(
            layouts=LAYOUTS, observations=500, epochs=350, warmup_epochs=40,
            learning_rate_q=0.008, learning_rate_supports=0.004,
            learning_rate_line=0.003, learning_rate_hyperparameters=0.003,
            snapshot_every=25, prediction_batch_size=256, threads=2, device=DEVICE,
        )
        if not STITCH_LEARNING_RATES or any(not np.isfinite(v) or v <= 0 for v in STITCH_LEARNING_RATES):
            raise ValueError("STITCH_LEARNING_RATES must contain finite positive values")
        if torch.device(DEVICE).type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime. This notebook does not fall back to CPU")
        print({"GPU": torch.cuda.get_device_name(DEVICE), "torch": torch.__version__,
               "CUDA": torch.version.cuda, "dtype": "float64"})
        RUN_ROOT = ROOT / "results/periodic_heat_colab" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        (RUN_ROOT / "source_snapshot.zip").write_bytes(SOURCE_ZIP)
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
        "language_info": {"name": "python", "version": "3.12"}, "gpw_source_sha256": digest},
        "nbformat": 4, "nbformat_minor": 5}
    target = root / "notebooks/GPW_Periodic_Heat_Colab.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
