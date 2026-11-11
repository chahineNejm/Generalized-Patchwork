"""Build the GitHub-backed OISST experiment launcher."""

import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata


def build_notebook(root):
    cells = [
        cell("markdown", """
        # OISST model comparison

        Select a GPU runtime and Run all. Code is installed from the latest main
        of chahineNejm/Generalized-Patchwork; the resolved commit is saved.
        Defaults live in experiments/default_hyperparameters.py. Edit overrides
        below, not model definitions. No model or dataset implementation is embedded.
        """),
        cell("markdown", "## 1. Load code and dependencies"),
        cell("code", github_bootstrap("/content/gpw_oisst")),
        cell("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--editable", str(ROOT) + "[data,ukhousing]"], check=True)'''),
        cell("markdown", "## 2. Defaults and explicit overrides"),
        cell("code", """
        from dataclasses import replace
        from datetime import UTC, datetime
        from IPython.display import Image, display
        from google.colab import files
        import pandas as pd
        import torch
        from experiments.default_hyperparameters import oisst_defaults
        from experiments.oisst_benchmark import run_comparison
        from experiments.run_selection import select_runs

        DATA, RUNS = oisst_defaults(device="cuda:0")
        MODELS = ["SVGP", "GPW", "ProSpar", "VNNGP", "Standard Patchwork"]
        # Optional overrides, e.g. {"GPW": {"epochs": 150, "warmup_epochs": 30}}.
        OVERRIDES = {}
        DATA_OVERRIDES = {}  # e.g. {"save_inducing_positions": False}
        RUNS = select_runs(RUNS, models=MODELS, overrides=OVERRIDES)
        DATA = replace(DATA, **DATA_OVERRIDES)
        RUN_ROOT = ROOT / "results/oisst" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        if not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime; there is no CPU fallback")
        display(pd.DataFrame([{"run": name, **config.__dict__} for name, config in RUNS]))
        print(DATA)
        """),
        cell("markdown", "## 3. Run, visualize and save"),
        cell("code", """
        live = display("Starting OISST comparison...", display_id=True)

        def show_progress(path, epoch):
            live.update(Image(filename=str(path / "visuals/progress.png")))
            display(Image(filename=str(path / "visuals/training.png")))

        def show_completed(path, archive, metrics):
            display(pd.DataFrame([metrics]).T)
            display(Image(filename=str(path / "visuals/maps.png")))
            display(Image(filename=str(path / "visuals/metrics.png")))
            display(Image(filename=str(path / "visuals/hyperparameters.png")))
            metadata = json.loads((path / "model_metadata.json").read_text())
            if metadata["inducing_positions_saved"]:
                display(Image(filename=str(path / "visuals/inducing_positions.png")))
            files.download(str(archive))

        result = run_comparison(
            DATA, RUNS, directory=RUN_ROOT, data_root=ROOT / "data",
            source_provenance=SOURCE_PROVENANCE, source_zip=SOURCE_ZIP,
            on_snapshot=show_progress, on_complete=show_completed,
        )
        display(pd.read_json(result / "comparison.json"))
        print("Results:", result)
        """),
        cell("markdown", "## 4. Download the complete comparison"),
        cell("code", """
        import shutil
        archive = shutil.make_archive(str(RUN_ROOT), "zip", root_dir=RUN_ROOT)
        files.download(archive)
        """),
    ]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU",
        "colab": {"name": "OISST_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "gpw_source": source_metadata()}, "nbformat": 4, "nbformat_minor": 5}
    target = Path(root) / "notebooks/OISST_Colab.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
