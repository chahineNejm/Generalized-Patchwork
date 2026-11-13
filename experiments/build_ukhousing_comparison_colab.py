"""Build the approved ten-configuration UKHousing launcher."""

import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata


def build_notebook(root):
    cells = [
        cell("markdown", """
        # UKHousing: matched inducing budgets

        Select a GPU runtime and Run all. Models/data load from latest main;
        no implementations are embedded. Setup can be rerun without reconnecting.
        Every package-dependent stage runs in a fresh worker.

        SVGP 1,024; GPW and ProSpar each 8 x 128 and 4 x 256; VNNGP K=32/128
        with one fixed inducing location per training observation. Standard
        Patchwork: checkerboard 4 x 2, GPW-center Voronoi and 8-leaf PC tree,
        each using the 8-patch GPW initialization's 1,024 observed targets.
        These are reduced-data baselines, not equal-information comparisons.

        Approved protocol: ARD RBF, float64, 100 epochs, 20 kernel/noise warm-up
        for variational models. Standard Patchwork uses 100 joint-likelihood
        updates without warm-up. Other existing UKHousing settings are retained.
        ProSpar starts with temperatures=0.1, exponent=2, nuggets=0.01, prior_jitter=0.

        The 2018-flats reconstruction is not the verified VNNGP snapshot.
        This is NOT a reproduction: that paper used Matern-5/2, other initial
        values, optimizer sweeps and three seeds. Here seed=31, split=64/16/20,
        training-only standardization of log prices, fixed geographic ROI.
        Source/data checksums, full configurations, checkpoints, predictions,
        histories, metrics and maps are archived per completed run.
        Failures stop downstream execution; no recovery or device substitution.
        """),
        cell("markdown", "## 1. Source and dependencies"),
        cell("code", github_bootstrap("/content/gpw_ukhousing_comparison", isolated=True),
             stage="checkout", invalidates=("install", "settings", "data", "training", "download")),
        cell("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--editable", str(ROOT) + "[ukhousing]"], check=True)
import runpy
run_worker = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]

def call_worker(action, *, on_event=None, **values):
    return run_worker(ROOT, {"action": action, "source_provenance": SOURCE_PROVENANCE, **values},
                      module="experiments.ukhousing_worker", on_event=on_event)''',
             stage="install", requires=("checkout",), invalidates=("settings", "data", "training", "download")),
        cell("markdown", """
        ## 2. Select models and override hyperparameters

        GPW denotes 8 x 128. Remove names to select fewer runs; order is preserved.
        All effective fields are displayed and saved. Example:
        OVERRIDES = {"GPW": {"epochs": 50, "warmup_epochs": 10}}.
        Baselines reference GPW even when it is deselected, not GPW 4 x 256.
        ProSpar overrides use an explicit prospar dictionary with all four fields.
        """),
        cell("code", '''from datetime import UTC, datetime
from IPython.display import Image, Markdown, display
from google.colab import files
import pandas as pd

MODELS = ["SVGP", "GPW", "GPW 4 x 256", "ProSpar 8 x 128", "ProSpar 4 x 256",
          "VNNGP K32", "VNNGP K128", "Standard Patchwork",
          "Standard Patchwork (GPW regions)", "Standard Patchwork (PC tree)"]
OVERRIDES = {}
DATA = {"seed": 31, "train_fraction": 0.64, "validation_fraction": 0.16}
settings = call_worker("settings", device="cuda:0", models=MODELS, overrides=OVERRIDES)
RUNS = settings["runs"]
WORK_ROOT = ROOT / "results/ukhousing_comparison"
display(pd.DataFrame([{"run": row["name"], **row["config"]} for row in RUNS]))''',
             stage="settings", requires=("install",), invalidates=("training", "download")),
        cell("markdown", """
        ## 3. Download and prepare data once

        Keep this stage unchanged when only model settings change. Training
        verifies prepared configuration/checksums; it never silently rebuilds data.
        Rerun this cell explicitly to prepare a new immutable split directory.
        """),
        cell("code", '''PREPARED_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("prepared_%Y%m%dT%H%M%S%fZ")
prepared = call_worker("prepare", data=DATA, directory=str(PREPARED_DIRECTORY))
print("Prepared:", prepared["directory"])''', stage="data", requires=("install", "settings"),
             invalidates=("training", "download")),
        cell("markdown", """
        ## 4. Train selected configurations

        Edit MODELS/OVERRIDES and rerun settings, then this cell, to run only the
        selected configurations on the verified split. Each invocation trains
        from scratch in a new batch. Completed runs remain available after a failure.
        One live plot updates in place; final maps and metrics display per run.
        Worker exit releases GPU memory before the next notebook action; the
        existing runner also releases each model/cache before its archive event.
        """),
        cell("code", '''live = display("Starting UKHousing comparison...", display_id=True)

def show_event(event):
    path = Path(event["directory"])
    if event["kind"] == "snapshot":
        live.update(Image(filename=str(path / "progress.png")))
    elif event["kind"] == "complete":
        display(Markdown("### " + event["name"]))
        display(pd.DataFrame([event["metrics"]]).T)
        display(Image(filename=str(path / "maps.png")))
        files.download(event["archive"])
    else:
        raise ValueError(f"Unknown worker event: {event['kind']}")

BATCH_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("batch_%Y%m%dT%H%M%S%fZ")
completed = call_worker("train", data=DATA, runs=RUNS, prepared_directory=str(PREPARED_DIRECTORY),
                        directory=str(BATCH_DIRECTORY), on_event=show_event)
display(pd.read_csv(Path(completed["directory"]) / "comparison.csv"))
print("Results:", completed["directory"])''', stage="training", requires=("install", "settings", "data"),
             invalidates=("download",)),
        cell("markdown", "## 5. Download complete batch"),
        cell("code", '''import shutil
archive = shutil.make_archive(str(BATCH_DIRECTORY), "zip", root_dir=BATCH_DIRECTORY)
files.download(archive)''', stage="download", requires=("training",)),
        cell("markdown", """
        ## References
        - [VNNGP paper](https://proceedings.mlr.press/v162/wu22h.html)
        - [Patchwork Kriging](https://jmlr.org/papers/v19/17-042.html)
        - [ProSpar derivation](https://arxiv.org/abs/2311.08752v1)
        - [UKHousing preparation reference](https://github.com/cunningham-lab/hipgp/blob/master/experiments-hip-gp/uk_housing_data.py)
        """),
    ]
    target = Path(root) / "notebooks/UKHousing_Comparison_Colab.ipynb"
    target.write_text(json.dumps({"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"name": target.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "gpw_source": source_metadata()},
        "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
