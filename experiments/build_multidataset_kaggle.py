"""Build a source-backed, explicitly gated Kaggle experiment notebook."""

import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata


def build_notebook(root):
    datasets = ["oisst", "ukhousing", "modis", "california", "tco", "tco_l2", "icethick", "protein", "sarcos", "flight"]
    datasets.extend(f"{name}_d{dimension}_n{count}" for name in ("michalewicz", "g_function")
                    for dimension in (3, 5, 10) for count in (50000, 100000))
    cells = [
        cell("markdown", """
        # GPW multi-dataset experiments: Kaggle

        Enable Internet and a GPU in Kaggle notebook settings. Attach the private
        `gpw-kaggle-inputs` dataset containing airline.npz, airline.json and
        icethk.txt from the supplied input bundle. Set INPUT_ROOT to its exact
        mounted directory if it differs. All three files are checksum-verified;
        no file search, unpickling or substitution occurs. The NPZ is the explicitly
        approved one-time conversion of your trusted pickle; its original is unchanged.
        Other real datasets download from the approved, pinned sources.

        Code loads latest main of chahineNejm/Generalized-Patchwork. Every stage
        uses a fresh worker, so setup can be rerun without restarting the kernel.
        Model/data code is never embedded in this notebook.

        All 22 dataset configurations are listed initially. Training is OFF until
        RUN_TRAINING=True. Review the complete plan first: all presets together
        currently mean 179 model runs. Three epochs do not make graph construction
        or large inducing systems cheap. VNNGP uses every training location.
        Select a small explicit DATASETS/MODELS subset for your first check.
        No point budget, dataset size, device or numerical value is reduced silently.

        New real and synthetic protocols use seeded 80/20; existing OISST/UKHousing
        protocols and official MODIS split/training-only detrending are retained.
        All models keep their approved ARD RBF/initialization/optimizer settings.
        VNNGP compares K32/K128. Standard Patchwork uses all input dimensions and
        paper-style balanced PC trees; the 2D datasets also expose checkerboard
        and initial-GPW-center Voronoi variants where configured. Reduced-data
        Standard Patchwork uses exactly the primary GPW initial selected targets,
        while GPW/SVGP/ProSpar/VNNGP train on all observations.

        ProSpar is deliberately smaller than its paper: matched budgets here are
        512/1024/2048, versus the paper's 200x25=5000 points. Actual/paper capacities
        are recorded with every ProSpar result. These are controlled comparisons,
        not reproductions of every paper's kernel, split or optimization protocol.

        Precipitation and satellite drag are explicitly DEFERRED: verified source
        data are unavailable. They are not silently skipped from a selected plan.
        Synthetic functions use the approved six dimension/sample-size settings
        per function; their 80/20 test sizes differ from the paper's fixed 25000.

        Results, plan, worker logs, tracebacks and per-run ZIPs are written under
        /kaggle/working/gpw_results. Every run saves source/data provenance, full
        effective configuration, initial/final checkpoints, predictions, metrics,
        timing, training/kernel histories and optional full-dimensional positions.
        Metrics include both explicitly labeled standardized and original-target
        scales; do not compare values with different units as the same metric.
        High-dimensional location plots are labeled two-feature projections.
        Save a Kaggle version with outputs to retain them beyond the session.
        """),
        cell("markdown", "## 1. Latest source and dependencies"),
        cell("code", github_bootstrap("/kaggle/working/gpw_source", isolated=True), stage="checkout",
             invalidates=("install", "settings", "preflight", "data", "training", "archive")),
        cell("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--editable", str(ROOT) + "[data,ukhousing,benchmark]"], check=True)
import runpy
run_worker = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]
WORK_ROOT = Path("/kaggle/working/gpw_results")
DATA_ROOT = Path("/kaggle/working/gpw_data")

def call_worker(action, *, on_event=None, **values):
    return run_worker(ROOT, {"action": action, "source_provenance": SOURCE_PROVENANCE, **values},
        module="experiments.multidataset_worker", jobs_directory=WORK_ROOT / "worker_jobs", on_event=on_event)''',
             stage="install", requires=("checkout",), invalidates=("settings", "preflight", "data", "training", "archive")),
        cell("markdown", """
        ## 2. Dataset/model selection and all call-time overrides

        Edit DATASETS to choose/order datasets. MODELS_BY_DATASET selects/orders
        model names shown in AVAILABLE_MODELS; an absent entry means the complete
        explicitly displayed preset for that dataset. Unknown names raise.
        MODEL_OVERRIDES is {dataset: {model_name: {field: value}}}.
        DATA_OVERRIDES is {dataset: {field: value}}; no source/schema relabeling.

        Example short selection: DATASETS=["california"] and
        MODELS_BY_DATASET={"california": ["SVGP", "GPW"]}.
        For a long run, explicitly change EPOCHS and WARMUP_EPOCHS; no automatic
        warm-up shrinking occurs. Rerunning training always starts fresh.
        """),
        cell("code", f'''from datetime import UTC, datetime
from IPython.display import Image, Markdown, display
import pandas as pd

INPUT_ROOT = "/kaggle/input/gpw-kaggle-inputs"
DATASETS = {datasets!r}
MODELS_BY_DATASET = {{}}
MODEL_OVERRIDES = {{}}
DATA_OVERRIDES = {{}}
EPOCHS = 3
WARMUP_EPOCHS = 1
SAVE_INDUCING_POSITIONS = True

settings = call_worker("settings", device="cuda:0", input_root=INPUT_ROOT, datasets=DATASETS,
    models=MODELS_BY_DATASET, model_overrides=MODEL_OVERRIDES, data_overrides=DATA_OVERRIDES,
    epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, save_inducing_positions=SAVE_INDUCING_POSITIONS)
PLAN, PLAN_SHA256 = settings["plan"], settings["plan_sha256"]
AVAILABLE_MODELS = settings["available"]
display(pd.DataFrame(settings["summary"]))
print("Total selected model runs:", settings["total_runs"])
print("Total training epochs across runs:", settings["total_epochs"])
print("Deferred datasets:", PLAN["deferred"])
display(pd.DataFrame([{{"dataset": row["dataset"], "model": row["name"], **row["config"]}} for row in PLAN["runs"]]))
print("Available model names:", AVAILABLE_MODELS)''', stage="settings", requires=("install",),
             invalidates=("preflight", "training", "archive")),
        cell("markdown", "## 3. Check attached inputs, requested GPU and available disk"),
        cell("code", '''PREFLIGHT_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("preflight_%Y%m%dT%H%M%S%fZ")
preflight = call_worker("preflight", plan=PLAN, plan_sha256=PLAN_SHA256, directory=str(PREFLIGHT_DIRECTORY))
print(preflight)''', stage="preflight", requires=("install", "settings"), invalidates=("training", "archive")),
        cell("markdown", """
        ## 4. Prepare selected datasets once or explicitly load prepared files

        Leave this stage untouched when only model settings change. Use
        PREPARED_ACTION="load" and an exact PREPARED_PATH to reuse saved arrays;
        configuration/checksum mismatches raise without rebuilding or filling.
        """),
        cell("code", '''PREPARED_ACTION = "prepare"
PREPARED_PATH = None
if PREPARED_ACTION == "prepare":
    PREPARED_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("prepared_%Y%m%dT%H%M%S%fZ")
    prepared = call_worker("prepare", plan=PLAN, plan_sha256=PLAN_SHA256,
        directory=str(PREPARED_DIRECTORY), data_root=str(DATA_ROOT))
elif PREPARED_ACTION == "load":
    if PREPARED_PATH is None:
        raise ValueError("Set PREPARED_PATH explicitly for verified prepared-data reuse")
    PREPARED_DIRECTORY = Path(PREPARED_PATH)
    prepared = call_worker("load", plan=PLAN, plan_sha256=PLAN_SHA256, directory=str(PREPARED_DIRECTORY))
else:
    raise ValueError("PREPARED_ACTION must be prepare or load")
counts = []
for spec in PLAN["data"]:
    import numpy as np
    with np.load(PREPARED_DIRECTORY / spec["name"] / "split.npz", allow_pickle=False) as rows:
        counts.append({"dataset": spec["name"], **{label: len(rows[label]) for label in ("train", "validation", "test")}})
display(pd.DataFrame(counts))
print("Prepared datasets:", prepared["directory"])''', stage="data", requires=("install", "settings", "preflight"),
             invalidates=("training", "archive")),
        cell("markdown", """
        ## 5. Explicitly authorize and run the displayed plan

        RUN_TRAINING=False intentionally raises before any model starts. Set it
        True only after reviewing selection, counts and capacities above. There
        is no endless sweep: exactly the selected configurations and epoch counts
        run. Any failure stops subsequent work with a full traceback. Interrupting
        the notebook terminates the worker. Completed results/ZIPs remain on disk.
        Changing the selection/hyperparameters requires rerunning settings and
        preflight; unchanged prepared data can be reused explicitly.
        """),
        cell("code", '''RUN_TRAINING = False
if RUN_TRAINING is not True:
    raise RuntimeError("Training disabled. Review the plan, then explicitly set RUN_TRAINING=True")
live = display("Starting selected experiments...", display_id=True)

def show_event(event):
    path = Path(event["directory"])
    if event["kind"] == "snapshot":
        live.update(Image(filename=str(path / "visuals/training.png")))
    elif event["kind"] == "complete":
        display(Markdown("### " + event["dataset"] + " / " + event["name"]))
        display(pd.DataFrame([event["metrics"]]).T)
        display(Image(filename=str(path / "visuals/predictions.png")))
        display(Image(filename=str(path / "visuals/hyperparameters.png")))
        metadata = json.loads((path / "model_metadata.json").read_text())
        if metadata["inducing_positions_saved"]:
            display(Image(filename=str(path / "visuals/inducing_positions.png")))
        print("Saved completed-run archive:", event["archive"])
    else:
        raise ValueError(f"Unknown worker event: {event['kind']}")

BATCH_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("experiments_%Y%m%dT%H%M%S%fZ")
result = call_worker("train", plan=PLAN, plan_sha256=PLAN_SHA256,
    prepared_directory=str(PREPARED_DIRECTORY), directory=str(BATCH_DIRECTORY),
    execute_training=RUN_TRAINING, on_event=show_event)
display(pd.read_csv(Path(result["directory"]) / "comparison.csv"))
print("Complete collection:", result["directory"])''', stage="training", requires=("install", "settings", "preflight", "data"),
             invalidates=("archive",)),
        cell("markdown", "## 6. Archive the completed collection"),
        cell("code", '''ARCHIVE_PATH = WORK_ROOT / datetime.now(UTC).strftime("all_results_%Y%m%dT%H%M%S%fZ.zip")
archived = call_worker("archive", plan=PLAN, plan_sha256=PLAN_SHA256,
    directory=str(BATCH_DIRECTORY), destination=str(ARCHIVE_PATH))
from IPython.display import FileLink
display(FileLink(archived["archive"]))
print("Worker requests/logs/failures:", WORK_ROOT / "worker_jobs")
print("Save a Kaggle version with outputs to preserve all files.")''', stage="archive", requires=("training",)),
    ]
    target = Path(root) / "notebooks/All_Datasets_Kaggle.ipynb"
    target.write_text(json.dumps({"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "gpw_source": source_metadata()},
        "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
