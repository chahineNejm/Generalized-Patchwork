"""Build the approved source-backed GPW stitching/rank Kaggle study."""

import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata
from experiments.default_hyperparameters import gpw_ablation_defaults


def build_notebook(root):
    defaults = gpw_ablation_defaults(device="cuda:0",
        input_root="/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs")
    cells = [
        cell("markdown", """
        # GPW stitching and variational-rank ablations

        Enable Internet and a GPU. Attach the existing gpw-kaggle-inputs dataset;
        only icethk.txt is needed from it (FLIGHT is not selected). INPUT_ROOT
        below is its exact mounted directory. Other datasets use the existing
        approved download/preprocessing protocols, with no source substitution.

        Default study: 11 datasets x 5 seeds x ranks {0,32} x stitch modes
        {none,fixed,learned} = 330 runs. Every dataset uses 1024 inducing points,
        eight patches of128. Model seeds are31-35. Data splits and synthetic
        realizations retain their existing seed31; model seeds vary initialization
        and training only. Six variants per dataset/seed share exactly the same
        selected training rows, support coordinates, centers, candidate graph and
        candidate stitches. Their geometry fingerprint is checked before training.

        None removes stitch constraints, not the GPW model. Fixed freezes only
        stitch locations. Learned optimizes them. Supports and separate local
        ARD RBF kernels train in all modes; kernels/noise are frozen for the first
        20 epochs, not supports/stitches. Rank0 removes V, not all dependence from
        stitching. Existing prior-dependent q initialization is retained, so the
        initial variational covariance need not match between ranks/modes.
        No normalized-product modification or shared-kernel comparison is included.

        At most100 epochs,20 warm-up; existing training-loss stopping uses
        patience10 and relative improvement0.001. Evaluation uses the last state,
        without checkpoint restoration. All parameters and effective overrides
        are saved. Epochs/warm-up are editable; for an explicit smoke test choose
        EPOCHS=3, WARMUP_EPOCHS=1. No real dataset is silently made smaller.

        The full study is not intended to fit a single GPU session. Select
        DATASETS and SEEDS explicitly to divide it into finite batches. Previously
        measured T4 times suggest roughly50-90 GPU-hours, not a runtime guarantee.
        Rerunning training starts fresh in a new directory, never resumes/skips.
        Prepared arrays can be reused only through the explicit verified load stage.

        Every completed run saves predictions, all metrics, geometry, parameter
        histories, position snapshots, checkpoints, environment, source/data
        provenance and its own ZIP before the next run begins. Failures stop the
        batch and save the traceback; completed artifacts remain. Output files
        are session-local until saved in a Kaggle version/downloaded externally.

        Latest main is loaded from the repository and every stage runs in a fresh
        worker, without reconnecting the notebook kernel. The notebook contains
        only configuration and calls, not model/data implementations.
        """),
        cell("markdown", "## 1. Load latest source and dependencies"),
        cell("code", github_bootstrap("/kaggle/working/gpw_source", isolated=True), stage="checkout",
             invalidates=("install", "settings", "preflight", "data", "training", "archive")),
        cell("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--editable", str(ROOT) + "[data,ukhousing,benchmark]"], check=True)
import runpy
run_worker = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]
WORK_ROOT = Path("/kaggle/working/gpw_ablation_results")
DATA_ROOT = Path("/kaggle/working/gpw_data")

def call_worker(action, *, on_event=None, **values):
    return run_worker(ROOT, {"action": action, "source_provenance": SOURCE_PROVENANCE, **values},
        module="experiments.multidataset_worker", jobs_directory=WORK_ROOT / "worker_jobs", on_event=on_event)''',
             stage="install", requires=("checkout",), invalidates=("settings", "preflight", "data", "training", "archive")),
        cell("markdown", """
        ## 2. Select the finite batch and review all effective settings

        DATASETS and SEEDS select the current batch. The full defaults give330
        runs; one seed gives66. MODEL_OVERRIDES={dataset:{field:value}} applies
        identically to every selected variant for that dataset, for example
        {"oisst":{"learning_rate_structure":0.001}}. Factorial controls have
        dedicated variables and cannot be overridden per condition. Unknown or
        conflicting fields raise. DATA_OVERRIDES uses the existing dataset schema;
        data seeds/source/dimension cannot be changed through this study.
        The shared-kernel study remains planned, not silently added.
        """),
        cell("code", f'''from datetime import UTC, datetime
from IPython.display import FileLink, Image, Markdown, display
import pandas as pd

INPUT_ROOT = "/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs"
DATASETS = {defaults["datasets"]!r}
SEEDS = {defaults["seeds"]!r}
RANKS = {defaults["ranks"]!r}
STITCH_MODES = {defaults["stitch_modes"]!r}
INDUCING_POINTS = {defaults["inducing_points"]}
PATCHES = {defaults["patches"]}
EPOCHS = {defaults["epochs"]}
WARMUP_EPOCHS = {defaults["warmup_epochs"]}
EARLY_STOPPING = {defaults["early_stopping"]!r}
MODEL_OVERRIDES = {{}}
DATA_OVERRIDES = {{}}
SAVE_INDUCING_POSITIONS = True

settings = call_worker("settings", study="gpw_stitch_rank", device="cuda:0", input_root=INPUT_ROOT,
    datasets=DATASETS, seeds=SEEDS, ranks=RANKS, stitch_modes=STITCH_MODES,
    inducing_points=INDUCING_POINTS, patches=PATCHES, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
    early_stopping=EARLY_STOPPING, model_overrides=MODEL_OVERRIDES, data_overrides=DATA_OVERRIDES,
    save_inducing_positions=SAVE_INDUCING_POSITIONS)
PLAN, PLAN_SHA256 = settings["plan"], settings["plan_sha256"]
display(pd.DataFrame(settings["summary"]))
print("Selected runs:", settings["total_runs"], "Maximum total epochs:", settings["total_epochs"])
print("Study:", PLAN["study"])
display(pd.DataFrame([{{"dataset": r["dataset"], "name": r["name"], **r["config"]}} for r in PLAN["runs"]]))''',
             stage="settings", requires=("install",), invalidates=("preflight", "training", "archive")),
        cell("markdown", "## 3. Check inputs, requested GPU and available disk"),
        cell("code", '''PREFLIGHT_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("preflight_%Y%m%dT%H%M%S%fZ")
print(call_worker("preflight", plan=PLAN, plan_sha256=PLAN_SHA256, directory=str(PREFLIGHT_DIRECTORY)))''',
             stage="preflight", requires=("install", "settings"), invalidates=("training", "archive")),
        cell("markdown", """
        ## 4. Prepare once, or explicitly load verified prepared arrays

        Keep this stage untouched when changing only the training selection.
        To reuse arrays, set PREPARED_ACTION="load" and PREPARED_PATH to the exact
        prepared folder, including one from the previous multi-dataset notebook
        if configurations/checksums match. Loading never rebuilds failed data.
        """),
        cell("code", '''PREPARED_ACTION = "prepare"
PREPARED_PATH = None
if PREPARED_ACTION == "prepare":
    PREPARED_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("prepared_%Y%m%dT%H%M%S%fZ")
    prepared = call_worker("prepare", plan=PLAN, plan_sha256=PLAN_SHA256,
        directory=str(PREPARED_DIRECTORY), data_root=str(DATA_ROOT))
elif PREPARED_ACTION == "load":
    if PREPARED_PATH is None:
        raise ValueError("Set PREPARED_PATH explicitly for verified data reuse")
    PREPARED_DIRECTORY = Path(PREPARED_PATH)
    prepared = call_worker("load", plan=PLAN, plan_sha256=PLAN_SHA256, directory=str(PREPARED_DIRECTORY))
else:
    raise ValueError("PREPARED_ACTION must be prepare or load")
print(prepared)''', stage="data", requires=("install", "settings", "preflight"), invalidates=("training", "archive")),
        cell("markdown", """
        ## 5. Authorize this batch and train

        Set RUN_TRAINING=True after reviewing the complete plan. Training history
        updates one display rather than appending a figure every snapshot. Final
        plots are saved on disk; completed-run links appear immediately. Failure
        or interruption stops the worker. Rerun settings/preflight after changes;
        this cell never automatically skips or resumes earlier models.
        """),
        cell("code", '''RUN_TRAINING = False
LIVE_PLOTS = True
if RUN_TRAINING is not True:
    raise RuntimeError("Training disabled. Review the plan, then explicitly set RUN_TRAINING=True")
training_view = display("Starting GPW ablations...", display_id=True)
metrics_view = display("No completed run yet.", display_id=True)

def show_event(event):
    path = Path(event["directory"])
    if event["kind"] == "snapshot":
        if LIVE_PLOTS:
            training_view.update(Image(filename=str(path / "visuals/training.png")))
    elif event["kind"] == "complete":
        metrics_view.update(pd.DataFrame([{"dataset": event["dataset"], "variant": event["name"], **event["metrics"]}]))
        print(event["dataset"], "/", event["name"], "saved:", event["archive"])
        display(FileLink(event["archive"]))
    else:
        raise ValueError(f"Unknown worker event: {event['kind']}")

BATCH_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("ablations_%Y%m%dT%H%M%S%fZ")
result = call_worker("train", plan=PLAN, plan_sha256=PLAN_SHA256,
    prepared_directory=str(PREPARED_DIRECTORY), directory=str(BATCH_DIRECTORY),
    execute_training=RUN_TRAINING, on_event=show_event)
display(pd.read_csv(BATCH_DIRECTORY / "comparison.csv"))
aggregate = json.loads((BATCH_DIRECTORY / "ablation_summary.json").read_text())
display(pd.DataFrame([{"dataset": r["dataset"], "rank": r["q_rank"], "stitches": r["stitch_mode"],
    "completed_seeds": r["completed_seeds"], **{k + "_mean": v["mean"] for k, v in r["metrics"].items()}}
    for r in aggregate]))
for dataset in DATASETS:
    display(Markdown("### " + dataset))
    display(Image(filename=str(BATCH_DIRECTORY / "visuals" / (dataset + ".png"))))
print("Full seed means and sample SDs:", BATCH_DIRECTORY / "ablation_summary.json")''',
             stage="training", requires=("install", "settings", "preflight", "data"), invalidates=("archive",)),
        cell("markdown", "## 6. Archive the completed batch"),
        cell("code", '''ARCHIVE_PATH = WORK_ROOT / datetime.now(UTC).strftime("gpw_ablations_%Y%m%dT%H%M%S%fZ.zip")
archived = call_worker("archive", plan=PLAN, plan_sha256=PLAN_SHA256,
    directory=str(BATCH_DIRECTORY), destination=str(ARCHIVE_PATH))
display(FileLink(archived["archive"]))
print("Save a Kaggle version with outputs or download the archives to preserve them outside this session.")''',
             stage="archive", requires=("training",)),
    ]
    target = Path(root) / "notebooks/GPW_Ablations_Kaggle.ipynb"
    target.write_text(json.dumps({"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "gpw_source": source_metadata()},
        "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
