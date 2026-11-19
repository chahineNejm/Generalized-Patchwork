"""Build the main-table VNNGP CPU schedule with explicit portable continuation."""

import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata
from experiments.default_hyperparameters import vnngp_cpu_table_defaults


def build_notebook(root):
    defaults = vnngp_cpu_table_defaults()
    cells = [cell("markdown", """
    # VNNGP on CPU: all main-table datasets

    Enable Internet; accelerator None. DATASETS selects all11 table datasets,
    excluding UKHousing, the heat cases and extra synthetic configurations.
    Editable SEEDS=[31,32,33] gives33 sequential runs. Attach the approved
    airline.npz,airline.json and icethk.txt inputs at INPUT_ROOT before starting.
    K=32 now; K=256 remains an explicit later choice. RBF ARD, fixed inducing
    locations at all training inputs, mean-field q(U). Maximum100 epochs,
    20 warm-up, Adam0.002 for q and0.001 for kernel/noise after warm-up.
    Existing training-loss early stopping: patience10, relative improvement0.001;
    evaluate the last trained model, never restore a best checkpoint.
    Paper initialization: lengthscales/output variance/noise variance0.6931.
    Existing bounds, jitter, batch512 and two CPU threads remain. No grid or LR
    scheduler. Existing dataset preprocessing/splits and data seed remain fixed.
    This is not a reproduction of the paper's Matern kernel or validation grid.

    A session pauses at an optimizer-step boundary after about10 training hours.
    Checkpoints every5 minutes, on pause and completion include the full model,
    optimizer, random states, minibatch permutation/cursor, partial epoch loss
    and early-stopping state. Unexpected termination can lose work since the
    last checkpoint. Setup, graph rebuilding, scoring and ZIP creation also take
    time;10 hours is a soft training-session limit, not a hard process deadline.
    Prediction pauses between batches; an unfinished prediction split is recomputed
    explicitly on resume. Completed prediction splits are retained.

    Each completed seed and session saves a portable ZIP under
    /kaggle/working/gpw_vnngp_cpu. Save/download it before session storage disappears.
    In the next session attach that ZIP as a Kaggle input, set ACTION="resume"
    and RESUME_ARCHIVE to its exact path. Keep DATASETS, SEEDS and settings unchanged.
    Resume never happens automatically. It continues the saved dataset/seed list;
    changing/adding seeds creates a separate experiment, not a checkpoint migration.
    A finished job rejects resume. Failures remain loud; no retries or substitutions.

    Resume requires the same source commit, dependency versions and verified data.
    If main changes, the runner stops rather than loading a checkpoint into new
    code. The original source ZIP is included for provenance; it is not executed
    automatically. Session files alone are not an external backup. This notebook
    cannot relaunch itself when Kaggle ends the session.
    """), cell("markdown", "## 1. Source and dependencies"),
        cell("code", github_bootstrap("/kaggle/working/gpw_source", isolated=True), stage="checkout",
             invalidates=("install", "settings", "training")),
        cell("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--editable", str(ROOT) + "[data,ukhousing,benchmark]"], check=True)
import runpy
run_worker = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]
WORK_ROOT = Path("/kaggle/working/gpw_vnngp_cpu")

def call_worker(action, **values):
    return run_worker(ROOT, {"action": action, "source_provenance": SOURCE_PROVENANCE, **values},
        module="experiments.vnngp_cpu_worker", jobs_directory=WORK_ROOT / "worker_jobs",
        on_event=show_event)''', stage="install", requires=("checkout",), invalidates=("settings", "training")),
        cell("markdown", """## 2. Main-table datasets and editable model seeds

    The default list matches the paper table. The two synthetic cases use D5,
    train50k/test12.5k. FLIGHT explicitly retains the approved800k subset and
    80/20 split. Other existing data/split protocols remain unchanged.
    Edit DATASETS to select/reorder runs; DATA_OVERRIDES is keyed by dataset.
    Remove its flight entry if flight is deselected. All manual inputs are checked
    before training; a missing input raises, never skips that dataset.
    Adding seeds multiplies compute. All seeds run for each dataset before
    proceeding to the next. One shared10-hour budget covers the whole session,
    not10 hours per dataset. Each completed seed ZIP includes the whole queue
    state and cumulative comparison.csv. No training during settings.
    """),
        cell("code", f'''from datetime import UTC, datetime
from IPython.display import FileLink, display
import pandas as pd

DATASETS = {defaults["datasets"]!r}
SEEDS = {defaults["seeds"]!r}
INPUT_ROOT = "/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs"
DATA_OVERRIDES = {defaults["data_overrides"]!r}
EPOCHS = {defaults["epochs"]}
WARMUP_EPOCHS = {defaults["warmup_epochs"]}
NEIGHBORS = {defaults["neighbors"]}
VNNGP_JITTER = {defaults["vnngp_jitter"]}
LEARNING_RATE_Q = {defaults["learning_rate_q"]}
LEARNING_RATE_HYPERPARAMETERS = {defaults["learning_rate_hyperparameters"]}
LENGTHSCALE = {defaults["lengthscale"]}
OUTPUT_VARIANCE = {defaults["outputscale"]}
NOISE_VARIANCE = 0.6931
EARLY_STOPPING = {defaults["early_stopping"]!r}
SESSION_HOURS = {defaults["session_hours"]}
CHECKPOINT_SECONDS = {defaults["checkpoint_seconds"]}
THREADS = {defaults["threads"]}
BATCH_SIZE = {defaults["batch_size"]}
PREDICTION_BATCH_SIZE = {defaults["prediction_batch_size"]}
PROGRESS_EVERY_EPOCHS = {defaults["progress_every_epochs"]}

def show_event(event):
    if event["kind"] != "complete":
        raise ValueError(f"Unknown worker event: {{event}}")
    display(pd.DataFrame([event["metrics"]]))
    display(FileLink(event["archive"]))

if NOISE_VARIANCE <= 0:
    raise ValueError("Initial noise variance must be positive")
settings = call_worker("settings", datasets=DATASETS, seeds=SEEDS, input_root=INPUT_ROOT,
    data_overrides=DATA_OVERRIDES, overrides={{"epochs": EPOCHS, "warmup_epochs": WARMUP_EPOCHS,
    "neighbors": NEIGHBORS, "vnngp_jitter": VNNGP_JITTER, "learning_rate_q": LEARNING_RATE_Q,
    "learning_rate_hyperparameters": LEARNING_RATE_HYPERPARAMETERS,
    "lengthscale": LENGTHSCALE, "outputscale": OUTPUT_VARIANCE, "noise_std": NOISE_VARIANCE**0.5,
    "early_stopping": EARLY_STOPPING, "session_hours": SESSION_HOURS,
    "checkpoint_seconds": CHECKPOINT_SECONDS, "threads": THREADS, "batch_size": BATCH_SIZE,
    "prediction_batch_size": PREDICTION_BATCH_SIZE, "progress_every_epochs": PROGRESS_EVERY_EPOCHS}})
PLAN, PLAN_SHA256 = settings["plan"], settings["plan_sha256"]
print("Datasets:", DATASETS, "Model seeds:", SEEDS)
print("Total runs:", settings["total_runs"], "Maximum total epochs:", settings["maximum_epochs"])
display(pd.DataFrame([{{"dataset": part["data"]["name"], **config}}
    for part in PLAN["dataset_plans"] for config in part["configs"]]))
display(pd.DataFrame([part["data"] for part in PLAN["dataset_plans"]]))
print("Training controls:", PLAN["dataset_plans"][0]["settings"])''', stage="settings", requires=("install",), invalidates=("training",)),
        cell("markdown", """## 3. Start or explicitly resume one session

    For a new experiment use ACTION="start". For continuation use ACTION="resume"
    and the exact previously saved portable ZIP in RESUME_ARCHIVE. New output
    directories are always created; previous outputs are never overwritten.
    Checkpoints contain tensors and simple state, loaded with weights_only=True;
    use only your own archives. Checksums detect corruption, not authorship.
    Set RUN_TRAINING=True after reviewing the complete plan.
    """),
        cell("code", '''ACTION = "start"
RESUME_ARCHIVE = None
RUN_TRAINING = False
if RUN_TRAINING is not True:
    raise RuntimeError("Training disabled; explicitly set RUN_TRAINING=True")
if ACTION not in ("start", "resume"):
    raise ValueError("ACTION must be start or resume")
if ACTION == "resume" and RESUME_ARCHIVE is None:
    raise ValueError("Set the exact attached RESUME_ARCHIVE path")
SESSION_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("session_%Y%m%dT%H%M%S%fZ")
result = call_worker(ACTION, plan=PLAN, plan_sha256=PLAN_SHA256,
    directory=str(SESSION_DIRECTORY), data_root="/kaggle/working/gpw_data",
    execute_training=RUN_TRAINING, resume_archive=RESUME_ARCHIVE,
    archive=str(SESSION_DIRECTORY) + ".zip")
print(result)
display(FileLink(result["archive"]))
print("If paused, save this ZIP and explicitly resume it in your next session.")''',
             stage="training", requires=("install", "settings"))]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
        "name": "python3"}, "language_info": {"name": "python"}, "source": source_metadata()},
        "nbformat": 4, "nbformat_minor": 5}
    destination = Path(root) / "notebooks/VNNGP_CPU_Kaggle.ipynb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
