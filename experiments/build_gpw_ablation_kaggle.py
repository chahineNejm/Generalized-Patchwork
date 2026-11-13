"""Build the approved source-backed GPW stitching/rank Kaggle study."""

import argparse
import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata
from experiments.default_hyperparameters import (
    gpw_ablation_defaults,
    gpw_flight_ablation_defaults,
    gpw_shared_kernel_defaults,
)


def build_notebook(root, *, shared_kernel=False, flight=False):
    if shared_kernel and flight:
        raise ValueError("Select either shared-kernel or FLIGHT ablations, not both")
    factory = gpw_ablation_defaults
    if shared_kernel:
        factory = gpw_shared_kernel_defaults
    elif flight:
        factory = gpw_flight_ablation_defaults
    defaults = factory(device="cuda:0",
        input_root="/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs")
    if shared_kernel:
        defaults = {**defaults, "ranks": [defaults["q_rank"]], "stitch_modes": ["learned"]}
    cells = [
        cell("markdown", """
        # GPW stitching and variational-rank ablations

        Enable Internet and a GPU. Attach the existing gpw-kaggle-inputs dataset;
        only icethk.txt is needed from it (FLIGHT is not selected). INPUT_ROOT
        below is its exact mounted directory. Other datasets use the existing
        approved download/preprocessing protocols, with no source substitution.

        Default study: California Housing, PROTEIN, TCO and OISST x 3 seeds x
        ranks {0,32} x stitch modes {none,fixed,learned} = 72 runs. TCO is
        included once. Every dataset uses 1024 inducing points,
        eight patches of128. Model seeds are31-33. Data splits and synthetic
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
        Training bars name the dataset/model/seed/rank/stitch mode and refresh
        every20 epochs, plus their initial/final state. Histories still save every
        epoch; visual snapshots keep their existing schedule. Current runs are
        not changed by this source update.

        Select DATASETS and SEEDS explicitly to divide the study into finite
        GPU-session batches. One seed across these four datasets gives24 runs.
        Other existing datasets remain available for explicit later selection.
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

        DATASETS and SEEDS select the current batch. The defaults give72
        runs; one seed gives24. MODEL_OVERRIDES={dataset:{field:value}} applies
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
DATA_OVERRIDES = {defaults.get("data_overrides", {})!r}
SAVE_INDUCING_POSITIONS = True
PROGRESS_EVERY_EPOCHS = {defaults["progress_every_epochs"]}

settings = call_worker("settings", study="gpw_stitch_rank", device="cuda:0", input_root=INPUT_ROOT,
    datasets=DATASETS, seeds=SEEDS, ranks=RANKS, stitch_modes=STITCH_MODES,
    inducing_points=INDUCING_POINTS, patches=PATCHES, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
    early_stopping=EARLY_STOPPING, model_overrides=MODEL_OVERRIDES, data_overrides=DATA_OVERRIDES,
    progress_every_epochs=PROGRESS_EVERY_EPOCHS,
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
    if shared_kernel:
        cells[0] = cell("markdown", """
        # GPW shared-kernel comparison

        Enable Internet and a GPU in Kaggle. No attached input bundle is required
        for the default four datasets. Code loads latest main in fresh workers;
        no model/data implementations are embedded and no reconnect is required.

        California Housing, PROTEIN, TCO and OISST x seeds31,32,33 =12 NEW runs.
        Each uses1024 observed-row inducing locations in8 patches of128, rank32,
        learned stitches and one shared learned ARD lengthscale vector/output
        variance across patches. There is still one global observation-noise SD.
        Supports, stitches and variational parameters remain separately trainable.
        Sharing kernel parameters does not identify the independent local latent
        functions or remove their stitching constraints.

        Match the current local-kernel rank32/learned-stitch ablations: same
        seed31 data splits/preprocessing and same initial supports/centers/graph/
        stitches per dataset/model seed. Kernel/noise starts, bounds, numerics,
        optimization and both predictive variance terms are unchanged. Only the
        kernel parameter tying changes. Effective configs, shared parameter count,
        initialization SHA256 and a local-reference manifest are saved per run.

        This notebook trains ONLY shared kernels. It does not rerun, search for,
        silently load or claim completion of local-kernel results. Compare those
        later only after checking effective configs (except kernel_mode), numerical
        algebra/source identity, data/splits and initialization fingerprints.

        Defaults:100 epochs maximum/20 warm-up, training-loss patience10,
        relative improvement0.001, evaluate last state without restoring a best
        checkpoint. Logs identify dataset/model/seed/kernel mode and refresh
        every20 epochs plus initial/final state. Histories remain per epoch.
        For a smoke check set EPOCHS=3,WARMUP_EPOCHS=1 explicitly.

        Review the finite12-run plan, then set RUN_TRAINING=True. Every completed
        model saves predictions, metrics, checkpoints, initial/final point plots,
        training/kernel histories and its ZIP before the next starts. Failures
        stop subsequent runs. Reruns start fresh, never auto-resume/skip. Results
        use /kaggle/working/gpw_ablation_results; download archives or save a
        Kaggle version with outputs before session storage is lost.
        """)
        cells[4] = cell("markdown", """
        ## 2. Select shared-kernel runs

        SEEDS defaults to31,32,33. Data seeds stay fixed at31. MODEL_OVERRIDES
        applies identically within each dataset, using the same call-time fields
        as the stitching/rank study. Any override must also match the local
        reference used later. No local result is imported by this notebook.
        """)
        cells[5] = cell("code", f'''from datetime import UTC, datetime
from IPython.display import FileLink, Image, display
import pandas as pd

INPUT_ROOT = "/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs"
DATASETS = {defaults["datasets"]!r}
SEEDS = {defaults["seeds"]!r}
Q_RANK = {defaults["q_rank"]}
INDUCING_POINTS = {defaults["inducing_points"]}
PATCHES = {defaults["patches"]}
EPOCHS = {defaults["epochs"]}
WARMUP_EPOCHS = {defaults["warmup_epochs"]}
EARLY_STOPPING = {defaults["early_stopping"]!r}
PROGRESS_EVERY_EPOCHS = {defaults["progress_every_epochs"]}
MODEL_OVERRIDES = {{}}
DATA_OVERRIDES = {{}}
SAVE_INDUCING_POSITIONS = True

settings = call_worker("settings", study="gpw_kernel_sharing", device="cuda:0", input_root=INPUT_ROOT,
    datasets=DATASETS, seeds=SEEDS, q_rank=Q_RANK, inducing_points=INDUCING_POINTS, patches=PATCHES,
    epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, early_stopping=EARLY_STOPPING,
    model_overrides=MODEL_OVERRIDES, data_overrides=DATA_OVERRIDES,
    progress_every_epochs=PROGRESS_EVERY_EPOCHS, save_inducing_positions=SAVE_INDUCING_POSITIONS)
PLAN, PLAN_SHA256 = settings["plan"], settings["plan_sha256"]
display(pd.DataFrame(settings["summary"]))
print("New shared-kernel runs:", settings["total_runs"], "Maximum total epochs:", settings["total_epochs"])
display(pd.DataFrame([{{"dataset": r["dataset"], "name": r["name"], **r["config"]}} for r in PLAN["runs"]]))''',
            stage="settings", requires=("install",), invalidates=("preflight", "training", "archive"))
        cells[11] = cell("code", '''RUN_TRAINING = False
LIVE_PLOTS = True
if RUN_TRAINING is not True:
    raise RuntimeError("Training disabled. Review the plan, then explicitly set RUN_TRAINING=True")
training_view = display("Starting shared-kernel GPW...", display_id=True)
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

BATCH_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("shared_kernels_%Y%m%dT%H%M%S%fZ")
result = call_worker("train", plan=PLAN, plan_sha256=PLAN_SHA256,
    prepared_directory=str(PREPARED_DIRECTORY), directory=str(BATCH_DIRECTORY),
    execute_training=RUN_TRAINING, on_event=show_event)
display(pd.read_csv(BATCH_DIRECTORY / "comparison.csv"))
aggregate = json.loads((BATCH_DIRECTORY / "kernel_summary.json").read_text())
display(pd.DataFrame([{"dataset": r["dataset"], "kernel_mode": r["kernel_mode"],
    "completed_seeds": r["completed_seeds"], **{k + "_mean": v["mean"] for k, v in r["metrics"].items()}}
    for r in aggregate]))
print("Shared-only seed means/sample SDs:", BATCH_DIRECTORY / "kernel_summary.json")
print("Local reference manifests are in each run archive; no local results were loaded.")''',
            stage="training", requires=("install", "settings", "preflight", "data"), invalidates=("archive",))
    if flight:
        cells[0] = cell("markdown", """
        # Rank64 stitching ablations: FLIGHT, TCO.L2 and ICETHICK

        Enable Internet and a GPU in Kaggle. Attach gpw-kaggle-inputs containing
        airline.npz,airline.json and icethk.txt at INPUT_ROOT. All are required;
        checksums/provenance are verified. No pickle reader or substitute source.

        Nine runs: FLIGHT800k,TCO.L2 and ICETHICK,model seed31,rank64 ONLY,
        each with no,fixed and learned stitches. Use1024 inducing points in8
        patches of128. Three modes within each dataset share identical initial
        observed rows,supports,centers,candidate
        graph and candidate stitches, checked by a saved geometry fingerprint.
        Separate local ARD RBF kernels and variational parameters remain free.
        None removes constraints, fixed freezes only stitches, learned moves them.
        Preserve prior-dependent q initialization; initial q covariance can differ.

        Uniform800,000-row subset without replacement from the5,929,413-row
        trusted author-prepared FLIGHT file, eight features,subset seed31 and
        80/20 split with data seed31:640,000 train/160,000 test. Input min-max
        scaling uses all selected rows; target scaling uses training rows only.
        This is size-matched, not the paper's exact records or90/10 split.
        Source-row indices and checksums are saved in every run archive.
        TCO.L2 and ICETHICK use their full approved datasets and existing80/20
        splits/preprocessing with data seed31. TCO.L2 is not the smaller TCO.
        Existing FLIGHT batch size,
        learning rates, initialization, bounds, numerics and stitch count remain.
        The1024-support ablation budget differs from the2048-point main preset.

        Defaults100 epochs maximum/20 kernel-noise warm-up; training-loss early
        stopping patience10, relative improvement0.001, evaluate the last state.
        Supports and learned stitches are not frozen during warm-up. Logs name
        dataset/model/seed/rank/stitch mode every20 epochs plus initial/final state;
        histories save every epoch. Change EPOCHS=3,WARMUP_EPOCHS=1 for an explicit
        short smoke check. No configuration is adjusted automatically.

        Provisional overnight estimate:6-9 hours on comparable GPU hardware,
        based on saved GPW timings and extrapolation for FLIGHT800k/rank64.
        This is not a measured FLIGHT runtime or a14-hour guarantee. The batch
        is finite; no time-triggered truncation,skipping or parameter changes.

        Review the complete plan, then set RUN_TRAINING=True. Code loads latest
        main in fresh workers without reconnects. Every completed model saves
        predictions, metrics, initial/final positions, histories, checkpoints,
        effective configuration/provenance and a ZIP before the next begins.
        Failures stop subsequent runs. Reruns start fresh, never resume/skip.
        Download ZIPs or save Kaggle outputs: session-local files are not a backup.
        """)
        cells[4] = cell("markdown", """
        ## 2. Review the nine-run rank64 plan

        Edit SEEDS in the configuration cell before running settings:
        SEEDS=[31] gives9 runs; SEEDS=[31,32,33] gives27 runs. Any nonempty list
        of distinct nonnegative integer model seeds is accepted. Each added seed
        adds9 fresh runs,not a resume. Keep[31] for the6-9-hour overnight estimate;
        three seeds would provisionally take18-27 hours. Data/split/subset seed
        stays31; model seeds change initialization/training only. The complete
        run count and effective seeds are printed before authorization.

        DATASETS=["flight","tco_l2","icethick"],SEEDS=[31],RANKS=[64],all three
        stitch modes. DATA_OVERRIDES explicitly selects800,000 FLIGHT rows only.
        INPUT_ROOT must contain airline.npz,airline.json and icethk.txt.
        Hyperparameters remain editable here; MODEL_OVERRIDES applies equally
        to all three modes within the named dataset.
        Data preprocessing and seed remain unchanged. One seed cannot estimate
        between-seed variability; summary sample SD is undefined, not zero.
        """)
    filename = "GPW_Ablations_Kaggle.ipynb"
    if shared_kernel:
        filename = "GPW_Shared_Kernels_Kaggle.ipynb"
    elif flight:
        filename = "GPW_FLIGHT_Ablations_Kaggle.ipynb"
    target = Path(root) / "notebooks" / filename
    target.write_text(json.dumps({"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "gpw_source": source_metadata()},
        "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--shared-kernel", action="store_true")
    selection.add_argument("--flight", action="store_true")
    args = parser.parse_args()
    print(build_notebook(Path(__file__).resolve().parents[1], shared_kernel=args.shared_kernel, flight=args.flight))
