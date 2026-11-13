"""Build a source-backed, explicitly gated Kaggle experiment notebook."""

import argparse
import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata
from experiments.default_hyperparameters import (
    benchmark_repeat_defaults,
    multidataset_notebook_defaults,
)


def build_notebook(root, *, repeats=False):
    factory = benchmark_repeat_defaults if repeats else multidataset_notebook_defaults
    defaults = factory(device="cuda:0", input_root="/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs")
    seed_settings = (f'SEEDS = {defaults["seeds"]!r}\nPROGRESS_EVERY_EPOCHS = {defaults["progress_every_epochs"]}\n'
                     if repeats else "")
    repeat_arguments = ('study="benchmark_repeats", seeds=SEEDS, progress_every_epochs=PROGRESS_EVERY_EPOCHS,\n    '
                        if repeats else "")
    model_overrides = json.dumps(defaults["model_overrides"], indent=4) if repeats else "{}"
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

        The reduced batch selects 11 dataset configurations and 61 model runs.
        FLIGHT and VNNGP are excluded. Each synthetic function uses D=5 and
        50000 training points. All existing GPW/ProSpar sizes are retained;
        Standard Patchwork uses only its PC-tree variant. Full presets remain
        available for explicit later selection. Training is OFF until
        RUN_TRAINING=True. Review the complete plan first.
        No point budget, dataset size, device or numerical value is reduced silently.

        New real and synthetic protocols use seeded 80/20; existing OISST/UKHousing
        protocols and official MODIS split/training-only detrending are retained.
        All models keep their approved ARD RBF/initialization/optimizer settings.
        Standard Patchwork uses all input dimensions and paper-style balanced
        PC trees. Grid/Voronoi variants are deferred from the current selection. Reduced-data
        Standard Patchwork uses exactly the primary GPW initial selected targets,
        while GPW/SVGP/ProSpar/VNNGP train on all observations.

        ProSpar is deliberately smaller than its paper: matched budgets here are
        512/1024, versus the paper's 200x25=5000 points. Actual/paper capacities
        are recorded with every ProSpar result. These are controlled comparisons,
        not reproductions of every paper's kernel, split or optimization protocol.

        Precipitation and satellite drag are explicitly DEFERRED: verified source
        data are unavailable. They are not silently skipped from a selected plan.
        Additional synthetic dimensions/sample sizes are deferred, not deleted.
        The selected 80/20 synthetic test size is12500, unlike the paper's25000.

        Train for at most100 epochs with20 warm-up. After warm-up, stop after10
        epochs without a relative training-loss improvement of at least0.1%.
        Epoch21 establishes the monitoring baseline. Patience ignores warm-up;
        no validation/test targets are used. Evaluate the last trained parameters,
        never restore an earlier checkpoint. Actual epoch counts and stopping
        reasons are saved. EPOCHS/WARMUP_EPOCHS/EARLY_STOPPING remain editable;
        set EARLY_STOPPING=None to explicitly disable stopping for a smoke run.

        Results, plan, worker logs, tracebacks and per-run ZIPs are written under
        /kaggle/working/gpw_results. Every run saves source/data provenance, full
        effective configuration, initial/final checkpoints, predictions, metrics,
        timing, training/kernel histories and optional full-dimensional positions.
        Metrics include both explicitly labeled standardized and original-target
        scales; do not compare values with different units as the same metric.
        High-dimensional location plots are labeled two-feature projections.
        Each completed model is saved and archived before the next starts, with
        a download link displayed immediately. Completed archives survive a later
        model failure within the session; download them before session storage is
        lost. This does not create an external backup automatically.
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
        displayed preset minus EXCLUDED_MODEL_TYPES. VNNGP is excluded by default;
        an explicit selection conflicting with an exclusion raises.
        Unknown names raise.
        MODEL_OVERRIDES is {dataset: {model_name: {field: value}}}.
        DATA_OVERRIDES is {dataset: {field: value}}; no source/schema relabeling.

        Example short selection: DATASETS=["california"] and
        MODELS_BY_DATASET={"california": ["SVGP", "GPW"]}.
        For a long run, explicitly change EPOCHS and WARMUP_EPOCHS; no automatic
        warm-up shrinking occurs. Rerunning training always starts fresh.
        """),
        cell("code", f'''from datetime import UTC, datetime
from IPython.display import FileLink, Image, Markdown, display
import pandas as pd

INPUT_ROOT = "/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs"
DATASETS = {defaults["datasets"]!r}
EXCLUDED_MODEL_TYPES = ["vnngp"]
MODELS_BY_DATASET = {json.dumps(defaults["models"], indent=4)}
MODEL_OVERRIDES = {model_overrides}
DATA_OVERRIDES = {{}}
EPOCHS = {defaults["epochs"]}
WARMUP_EPOCHS = {defaults["warmup_epochs"]}
EARLY_STOPPING = {defaults["early_stopping"]!r}
SAVE_INDUCING_POSITIONS = True
{seed_settings}
settings = call_worker("settings", {repeat_arguments}device="cuda:0", input_root=INPUT_ROOT, datasets=DATASETS,
    excluded_model_types=EXCLUDED_MODEL_TYPES,
    early_stopping=EARLY_STOPPING,
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
        display(FileLink(event["archive"]))
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
    if repeats:
        cells[0] = cell("markdown", """
        # Benchmark repeats: seeds32 and33

        Enable Internet and one GPU on Kaggle. Run stages in order; no attached
        input bundle is required for these four automatically downloaded datasets.
        Source loads latest main from chahineNejm/Generalized-Patchwork in fresh
        workers. No model/data implementations are embedded here.

        This finite batch has44 runs: California Housing12, PROTEIN12, TCO12,
        OISST8. Seeds32 and33 vary model initialization/training only. Data splits
        and preprocessing stay seed31. Existing seed31 results are not rerun,
        automatically loaded or pooled. All settings and seed identities are saved.

        Retain the tested configurations: SVGP, GPW8/4 patches and ProSpar8/4
        experts for California/PROTEIN/TCO, plus reduced-data PC-tree Patchwork.
        OISST retains its tested SVGP, GPW8, ProSpar8 and PC-tree Patchwork only.
        Total budgets:512 for California/OISST,1024 for PROTEIN/TCO. This is not
        the1024-point stitching/rank ablation. No VNNGP, FLIGHT or other datasets.

        Standard Patchwork uses the primary eight-patch GPW initial observed rows
        for each model seed and their targets; it does not train on all targets.
        GPW/SVGP/ProSpar train on all training observations. Keep the existing
        ARD RBF kernels, initialization, bounds, numerics, learning rates,
        ProSpar T=1/c=1/nuggets=0.01/prior_jitter=0 and other tested settings.
        Those explicit ProSpar overrides match the saved seed31 archive; they
        do not change the newer general defaults T=0.1/c=2. Do not pool results
        from different ProSpar starts as identical configurations.

        EPOCHS=100 maximum, WARMUP_EPOCHS=20; training-loss early stopping has
        patience10 and relative minimum improvement0.001 after warm-up. Evaluate
        the last state, not a restored best checkpoint. For a smoke test explicitly
        set EPOCHS=3 and WARMUP_EPOCHS=1. Review the plan, then RUN_TRAINING=True.

        Dataset/model/seed labels refresh every20 epochs plus initial/final state.
        Full loss history stays per epoch; visuals update one existing display.
        Completed metrics and ZIPs save before each next run. Results live under
        /kaggle/working/gpw_results. Download archives or save a Kaggle version
        with outputs before session storage is lost. Failures stop subsequent runs;
        no auto-resume/skip or overwrite. An estimated4-5h is not a runtime limit.
        """)
    filename = "Benchmark_Repeats_Kaggle.ipynb" if repeats else "All_Datasets_Kaggle.ipynb"
    target = Path(root) / "notebooks" / filename
    target.write_text(json.dumps({"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "gpw_source": source_metadata()},
        "nbformat": 4, "nbformat_minor": 5}, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", action="store_true")
    print(build_notebook(Path(__file__).resolve().parents[1], repeats=parser.parse_args().repeats))
