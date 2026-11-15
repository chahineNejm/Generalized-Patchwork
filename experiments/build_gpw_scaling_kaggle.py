"""Build the source-backed GPW / Standard Patchwork patch-count scaling notebook."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.build_gpw_ablation_kaggle import build_notebook as base_notebook
from experiments.build_heaton_colab import cell
from experiments.default_hyperparameters import gpw_scaling_defaults


def build_notebook(root):
    defaults = gpw_scaling_defaults(device="cuda:0",
        input_root="/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs")
    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        (folder / "notebooks").mkdir()
        notebook = json.loads(base_notebook(folder, flight=True).read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[0] = cell("markdown", """
    # GPW and Standard Patchwork: patch-count scaling (S01 OISST, S12 FLIGHT800k)

    Enable Internet and a GPU. Attach gpw-kaggle-inputs (airline.npz/json for FLIGHT).
    For each P in PATCH_COUNTS, GPW (Kaggle default: rank32, learned stitches) and
    Standard Patchwork (PC tree) run with the same fixed total of INDUCING_POINTS:
    GPW splits them into P patches of INDUCING_POINTS/P supports; Standard
    Patchwork uses exactly the selected observed rows/targets of the GPW run with
    the same P and seed, in P balanced PC-tree regions. Both use
    STITCHES_PER_BOUNDARY[P] stitches per connected pair / shared boundary.
    GPW patch_neighbors is min(PATCH_NEIGHBORS, P-1). Standard Patchwork trains on
    the selected subset only: a reduced-data baseline, not equal information.

    Defaults: P = 2, 4, 8, 16; 2048 points; 64 stitches for P = 2, 4 and 32 for
    P = 8, 16; seed31; FLIGHT uses the approved seeded 800,000-row subset, 80/20.
    OISST uses a train/test split without validation: 19,370 train / 2,000 test
    (seed31; this differs from the 64/16/20 split of the other OISST experiments).
    That is 16 runs. Edit SEEDS to add model seeds (data seed stays 31); each
    added seed adds 16 runs. Existing 100/20 schedule and training-loss stopping.

    Training time, prediction time and peak CUDA memory are saved per run with all
    metrics. No duration is guaranteed: FLIGHT GPW at P = 2 (1024 supports per
    patch) is expected to be the most expensive run. Every completed run saves its
    ZIP before the next starts; failures stop the batch without skip or resume.
    Download outputs or save a Kaggle version before session storage is lost.
    """)
    cells[4] = cell("markdown", """
    ## 2. Configure patch counts, budget, stitches and seeds

    PATCH_COUNTS must be distinct powers of two >= 2 that divide INDUCING_POINTS.
    STITCHES_PER_BOUNDARY needs one positive count per P (string keys).
    MODEL_OVERRIDES={dataset:{"GPW"|"Standard Patchwork (PC tree)":{field:value}}}
    may change other settings; scaling controls (P, budget, stitches, neighbors,
    partition, reference rows, seed) are rejected. Review the plan first.
    """)
    cells[5] = cell("code", f'''from datetime import UTC, datetime
from IPython.display import FileLink, Image, Markdown, display
import pandas as pd

INPUT_ROOT = "/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs"
SEEDS = {defaults["seeds"]!r}
DATASETS = {defaults["datasets"]!r}
PATCH_COUNTS = {defaults["patch_counts"]!r}
INDUCING_POINTS = {defaults["inducing_points"]}
STITCHES_PER_BOUNDARY = {defaults["stitches_per_boundary"]!r}
PATCH_NEIGHBORS = {defaults["patch_neighbors"]}
EPOCHS = {defaults["epochs"]}
WARMUP_EPOCHS = {defaults["warmup_epochs"]}
EARLY_STOPPING = {defaults["early_stopping"]!r}
DATA_OVERRIDES = {defaults["data_overrides"]!r}
MODEL_OVERRIDES = {{}}
SAVE_INDUCING_POSITIONS = True
PROGRESS_EVERY_EPOCHS = {defaults["progress_every_epochs"]}

settings = call_worker("settings", study="gpw_scaling", device="cuda:0", input_root=INPUT_ROOT,
    datasets=DATASETS, seeds=SEEDS, patch_counts=PATCH_COUNTS, inducing_points=INDUCING_POINTS,
    stitches_per_boundary=STITCHES_PER_BOUNDARY, patch_neighbors=PATCH_NEIGHBORS,
    excluded_model_types=["vnngp"], epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
    early_stopping=EARLY_STOPPING, model_overrides=MODEL_OVERRIDES, data_overrides=DATA_OVERRIDES,
    progress_every_epochs=PROGRESS_EVERY_EPOCHS, save_inducing_positions=SAVE_INDUCING_POSITIONS)
PLAN, PLAN_SHA256 = settings["plan"], settings["plan_sha256"]
display(pd.DataFrame(settings["summary"]))
print("Selected runs:", settings["total_runs"], "Maximum total epochs:", settings["total_epochs"])
print("Study:", PLAN["study"])
display(pd.DataFrame([{{"dataset": r["dataset"], "name": r["name"], **r["config"]}} for r in PLAN["runs"]]))''',
        stage="settings", requires=("install",), invalidates=("preflight", "training", "archive"))
    cells[11] = cell("code", '''RUN_TRAINING = False
LIVE_PLOTS = True
if RUN_TRAINING is not True:
    raise RuntimeError("Training disabled. Review the plan, then explicitly set RUN_TRAINING=True")
training_view = display("Starting GPW / Standard Patchwork scaling...", display_id=True)
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

BATCH_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("scaling_%Y%m%dT%H%M%S%fZ")
result = call_worker("train", plan=PLAN, plan_sha256=PLAN_SHA256,
    prepared_directory=str(PREPARED_DIRECTORY), directory=str(BATCH_DIRECTORY),
    execute_training=RUN_TRAINING, on_event=show_event)
table = pd.read_csv(BATCH_DIRECTORY / "comparison.csv")
display(table)
for metric in ("rmse_standardized", "nlpd_standardized", "coverage_95_standardized",
               "training_seconds", "peak_cuda_megabytes"):
    display(Markdown(f"**{metric}** by patch count"))
    display(table.pivot_table(index=["dataset", "patch_count"], columns="base_model", values=metric))''',
        stage="training", requires=("install", "settings", "preflight", "data"), invalidates=("archive",))
    cells[13]["source"] = [line.replace("gpw_ablations_", "gpw_scaling_") for line in cells[13]["source"]]
    target = Path(root) / "notebooks/GPW_Scaling_Kaggle.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
