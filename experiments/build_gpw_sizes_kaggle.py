"""Build the source-backed two-capacity stitching comparison."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.build_gpw_ablation_kaggle import build_notebook as base_notebook
from experiments.build_heaton_colab import cell
from experiments.default_hyperparameters import gpw_size_ablation_defaults


def build_notebook(root):
    defaults = gpw_size_ablation_defaults(device="cuda:0",
        input_root="/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs")
    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        (folder / "notebooks").mkdir()
        notebook = json.loads(base_notebook(folder, flight=True).read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[0] = cell("markdown", """
    # GPW: two capacities and three stitch modes

    FLIGHT800k,TCO.L2 and ICETHICK,rank64 only. Two explicit configurations:
    8 patches x32 points (256 total),8 stitches per connected pair;
    4 patches x256 points (1024 total),16 stitches per connected pair.
    Each compares none/fixed/learned stitches. None activates zero constraints.
    These are NOT size-matched configurations: capacity,patch count and stitch
    density all change. Compare stitch modes within each configuration first.

    SEEDS=[31] gives18 runs. Change/add seeds directly,for example[31,32,33]
    gives54 runs. Data/subset/split seed remains31. FLIGHT samples800,000 rows
    without replacement,80/20 split;TCO.L2/ICETHICK use full approved datasets.
    Attach airline.npz,airline.json,icethk.txt;enable Internet and a GPU.

    Both configurations reuse exactly the same verified prepared arrays/splits.
    Within each capacity/dataset/seed,the three modes share initial supports,
    centers,candidate graph and candidate stitches. Initialization is NOT held
    identical across capacities: there are different numbers of points/patches.
    Local kernels,q family,rank64 and existing prior-dependent q initialization
    remain;100 epochs/20 warm-up and existing stopping/numerics/optimizers remain.

    Each configuration is an explicit independent fresh-worker batch. Completed
    models save a ZIP before the next model;each completed configuration also
    saves its collection ZIP before the next configuration. Failures stop the
    schedule;there is no auto-skip,resume,device change or time truncation.
    The earlier6-9-hour estimate DOES NOT apply to these18 runs. This schedule
    has no measured duration or14-hour guarantee. Review the finite plan first.
    Save/download outputs externally;session files are not an external backup.
    """)
    cells[4] = cell("markdown", """
    ## 2. Configure capacities,seeds and datasets

    Edit SEEDS directly. CONFIGURATIONS can explicitly select either/both sizes;
    stitches_per_edge applies to fixed/learned only. MODEL_OVERRIDES applies
    equally to all modes/sizes per dataset;it cannot override stitches_per_edge.
    Review the printed total before RUN_TRAINING=True. No benchmark is launched
    by this settings stage. Data are prepared once and verified for both plans.
    """)
    cells[5] = cell("code", f'''from datetime import UTC, datetime
from IPython.display import FileLink, Image, display
import json
import re
import pandas as pd

SEEDS = {defaults["seeds"]!r}
CONFIGURATIONS = {defaults["configurations"]!r}
DATASETS = {defaults["datasets"]!r}
INPUT_ROOT = "/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs"
RANKS = {defaults["ranks"]!r}
STITCH_MODES = {defaults["stitch_modes"]!r}
EPOCHS = {defaults["epochs"]}
WARMUP_EPOCHS = {defaults["warmup_epochs"]}
EARLY_STOPPING = {defaults["early_stopping"]!r}
DATA_OVERRIDES = {defaults["data_overrides"]!r}
MODEL_OVERRIDES = {{}}
SAVE_INDUCING_POSITIONS = True
PROGRESS_EVERY_EPOCHS = {defaults["progress_every_epochs"]}

if not isinstance(CONFIGURATIONS, list) or not CONFIGURATIONS:
    raise ValueError("CONFIGURATIONS must explicitly select at least one capacity")
if set(MODEL_OVERRIDES) - set(DATASETS):
    raise ValueError("MODEL_OVERRIDES contains unselected datasets")
if any("stitches_per_edge" in values for values in MODEL_OVERRIDES.values()):
    raise ValueError("Set stitch counts only in CONFIGURATIONS")
SCHEDULE = []
for setup in CONFIGURATIONS:
    if set(setup) != {{"name", "inducing_points", "patches", "stitches_per_edge"}}:
        raise ValueError("Each configuration requires name,inducing_points,patches,stitches_per_edge")
    if not isinstance(setup["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", setup["name"]):
        raise ValueError("Configuration names must be nonempty ASCII letters,digits,hyphens or underscores")
    if any(item["name"] == setup["name"] for item in SCHEDULE):
        raise ValueError("Configuration names must be distinct")
    overrides = {{dataset: {{**MODEL_OVERRIDES.get(dataset, {{}}), "stitches_per_edge": setup["stitches_per_edge"]}}
                 for dataset in DATASETS}}
    settings = call_worker("settings", study="gpw_stitch_rank", device="cuda:0", input_root=INPUT_ROOT,
        datasets=DATASETS, seeds=SEEDS, ranks=RANKS, stitch_modes=STITCH_MODES,
        inducing_points=setup["inducing_points"], patches=setup["patches"],
        epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS, early_stopping=EARLY_STOPPING,
        model_overrides=overrides, data_overrides=DATA_OVERRIDES,
        progress_every_epochs=PROGRESS_EVERY_EPOCHS, save_inducing_positions=SAVE_INDUCING_POSITIONS)
    SCHEDULE.append({{"name": setup["name"], **settings}})
if any(item["plan"]["data"] != SCHEDULE[0]["plan"]["data"] for item in SCHEDULE):
    raise ValueError("Capacity plans must have identical dataset configurations")
print("Total runs:", sum(item["total_runs"] for item in SCHEDULE),
      "Maximum total epochs:", sum(item["total_epochs"] for item in SCHEDULE))
display(pd.DataFrame([{{"configuration": item["name"], "dataset": row["dataset"], **row["config"]}}
    for item in SCHEDULE for row in item["plan"]["runs"]]))''', stage="settings", requires=("install",),
        invalidates=("preflight", "training", "archive"))
    cells[7] = cell("code", '''PREFLIGHT_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("size_preflight_%Y%m%dT%H%M%S%fZ")
for item in SCHEDULE:
    print(item["name"], call_worker("preflight", plan=item["plan"], plan_sha256=item["plan_sha256"],
        directory=str(PREFLIGHT_DIRECTORY / item["name"])))''', stage="preflight",
        requires=("install", "settings"), invalidates=("training", "archive"))
    cells[9] = cell("code", '''PREPARED_ACTION = "prepare"
PREPARED_PATH = None
first = SCHEDULE[0]
if PREPARED_ACTION == "prepare":
    PREPARED_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("size_prepared_%Y%m%dT%H%M%S%fZ")
    prepared = call_worker("prepare", plan=first["plan"], plan_sha256=first["plan_sha256"],
        directory=str(PREPARED_DIRECTORY), data_root=str(DATA_ROOT))
elif PREPARED_ACTION == "load":
    if PREPARED_PATH is None:
        raise ValueError("Set PREPARED_PATH explicitly for verified data reuse")
    PREPARED_DIRECTORY = Path(PREPARED_PATH)
    prepared = call_worker("load", plan=first["plan"], plan_sha256=first["plan_sha256"],
        directory=str(PREPARED_DIRECTORY))
else:
    raise ValueError("PREPARED_ACTION must be prepare or load")
for item in SCHEDULE[1:]:
    call_worker("load", plan=item["plan"], plan_sha256=item["plan_sha256"], directory=str(PREPARED_DIRECTORY))
print(prepared)''', stage="data", requires=("install", "settings", "preflight"),
        invalidates=("training", "archive"))
    cells[11] = cell("code", '''RUN_TRAINING = False
LIVE_PLOTS = True
if RUN_TRAINING is not True:
    raise RuntimeError("Training disabled. Review all capacity plans,then set RUN_TRAINING=True")
training_view = display("Starting selected capacities...", display_id=True)
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

STAMP = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
STUDY_DIRECTORY = WORK_ROOT / ("two_sizes_" + STAMP)
STUDY_DIRECTORY.mkdir(parents=True, exist_ok=False)
(STUDY_DIRECTORY / "schedule.json").write_text(json.dumps(SCHEDULE, indent=2, allow_nan=False))
completed = []
for item in SCHEDULE:
    print("Starting configuration:", item["name"])
    batch = STUDY_DIRECTORY / item["name"]
    call_worker("train", plan=item["plan"], plan_sha256=item["plan_sha256"],
        prepared_directory=str(PREPARED_DIRECTORY), directory=str(batch),
        execute_training=RUN_TRAINING, on_event=show_event)
    rows = pd.read_csv(batch / "comparison.csv")
    rows.insert(0, "configuration", item["name"])
    completed.append(rows)
    pd.concat(completed, ignore_index=True).to_csv(STUDY_DIRECTORY / "comparison.csv", index=False)
    archived = call_worker("archive", plan=item["plan"], plan_sha256=item["plan_sha256"],
        directory=str(batch), destination=str(WORK_ROOT / ("gpw_" + item["name"] + "_" + STAMP + ".zip")))
    display(FileLink(archived["archive"]))
display(pd.concat(completed, ignore_index=True))''', stage="training",
        requires=("install", "settings", "preflight", "data"), invalidates=("archive",))
    cells[12] = cell("markdown", "## 6. Archive the completed two-capacity collection")
    cells[13] = cell("code", '''import zipfile
ARCHIVE_PATH = WORK_ROOT / ("gpw_two_sizes_" + STAMP + ".zip")
with zipfile.ZipFile(ARCHIVE_PATH, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(p for p in STUDY_DIRECTORY.rglob("*") if p.is_file()):
        bundle.write(path, path.relative_to(STUDY_DIRECTORY))
display(FileLink(str(ARCHIVE_PATH)))
print("Save a Kaggle version with outputs or download archives before session storage is lost.")''',
        stage="archive", requires=("training",))
    target = Path(root) / "notebooks/GPW_Two_Sizes_Kaggle.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
