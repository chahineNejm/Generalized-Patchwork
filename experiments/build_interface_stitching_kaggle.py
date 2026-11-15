"""Build the interface-gap stitching study notebook (periodic heat, grid-aligned GPW)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.build_gpw_ablation_kaggle import build_notebook as base_notebook
from experiments.build_heaton_colab import cell
from experiments.default_hyperparameters import interface_stitching_defaults


def build_notebook(root):
    defaults = interface_stitching_defaults(device="cuda:0",
        input_root="/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs")
    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        (folder / "notebooks").mkdir()
        notebook = json.loads(base_notebook(folder).read_text(encoding="utf-8"))
    cells = notebook["cells"]
    cells[0] = cell("markdown", """
    # Interface stitching study: periodic heat with a training gap

    Enable Internet and a GPU. No attached input is required (fields are simulated).
    Question: does stitching help where one patch has no data next to the interface?

    Datasets (same simulator; seed31 80/20 split, then the gap rows move to test):
    - periodic_heat_jagged: periods x5/y4, zigzag material interface x = 0.5 +/- 0.06
      (4 teeth), no training points for 0.5 <= x < 0.55.
    - periodic_heat_gap: the original periods 10x10 field, straight interface, same gap.
    - periodic_heat_jagged_full: the jagged field without a gap (control).
    Gap datasets are also scored per region: gap, near interface (|x-0.5| < 0.1), interior.

    GPW: 2 patches x 32 supports from the Standard Patchwork grid split at x = 0.5,
    8 stitches per boundary on that line; ranks 0/8 x stitches none/fixed/learned;
    1000 epochs maximum, 50 warm-up, training-loss early stopping. Seeds 31-35 vary
    model initialization only (data stay seed31): 90 runs. A CPU check of jagged/
    seeds 31-33 took about 2-3 minutes per run; GPU time is not measured.
    Every completed run saves its ZIP before the next starts; download them as they appear.
    """)
    cells[5] = cell("code", f'''from datetime import UTC, datetime
from IPython.display import FileLink, Image, Markdown, display
import pandas as pd

INPUT_ROOT = "/kaggle/input/datasets/chahinenejma/gpw-kaggle-inputs"
SEEDS = {defaults["seeds"]!r}
DATASETS = {defaults["datasets"]!r}
RANKS = {defaults["ranks"]!r}
STITCH_MODES = {defaults["stitch_modes"]!r}
INDUCING_POINTS = {defaults["inducing_points"]}
PATCHES = {defaults["patches"]}
EPOCHS = {defaults["epochs"]}
WARMUP_EPOCHS = {defaults["warmup_epochs"]}
EARLY_STOPPING = {defaults["early_stopping"]!r}
MODEL_OVERRIDES = {defaults["model_overrides"]!r}
DATA_OVERRIDES = {{}}
SAVE_INDUCING_POSITIONS = True
PROGRESS_EVERY_EPOCHS = {defaults["progress_every_epochs"]}

settings = call_worker("settings", study="gpw_stitch_rank", device="cuda:0", input_root=INPUT_ROOT,
    datasets=DATASETS, seeds=SEEDS, ranks=RANKS, stitch_modes=STITCH_MODES,
    inducing_points=INDUCING_POINTS, patches=PATCHES, epochs=EPOCHS, warmup_epochs=WARMUP_EPOCHS,
    early_stopping=EARLY_STOPPING, model_overrides=MODEL_OVERRIDES, data_overrides=DATA_OVERRIDES,
    progress_every_epochs=PROGRESS_EVERY_EPOCHS, save_inducing_positions=SAVE_INDUCING_POSITIONS)
PLAN, PLAN_SHA256 = settings["plan"], settings["plan_sha256"]
display(pd.DataFrame(settings["summary"]))
print("Selected runs:", settings["total_runs"], "Maximum total epochs:", settings["total_epochs"])
display(pd.DataFrame([{{"dataset": r["dataset"], "name": r["name"], **r["config"]}} for r in PLAN["runs"]]))''',
        stage="settings", requires=("install",), invalidates=("preflight", "training", "archive"))
    source = "".join(cells[11]["source"])
    marker = 'display(pd.read_csv(BATCH_DIRECTORY / "comparison.csv"))'
    if source.count(marker) != 1:
        raise ValueError("Base training cell changed; update the interface notebook builder")
    head = source[:source.index(marker)].replace('"Starting GPW ablations..."', '"Starting interface stitching study..."')
    head = head.replace('strftime("ablations_', 'strftime("interface_')
    cells[11] = cell("code", head[head.index("RUN_TRAINING = False"):] + '''table = pd.read_csv(BATCH_DIRECTORY / "comparison.csv")
table["condition"] = "rank" + table["q_rank"].astype(str) + " / " + table["stitch_mode"]
display(table)
metrics = ["rmse_standardized", "nlpd_standardized", "coverage_95_standardized",
           "gap_rmse_standardized", "gap_nlpd_standardized", "near_interface_rmse_standardized",
           "near_interface_nlpd_standardized", "interior_rmse_standardized", "training_seconds"]
for dataset in DATASETS:
    local = table[table["dataset"] == dataset]
    display(Markdown(f"### {dataset}: mean (sample SD) over seeds"))
    present = [m for m in metrics if m in local and local[m].notna().any()]
    mean, sd = local.groupby("condition")[present].mean(), local.groupby("condition")[present].std()
    display(mean.round(4).astype(str) + " (" + sd.round(4).astype(str) + ")")
    for metric in [m for m in ("gap_rmse_standardized", "rmse_standardized") if m in present][:1]:
        display(Markdown(f"Per-seed {metric}"))
        display(local.pivot_table(index="seed", columns="condition", values=metric).round(4))
    display(Image(filename=str(BATCH_DIRECTORY / "visuals" / (dataset + ".png"))))
print("Seed means/SDs for all metrics:", BATCH_DIRECTORY / "ablation_summary.json")''', stage="training", requires=("install", "settings", "preflight", "data"), invalidates=("archive",))
    cells[13]["source"] = [line.replace("gpw_ablations_", "gpw_interface_") for line in cells[13]["source"]]
    target = Path(root) / "notebooks/GPW_Interface_Stitching_Kaggle.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
