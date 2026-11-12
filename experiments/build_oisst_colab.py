"""Build the GitHub-backed OISST experiment launcher."""

import ast
import hashlib
import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata


def build_notebook(root, *, preserve_settings=False):
    cells = [
        cell("markdown", """
        # OISST model comparison

        Select a GPU runtime and Run all once. Code is installed from the latest main
        of chahineNejm/Generalized-Patchwork; the resolved commit is saved.
        Defaults live in experiments/default_hyperparameters.py. Edit overrides
        below, not model definitions. No model or dataset implementation is embedded.
        Then rerun only the stage you want. Failed stages block their dependents;
        rerunning a failed stage is an explicit retry, never an automatic recovery.
        Setup can be rerun without restarting or reconnecting: it verifies and
        updates the checkout, preserving data/results. Each stage uses a fresh
        Python worker, so updated source never mixes with old notebook imports.

        Standard Patchwork uses GPW's initial observed locations and targets,
        not all training observations. The first baseline uses a 4 x 2 grid
        (longitude x latitude); the second uses Voronoi cells of GPW's initial
        clustering centers. Both keep their regions/stitches fixed. GPW still
        learns from all training observations, so these are reduced-data baselines.
        Empty cells raise. Saved partition figures and row indices verify the setup.
        """),
        cell("markdown", "## 1. Load code and dependencies"),
        cell("code", github_bootstrap("/content/gpw_oisst", isolated=True), stage="checkout",
             invalidates=("install", "settings", "data", "training", "analysis", "download")),
        cell("code", '''subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
    "--editable", str(ROOT) + "[data,ukhousing]"], check=True)
import runpy
run_worker = runpy.run_path(str(ROOT / "experiments/colab_client.py"))["run_worker"]

def call_worker(action, *, on_event=None, **values):
    return run_worker(ROOT, {"action": action, "source_provenance": SOURCE_PROVENANCE, **values},
                      on_event=on_event)''', stage="install",
             requires=("checkout",), invalidates=("settings", "data", "training", "analysis", "download")),
        cell("markdown", "## 2. Defaults and explicit overrides"),
        cell("code", """
        from datetime import UTC, datetime
        from IPython.display import Image, display
        from google.colab import files
        import pandas as pd
        MODELS = ["SVGP", "GPW", "ProSpar", "VNNGP", "Standard Patchwork", "Standard Patchwork (GPW regions)"]
        # Point budgets come from GPW; grid shape and stitches are baseline-specific.
        # Example: {"GPW": {"inducing_points": 512, "patches": 8},
        # "Standard Patchwork": {"exact_grid_shape": (4, 2), "exact_stitches_per_edge": 5}}.
        OVERRIDES = {}
        DATA_OVERRIDES = {}  # e.g. {"save_inducing_positions": False}
        settings = call_worker("settings", device="cuda:0", models=MODELS,
                               overrides=OVERRIDES, data_overrides=DATA_OVERRIDES)
        DATA, RUNS = settings["data"], settings["runs"]
        WORK_ROOT = ROOT / "results/oisst"
        display(pd.DataFrame([{"run": row["name"], "training_inputs":
            "GPW initial subset" if row["config"]["model"] == "patchwork" else "all",
            **row["config"]} for row in RUNS]))
        print(DATA)
        """, stage="settings", requires=("install",), invalidates=("training", "analysis", "download")),
        cell("markdown", """
        ## 3. Prepare once or explicitly load saved prepared data

        Keep this stage unchanged when only MODELS or model hyperparameters change.
        To restore saved data, select DATA_ACTION="load" and set PREPARED_PATH.
        No missing or mismatched data are rebuilt automatically.
        """),
        cell("code", """
        DATA_ACTION = "prepare"  # "prepare" or "load"
        PREPARED_PATH = None  # Exact directory for DATA_ACTION="load".
        if DATA_ACTION == "prepare":
            PREPARED_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("prepared_%Y%m%dT%H%M%S%fZ")
            call_worker("prepare", data=DATA, directory=str(PREPARED_DIRECTORY))
        elif DATA_ACTION == "load":
            if PREPARED_PATH is None:
                raise ValueError("Set PREPARED_PATH explicitly before loading prepared data")
            PREPARED_DIRECTORY = Path(PREPARED_PATH)
            call_worker("load", data=DATA, directory=str(PREPARED_DIRECTORY))
        else:
            raise ValueError("DATA_ACTION must be prepare or load")
        print("Prepared data:", PREPARED_DIRECTORY)
        display(Image(filename=str(PREPARED_DIRECTORY / "exploratory/targets.png")))
        """, stage="data", requires=("install", "settings"), invalidates=("training", "analysis", "download")),
        cell("markdown", """
        ## 4. Train only the selected models

        Edit MODELS/OVERRIDES and rerun settings, then this cell. Prepared data
        are reused only after config/checksum validation. Each execution creates
        a new batch and trains selected models from scratch; previous results stay.
        After failure, select only the model you wish to retry and rerun this cell.
        No automatic checkpoint resumption, skipping or overwriting.
        """),
        cell("code", """
        live = display("Starting OISST comparison...", display_id=True)
        history = display("Training history...", display_id=True)

        def show_progress(path, epoch):
            live.update(Image(filename=str(path / "visuals/progress.png")))
            history.update(Image(filename=str(path / "visuals/training.png")))

        def show_completed(path, archive, metrics):
            display(pd.DataFrame([metrics]).T)
            display(Image(filename=str(path / "visuals/maps.png")))
            display(Image(filename=str(path / "visuals/hyperparameters.png")))
            metadata = json.loads((path / "model_metadata.json").read_text())
            if metadata["inducing_positions_saved"]:
                display(Image(filename=str(path / "visuals/inducing_positions.png")))
            if metadata["partition_artifacts_saved"]:
                display(Image(filename=str(path / "visuals/partition.png")))
            files.download(str(archive))

        def show_worker_event(event):
            if event["kind"] == "snapshot":
                show_progress(Path(event["directory"]), event["epoch"])
            elif event["kind"] == "complete":
                show_completed(Path(event["directory"]), Path(event["archive"]), event["metrics"])
            else:
                raise ValueError(f"Unknown worker event: {event['kind']}")

        BATCH_DIRECTORY = WORK_ROOT / datetime.now(UTC).strftime("batch_%Y%m%dT%H%M%S%fZ")
        completed = call_worker("train", data=DATA, runs=RUNS,
            prepared_directory=str(PREPARED_DIRECTORY), directory=str(BATCH_DIRECTORY),
            on_event=show_worker_event)
        result = Path(completed["directory"])
        display(pd.read_json(result / "comparison.json"))
        print("Results:", result)
        """, stage="training", requires=("install", "settings", "data"), invalidates=("analysis", "download")),
        cell("markdown", """
        ## 5. Recompute metrics and figures without training

        Select latest_batch for the last successful training batch, or saved_runs
        and fill SAVED_RUNS with exact completed model-run directories. This stage
        reads saved predictions only. Original results are never overwritten.
        The saved_runs action can be used after setup without data preparation or
        training, including to inspect earlier completed runs after another fails.
        """),
        cell("code", """
        from datetime import UTC, datetime
        from IPython.display import Image, display
        import pandas as pd

        ANALYSIS_SOURCE = "latest_batch"  # "latest_batch" or "saved_runs"
        SAVED_RUNS = []  # Exact completed model-run paths, not whole batch paths.
        if ANALYSIS_SOURCE == "latest_batch":
            if not _OISST_STAGES.get("training", False):
                raise RuntimeError("Latest training stage failed or has not run; explicitly select saved_runs instead")
            analysis_batch = str(BATCH_DIRECTORY)
        elif ANALYSIS_SOURCE == "saved_runs":
            analysis_batch = None
        else:
            raise ValueError("ANALYSIS_SOURCE must be latest_batch or saved_runs")
        ANALYSIS_DIRECTORY = ROOT / "results/oisst" / datetime.now(UTC).strftime("analysis_%Y%m%dT%H%M%S%fZ")
        call_worker("analysis", analysis_source=ANALYSIS_SOURCE, batch_directory=analysis_batch,
                    saved_runs=SAVED_RUNS, directory=str(ANALYSIS_DIRECTORY))
        analysis_table = pd.read_json(ANALYSIS_DIRECTORY / "comparison.json")
        display(analysis_table)
        for path in analysis_table["analysis_directory"]:
            display(Image(filename=str(Path(path) / "visuals/maps.png")))
        print("Analysis:", ANALYSIS_DIRECTORY)
        """, stage="analysis", requires=("install",), invalidates=("download",)),
        cell("markdown", "## 6. Download a batch, analysis or explicitly named directory"),
        cell("code", """
        from datetime import UTC, datetime
        from google.colab import files

        DOWNLOAD_SOURCE = "latest_batch"  # "latest_batch", "analysis" or "saved_directory"
        SAVED_DIRECTORY = None
        if DOWNLOAD_SOURCE == "latest_batch":
            if not _OISST_STAGES.get("training", False) or not _OISST_STAGES.get("analysis", False):
                raise RuntimeError("Complete the latest training/analysis stages or explicitly select saved_directory")
            download_directory = BATCH_DIRECTORY
        elif DOWNLOAD_SOURCE == "analysis":
            if not _OISST_STAGES.get("analysis", False):
                raise RuntimeError("Complete the analysis stage first")
            download_directory = ANALYSIS_DIRECTORY
        elif DOWNLOAD_SOURCE == "saved_directory":
            if SAVED_DIRECTORY is None:
                raise ValueError("Set SAVED_DIRECTORY explicitly")
            download_directory = Path(SAVED_DIRECTORY)
        else:
            raise ValueError("Unknown DOWNLOAD_SOURCE")
        archived = call_worker("archive", directory=str(download_directory), destination=str(
            ROOT / "results/oisst" / datetime.now(UTC).strftime("download_%Y%m%dT%H%M%S%fZ.zip")))
        archive = Path(archived["archive"])
        files.download(str(archive))
        """, stage="download", requires=("install",)),
    ]
    notebook = {"cells": cells, "metadata": {"accelerator": "GPU",
        "colab": {"name": "OISST_Colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "gpw_source": source_metadata()}, "nbformat": 4, "nbformat_minor": 5}
    target = Path(root) / "notebooks/OISST_Colab.ipynb"
    if preserve_settings and target.exists():
        original = target.read_bytes()
        backup = Path(root) / "results/notebook_backups" / (hashlib.sha256(original).hexdigest() + ".ipynb")
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            backup.write_bytes(original)
        controls = {"MODELS", "OVERRIDES", "DATA_OVERRIDES", "DATA_ACTION", "PREPARED_PATH",
                    "ANALYSIS_SOURCE", "SAVED_RUNS", "DOWNLOAD_SOURCE", "SAVED_DIRECTORY"}
        saved_values = {}
        for old_cell in json.loads(target.read_text(encoding="utf-8"))["cells"]:
            if old_cell["cell_type"] == "code":
                for node in ast.walk(ast.parse("".join(old_cell["source"]))):
                    if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                            and node.targets[0].id in controls):
                        saved_values[node.targets[0].id] = node.value
        if "MODELS" in saved_values and ast.literal_eval(saved_values["MODELS"]) == [
                "SVGP", "GPW", "ProSpar", "VNNGP", "Standard Patchwork"]:
            saved_values["MODELS"].elts.append(ast.Constant("Standard Patchwork (GPW regions)"))
        for new_cell in cells:
            if new_cell["cell_type"] == "code":
                tree = ast.parse("".join(new_cell["source"]))
                for node in ast.walk(tree):
                    if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                            and node.targets[0].id in saved_values):
                        node.value = saved_values[node.targets[0].id]
                source = ast.unparse(tree) + "\n"
                new_cell["source"] = source.splitlines(keepends=True)
                new_cell["id"] = hashlib.sha256(source.encode()).hexdigest()[:12]
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1], preserve_settings=True))
