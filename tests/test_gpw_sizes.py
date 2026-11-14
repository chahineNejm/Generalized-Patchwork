import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

from experiments import multidataset_worker as worker
from experiments.build_gpw_sizes_kaggle import build_notebook

ROOT = Path(__file__).resolve().parents[1]


def notebook_cells(tmp_path):
    (tmp_path / "notebooks").mkdir()
    notebook = json.loads(build_notebook(tmp_path).read_text())
    assert notebook == json.loads((ROOT / "notebooks/GPW_Two_Sizes_Kaggle.ipynb").read_text())
    for entry in notebook["cells"]:
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), "two-sizes", "exec")
            assert entry["outputs"] == []
    return notebook["cells"]


def environment(tmp_path, monkeypatch, cells, *, seeds="[31]"):
    display = ModuleType("IPython.display")
    display.display = lambda *a, **kw: SimpleNamespace(update=lambda *a: None)
    display.FileLink = lambda path: path
    display.Image = lambda **kw: None
    monkeypatch.setitem(sys.modules, "IPython", ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    calls = []

    def call_worker(action, **values):
        calls.append((action, values))
        assert action == "settings"
        return worker.make_plan(values)

    scope = {"call_worker": call_worker, "Path": Path, "WORK_ROOT": tmp_path / "results",
             "DATA_ROOT": tmp_path / "data", "_OISST_STAGES": {"install": True}}
    source = "".join(cells[5]["source"]).replace("SEEDS = [31]", "SEEDS = " + seeds)
    execute(source, scope)
    return scope, calls


def execute(source, scope):
    exec(compile(source, "generated-notebook", "exec"), scope)  # noqa: S102 - execute our generated cells in tests


@pytest.mark.parametrize("seeds,count", [("[31]", 18), ("[31, 32, 33]", 54)])
def test_capacity_plans_have_exact_sizes_stitches_and_shared_data(tmp_path, monkeypatch, seeds, count):
    cells = notebook_cells(tmp_path)
    scope, calls = environment(tmp_path, monkeypatch, cells, seeds=seeds)
    schedule = scope["SCHEDULE"]
    assert sum(item["total_runs"] for item in schedule) == count
    assert schedule[0]["plan"]["data"] == schedule[1]["plan"]["data"]
    expected = [(256, 8, 8), (1024, 4, 16)]
    for item, configuration in zip(schedule, expected, strict=True):
        configs = [r["config"] for r in item["plan"]["runs"]]
        assert {(c["inducing_points"], c["patches"], c["stitches_per_edge"]) for c in configs} == {configuration}
        assert {c["q_rank"] for c in configs} == {64}
        assert {c["stitch_mode"] for c in configs} == {"none", "fixed", "learned"}
        assert {(c["epochs"], c["warmup_epochs"]) for c in configs} == {(100, 20)}
        assert {s["seed"] for s in item["plan"]["data"]} == {31}
    assert len(calls) == 2


def test_preflight_failure_blocks_preparation_and_training(tmp_path, monkeypatch):
    cells = notebook_cells(tmp_path)
    scope, calls = environment(tmp_path, monkeypatch, cells)

    def fail(action, **values):
        assert action == "preflight"
        raise ValueError("explicit preflight failure")

    scope["call_worker"] = fail
    with pytest.raises(ValueError, match="explicit preflight failure"):
        execute("".join(cells[7]["source"]), scope)
    with pytest.raises(RuntimeError, match="preflight"):
        execute("".join(cells[9]["source"]), scope)
    assert not scope["_OISST_STAGES"]["preflight"]
    assert len(calls) == 2


@pytest.mark.parametrize("fail_second", [False, True])
def test_both_sizes_reuse_data_and_archive_before_next_batch(tmp_path, monkeypatch, fail_second):
    cells = notebook_cells(tmp_path)
    scope, _ = environment(tmp_path, monkeypatch, cells)
    calls = []

    def call_worker(action, **values):
        calls.append((action, values))
        if action == "train":
            if fail_second and sum(a == "train" for a, _ in calls) == 2:
                raise ValueError("second capacity failed")
            directory = Path(values["directory"])
            directory.mkdir(parents=True)
            pd.DataFrame([{"dataset": "tiny", "rmse": .1}]).to_csv(directory / "comparison.csv", index=False)
        return {"directory": values["directory"]} if action != "archive" else {"archive": values["destination"]}

    scope["call_worker"] = call_worker
    execute("".join(cells[7]["source"]), scope)
    execute("".join(cells[9]["source"]), scope)
    assert [a for a, _ in calls] == ["preflight", "preflight", "prepare", "load"]
    assert calls[2][1]["directory"] == calls[3][1]["directory"]
    source = "".join(cells[11]["source"])
    with pytest.raises(RuntimeError, match="Training disabled"):
        execute(source, scope)
    enabled = source.replace("RUN_TRAINING = False", "RUN_TRAINING = True")
    if fail_second:
        with pytest.raises(ValueError, match="second capacity failed"):
            execute(enabled, scope)
        assert [a for a, _ in calls][-3:] == ["train", "archive", "train"]
        assert not scope["_OISST_STAGES"]["training"]
        with pytest.raises(RuntimeError, match="training"):
            execute("".join(cells[13]["source"]), scope)
        return
    execute(enabled, scope)
    assert [a for a, _ in calls][-4:] == ["train", "archive", "train", "archive"]
    assert calls[-4][1]["prepared_directory"] == calls[-2][1]["prepared_directory"]
    assert calls[-4][1]["directory"] != calls[-2][1]["directory"]
    assert (scope["STUDY_DIRECTORY"] / "comparison.csv").is_file()
    assert scope["_OISST_STAGES"]["training"]
    execute("".join(cells[13]["source"]), scope)
    with zipfile.ZipFile(scope["ARCHIVE_PATH"]) as bundle:
        assert {"schedule.json", "comparison.csv", "8x32_stitches8/comparison.csv",
                "4x256_stitches16/comparison.csv"} <= set(bundle.namelist())
