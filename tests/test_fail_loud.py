import json
import sys

import pytest
import torch

from experiments.build_heaton_colab import cell
from pwgp.initialization import patch_neighbor_graph
from pwgp.run_records import record_failures


@pytest.mark.parametrize("error", [RuntimeError("training failed"), KeyboardInterrupt("stop")])
def test_failure_record_preserves_exception_and_stops_execution(tmp_path, error):
    hook = sys.excepthook
    reached = []
    with pytest.raises(type(error)) as raised:
        with record_failures(tmp_path, status_key="state"):
            raise error
        reached.append(True)
    assert raised.value is error
    assert not reached
    assert sys.excepthook is hook
    record = json.loads((tmp_path / "failure.json").read_text())
    assert record["exception_type"] == type(error).__name__
    assert str(error) in record["traceback"]
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed"


def test_success_has_no_failure_record(tmp_path):
    with record_failures(tmp_path, status_key="status"):
        pass
    assert not (tmp_path / "failure.json").exists()


def test_colab_failure_blocks_later_cells():
    namespace = {}
    failed = "".join(cell("code", 'raise RuntimeError("original failure")')["source"])
    later = "".join(cell("code", 'downstream_ran = True')["source"])
    with pytest.raises(RuntimeError, match="original failure"):
        exec(failed, namespace)  # noqa: S102
    with pytest.raises(RuntimeError, match="earlier cell failed"):
        exec(later, namespace)  # noqa: S102
    assert "downstream_ran" not in namespace


def test_colab_success_preserves_last_expression_without_repeating_it():
    namespace = {"calls": []}
    source = "".join(cell("code", "calls.append(1)\nlen(calls)")["source"])
    exec(source, namespace)  # noqa: S102
    assert namespace["calls"] == [1]
    assert namespace["_GPW_CELL_RESULT"] == 1
    assert namespace["_GPW_STAGE_FAILED"] is False


def test_patch_neighbors_are_not_capped():
    supports = torch.rand(3, 4, 2, dtype=torch.float64)
    assert len(patch_neighbor_graph(supports, neighbors=2)) == 3
    for value in (3, 100, 0, -1, 1.5, True):
        with pytest.raises((TypeError, ValueError), match="neighbors must"):
            patch_neighbor_graph(supports, neighbors=value)
    assert patch_neighbor_graph(supports[:1], neighbors=0) == []
    with pytest.raises(ValueError, match="neighbors must"):
        patch_neighbor_graph(supports[:1], neighbors=2)


def test_heaton_caught_failure_is_recorded(tmp_path, monkeypatch):
    from experiments import heaton_benchmark as benchmark

    def fail(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(benchmark, "download_file", fail)
    output = tmp_path / "heaton"
    with pytest.raises(RuntimeError, match="download failed"):
        benchmark.run_heaton(benchmark.HeatonConfig(device="cpu"),
                             run_directory=str(output), capture_git=False)
    assert json.loads((output / "status.json").read_text())["status"] == "failed"
    assert "download failed" in json.loads((output / "failure.json").read_text())["traceback"]


def test_ukhousing_initialization_failure_is_recorded(tmp_path, monkeypatch):
    from experiments import ukhousing_benchmark as benchmark

    def fail(*args, **kwargs):
        raise RuntimeError("initialization failed")

    monkeypatch.setattr(benchmark, "build_model", fail)
    output = tmp_path / "uk"
    with pytest.raises(RuntimeError, match="initialization failed"):
        benchmark.run_ukhousing(benchmark.UKConfig(device="cpu"), {}, directory=output)
    assert json.loads((output / "status.json").read_text())["state"] == "failed"
    assert "initialization failed" in json.loads((output / "failure.json").read_text())["traceback"]
