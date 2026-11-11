import ast
import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments import oisst_benchmark as benchmark
from experiments.build_heaton_colab import cell
from experiments.build_oisst_colab import build_notebook
from experiments.default_hyperparameters import oisst_defaults
from experiments.exploratory_analysis import recompute_saved_runs
from experiments.run_selection import select_runs
from pwgp.data.download import file_checksum
from pwgp.data.types import RegressionDataset


def source_zip():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("source.txt", "synthetic source fixture")
    return stream.getvalue()


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    config, runs = oisst_defaults(device="cpu")
    rng = np.random.default_rng(19)
    x = rng.uniform([281, 21], [329, 49], size=(100, 2))
    y = 10 + np.sin(x[:, 0]) + np.cos(x[:, 1])
    split = benchmark.prepare_split(RegressionDataset("fixture", x, y, ("lon", "lat"), "C"), config)
    monkeypatch.setattr(benchmark, "acquire", lambda *a, **kw: (split, {"source": "synthetic fixture"}))
    directory = benchmark.prepare_experiment(config, directory=tmp_path / "prepared",
        data_root=tmp_path, source_provenance={"mode": "fixture"}, source_zip=source_zip())

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit reuse must not acquire or preprocess data")

    monkeypatch.setattr(benchmark, "acquire", forbidden)
    monkeypatch.setattr(benchmark, "prepare_split", forbidden)
    return config, runs, directory, split


def test_load_prepared_reuses_exact_data_and_allows_output_only_changes(prepared):
    config, _, directory, initial = prepared
    loaded = benchmark.load_prepared(directory, data_config=replace(config, save_inducing_positions=False))
    for key in ("x", "y"):
        torch.testing.assert_close(loaded[key], initial[key], atol=0, rtol=0)
    for key in ("train", "validation", "test", "raw_x", "raw_y"):
        np.testing.assert_array_equal(loaded[key], initial[key])


def test_prepared_mismatch_and_corruption_raise_without_rebuilding(prepared, tmp_path):
    config, runs, directory, _ = prepared
    with pytest.raises(ValueError, match="Data settings changed"):
        benchmark.load_prepared(directory, data_config=replace(config, seed=32))
    failed = tmp_path / "mismatch"
    with pytest.raises(ValueError, match="Data settings changed"):
        benchmark.run_prepared_comparison(replace(config, bounds=(-79., -30., 20., 50.)), runs[:1],
            prepared_directory=directory, directory=failed,
            source_provenance={"mode": "fixture"}, source_zip=source_zip())
    assert json.loads((failed / "data_config.json").read_text())["bounds"][0] == -79
    path = directory / "prepared.npz"
    path.write_bytes(path.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch: prepared.npz"):
        benchmark.load_prepared(directory, data_config=config)


def test_failed_preparation_is_never_reused(prepared):
    config, _, directory, _ = prepared
    benchmark.write_json(directory / "status.json", {"state": "failed"})
    with pytest.raises(ValueError, match="must be complete"):
        benchmark.load_prepared(directory, data_config=config)


def test_real_prepared_training_and_reanalysis_round_trip(prepared, tmp_path):
    config, runs, directory, _ = prepared
    selected = select_runs(runs, models=["SVGP"], overrides={"SVGP": {
        "epochs": 1, "warmup_epochs": 0, "inducing_points": 4,
        "batch_size": 64, "prediction_batch_size": 16, "snapshot_every": 1}})
    batch = benchmark.run_prepared_comparison(config, selected, prepared_directory=directory,
        directory=tmp_path / "real_batch", source_provenance={"mode": "fixture"}, source_zip=source_zip())
    paths = benchmark.completed_run_directories(batch)
    analysis = recompute_saved_runs(paths, destination=tmp_path / "real_analysis")
    original = json.loads((paths[0] / "metrics.json").read_text())
    recomputed = json.loads((analysis / "00/metrics.json").read_text())
    for key in ("rmse", "mae", "nlpd", "crps", "coverage_95", "mean_interval_width"):
        assert recomputed[key] == pytest.approx(original[key], abs=1e-12)
    assert json.loads((paths[0] / "status.json").read_text())["state"] == "complete"


def test_explicit_retry_trains_only_selected_models_in_a_new_batch(prepared, tmp_path, monkeypatch):
    config, runs, directory, _ = prepared
    calls = []
    failing = [True]

    def fake_model(model_config, split, *, directory, **options):
        calls.append(model_config.model)
        directory.mkdir()
        if model_config.model == "gpw" and failing[0]:
            raise RuntimeError("original GPW failure")
        benchmark.write_json(directory / "metrics.json", {"nlpd": 1.0})
        benchmark.write_json(directory / "status.json", {"state": "complete"})
        return directory

    monkeypatch.setattr(benchmark, "run_model", fake_model)
    kwargs = {"prepared_directory": directory, "source_provenance": {"mode": "fixture"},
              "source_zip": source_zip()}
    initial = select_runs(runs, models=["SVGP", "GPW", "ProSpar"], overrides={})
    first = tmp_path / "first"
    with pytest.raises(RuntimeError, match="original GPW failure"):
        benchmark.run_prepared_comparison(config, initial, directory=first, **kwargs)
    assert calls == ["svgp", "gpw"]
    saved = file_checksum(first / "00_svgp/metrics.json")
    assert json.loads((first / "status.json").read_text())["state"] == "failed"
    with pytest.raises(ValueError, match="not complete"):
        benchmark.completed_run_directories(first)
    failing[0] = False
    second = benchmark.run_prepared_comparison(config, select_runs(runs, models=["GPW"], overrides={}),
                                               directory=tmp_path / "retry", **kwargs)
    assert calls == ["svgp", "gpw", "gpw"]
    assert file_checksum(first / "00_svgp/metrics.json") == saved
    assert benchmark.completed_run_directories(second) == [second / "00_gpw"]
    with pytest.raises(FileExistsError):
        benchmark.run_prepared_comparison(config, initial, directory=second, **kwargs)


def test_failed_stage_blocks_dependents_but_can_be_explicitly_retried():
    namespace = {"_OISST_STAGES": {"install": True, "settings": True, "data": True,
                                  "training": True, "analysis": True, "download": True}}
    failed = "".join(cell("code", 'raise RuntimeError("original stage error")', stage="training",
                         requires=("data", "settings"), invalidates=("analysis", "download"))["source"])
    later = "".join(cell("code", "unexpected = True", stage="analysis", requires=("training",))["source"])
    with pytest.raises(RuntimeError, match="original stage error"):
        exec(failed, namespace)  # noqa: S102
    with pytest.raises(RuntimeError, match="training stage"):
        exec(later, namespace)  # noqa: S102
    assert "unexpected" not in namespace
    assert namespace["_OISST_STAGES"]["data"]
    assert not namespace["_OISST_STAGES"]["download"]
    retry = "".join(cell("code", "attempt = 2", stage="training", requires=("data", "settings"))["source"])
    exec(retry, namespace)  # noqa: S102
    exec(later, namespace)  # noqa: S102
    assert namespace["attempt"] == 2 and namespace["unexpected"]


def test_settings_rerun_keeps_preparation_but_invalidates_training():
    namespace = {"_OISST_STAGES": {"install": True, "data": True, "training": True, "analysis": True}}
    code = "".join(cell("code", "MODELS = ['GPW']", stage="settings", requires=("install",),
                       invalidates=("training", "analysis", "download"))["source"])
    exec(code, namespace)  # noqa: S102
    assert namespace["_OISST_STAGES"]["data"]
    assert not namespace["_OISST_STAGES"]["training"]


@pytest.fixture
def saved_run(tmp_path):
    directory = tmp_path / "batch/00_svgp"
    directory.mkdir(parents=True)
    benchmark.write_json(directory.parent / "preprocessing.json", {"target_mean": 10.0, "target_std": 2.0})
    benchmark.write_json(directory / "status.json", {"state": "complete"})
    benchmark.write_json(directory / "model_metadata.json", {"inducing_positions_saved": False})
    benchmark.write_json(directory / "history.json", {"losses": [2.0, 1.0]})
    benchmark.write_json(directory / "hyperparameter_history.json", [{"epoch": epoch,
        "lengthscales": [[0.1, 0.2]], "outputscales": [1.0], "noise_std": 0.1} for epoch in (0, 1, 2)])
    benchmark.write_json(directory / "metrics.json", {"rmse": 999.0, "nlpd": 999.0,
        "coverage_95": 0.0, "mean_interval_width": 999.0, "training_seconds": 12.0})
    for label in ("validation", "test"):
        np.savez_compressed(directory / f"{label}_predictions.npz", truth=np.array([9., 10., 12.]),
            mean=np.array([9.1, 9.8, 11.7]), observation_variance=np.array([0.3, 0.4, 0.5]),
            coordinates=np.array([[285., 25.], [295., 30.], [310., 40.]]), indices=np.arange(3))
    return directory


def test_saved_reanalysis_recomputes_scores_without_training_or_prediction(saved_run, tmp_path, monkeypatch):
    before = {str(p): file_checksum(p) for p in saved_run.parent.rglob("*") if p.is_file()}

    def forbidden(*args, **kwargs):
        pytest.fail("Saved-result analysis must not build, load, train or predict a model")

    for name in ("build_model", "train", "predict", "acquire"):
        monkeypatch.setattr(benchmark, name, forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    output = recompute_saved_runs([saved_run], destination=tmp_path / "analysis")
    metrics = json.loads((output / "00/metrics.json").read_text())
    assert metrics["rmse"] == pytest.approx(np.sqrt((0.01 + 0.04 + 0.09) / 3))
    assert metrics["nlpd"] != 999 and metrics["training_seconds"] == 12.0
    assert json.loads((output / "00/provenance.json").read_text())["training_performed"] is False
    for name in ("training.png", "hyperparameters.png", "maps.png", "metrics.png"):
        assert (output / "00/visuals" / name).stat().st_size > 0
    assert {str(p): file_checksum(p) for p in saved_run.parent.rglob("*") if p.is_file()} == before


def test_reanalysis_failure_does_not_continue_or_use_old_metrics(saved_run, tmp_path):
    benchmark.write_json(saved_run / "status.json", {"state": "failed"})
    with pytest.raises(ValueError, match="completed model run"):
        recompute_saved_runs([saved_run], destination=tmp_path / "analysis")
    assert json.loads((tmp_path / "analysis/status.json").read_text())["state"] == "failed"
    assert json.loads((tmp_path / "analysis/00/request.json").read_text())["original_run"] == str(saved_run.resolve())
    assert not (tmp_path / "analysis/00/metrics.json").exists()


def test_archives_are_explicit_and_never_overwrite(tmp_path):
    source = tmp_path / "results"
    source.mkdir()
    (source / "metrics.json").write_text("{}")
    archive = benchmark.archive_directory(source, destination=tmp_path / "results.zip")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["metrics.json"]
    with pytest.raises(FileExistsError):
        benchmark.archive_directory(source, destination=archive)
    with pytest.raises(ValueError, match="outside"):
        benchmark.archive_directory(source, destination=source / "recursive.zip")


def test_notebook_rebuild_preserves_explicit_user_controls(tmp_path):
    (tmp_path / "notebooks").mkdir()
    path = build_notebook(tmp_path)
    notebook = json.loads(path.read_text())
    for notebook_cell in notebook["cells"]:
        if notebook_cell["cell_type"] != "code":
            continue
        source = "".join(notebook_cell["source"])
        if 'OVERRIDES = {}' in source:
            notebook_cell["source"] = source.replace('OVERRIDES = {}',
                'OVERRIDES = {"GPW": {"epochs": 2, "warmup_epochs": 1}}').splitlines(keepends=True)
    path.write_text(json.dumps(notebook))
    build_notebook(tmp_path, preserve_settings=True)
    values = []
    for notebook_cell in json.loads(path.read_text())["cells"]:
        if notebook_cell["cell_type"] == "code":
            values.extend(ast.literal_eval(node.value) for node in ast.walk(ast.parse("".join(notebook_cell["source"])))
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "OVERRIDES")
    assert values == [{"GPW": {"epochs": 2, "warmup_epochs": 1}}]
