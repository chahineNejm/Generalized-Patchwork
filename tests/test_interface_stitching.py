import json

from experiments import multidataset_worker as worker
from experiments.build_interface_stitching_kaggle import build_notebook
from experiments.default_hyperparameters import interface_stitching_defaults, multidataset_defaults


def test_defaults_plan_has_ninety_grid_runs(tmp_path):
    d = interface_stitching_defaults(device="cpu", input_root=str(tmp_path))
    result = worker.make_plan({"study": "gpw_stitch_rank", "device": "cpu", "input_root": str(tmp_path),
        "datasets": d["datasets"], "seeds": d["seeds"], "ranks": d["ranks"], "stitch_modes": d["stitch_modes"],
        "inducing_points": d["inducing_points"], "patches": d["patches"], "epochs": d["epochs"],
        "warmup_epochs": d["warmup_epochs"], "early_stopping": d["early_stopping"],
        "model_overrides": d["model_overrides"], "data_overrides": {},
        "progress_every_epochs": d["progress_every_epochs"], "save_inducing_positions": True})
    assert result["total_runs"] == 90
    for run in result["plan"]["runs"]:
        assert run["config"]["gpw_partition"] == "grid" and run["config"]["stitches_per_edge"] == 8


def test_control_dataset_has_no_gap():
    data, _, _ = multidataset_defaults(device="cpu", input_root="manual")
    spec = data["periodic_heat_jagged_full"]
    assert spec.simulation["interface_amplitude"] == 0.06 and "gap_width" not in spec.simulation


def test_notebook_is_reproducible(tmp_path):
    (tmp_path / "notebooks").mkdir()
    built = json.loads(build_notebook(tmp_path).read_text(encoding="utf-8"))
    source = "".join(built["cells"][5]["source"])
    assert 'study="gpw_stitch_rank"' in source and "periodic_heat_jagged_full" in source
