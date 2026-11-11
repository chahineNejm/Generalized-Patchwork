import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments import heaton_benchmark as heaton
from experiments.ukhousing_benchmark import UKConfig, build_model
from pwgp.config import ProSparConfig


def test_runners_require_explicit_prospar_configuration(tmp_path):
    with pytest.raises(ValueError, match="explicit temperatures"):
        heaton.run_heaton(replace(heaton.HeatonConfig(), model="prospar"),
                          run_directory=str(tmp_path / "missing"))
    split = {"x": torch.zeros(3, 2), "y": torch.zeros(3), "train": np.arange(3)}
    with pytest.raises(ValueError, match="explicit temperatures"):
        build_model(replace(UKConfig(), model="prospar", device="cpu"), split)


def test_heaton_prospar_runner_saves_configuration_geometry_and_scores(tmp_path, monkeypatch):
    grid = np.linspace(0, 1, 10)
    coordinates = np.stack(np.meshgrid(grid, grid), axis=-1).reshape(-1, 2)
    truth = np.sin(9 * coordinates[:, 0]) + np.cos(5 * coordinates[:, 1])
    training = np.arange(len(truth)) % 4 != 0
    raw = tmp_path / "data/raw/AllSatelliteTemps.RData"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"synthetic fixture, not official MODIS data")
    monkeypatch.setattr(heaton, "download_file", lambda *args, **kwargs: raw)
    monkeypatch.setattr(heaton, "load_heaton", lambda path: (coordinates, truth, training, ~training))
    config = replace(heaton.HeatonConfig(), model="prospar", device="cpu", epochs=2,
                     warmup_epochs=1, inducing_points=12, patches=3, batch_size=32,
                     prediction_batch_size=16, snapshot_every=2,
                     prospar=ProSparConfig(temperatures=2.0, exponent=1.0,
                                          inducing_nuggets=0.02, prior_jitter=0.0))
    snapshots = []
    result = heaton.run_heaton(config, data_root=tmp_path / "data",
                               run_directory=str(tmp_path / "run"), capture_git=False,
                               on_snapshot=lambda path, epoch: snapshots.append(epoch))
    assert snapshots == [1, 2]
    assert json.loads((result / "status.json").read_text())["status"] == "complete"
    assert np.isfinite(json.loads((result / "metrics.json").read_text())["nlpd"])
    assert json.loads((result / "config.json").read_text())["prospar"]["prior_jitter"] == 0.0
    assert json.loads((result / "model_metadata.json").read_text())["model_class"].endswith("ProSparGP")
    learned = json.loads((result / "learned_parameters.json").read_text())
    assert len(learned["temperatures"]) == 3
    assert [len(d) for d in learned["inducing_nuggets"]] == [4, 4, 4]
    with np.load(result / "geometry_history.npz") as geometry:
        assert geometry["supports"].shape == (3, 3, 4, 2)
    for name in ("model.pt", "checkpoint.pt", "progress.png", "maps.png"):
        assert (result / name).stat().st_size > 0
