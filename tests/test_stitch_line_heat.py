import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.stitch_line_heat import RunConfig, StraightLineParameter, build_model, run
from experiments.stitch_line_preview import LayoutConfig, layout
from pwgp import GeneralizedPatchworkGP
from pwgp.config import NumericalConfig
from pwgp.parameters import UnitCubeParameter


def test_production_stitches_move_independently_without_line_constraint():
    model = build_model(RunConfig(), 0)
    assert isinstance(model.stitch, UnitCubeParameter)
    assert model.stitch.raw.shape == (1, 10, 2)
    initial = model.stitches().detach().clone()
    with torch.no_grad():
        model.stitch.raw[0, 4, 0].add_(0.1)
    final = model.stitches().detach()
    assert final[0, 4, 0] != initial[0, 4, 0]
    mask = torch.ones(10, dtype=torch.bool)
    mask[4] = False
    torch.testing.assert_close(final[0, mask], initial[0, mask])


def test_line_geometry_matches_approved_starts_and_has_correct_gradients():
    config = LayoutConfig()
    for index, angle in enumerate(config.line_angles):
        _, stitches, center, _ = layout(config, index)
        line = StraightLineParameter(center, angle, 10, 0.05)
        torch.testing.assert_close(line()[0], torch.from_numpy(stitches), rtol=0, atol=1e-14)
        with torch.no_grad():
            line.raw.add_(torch.tensor([0.2, 0.3], dtype=torch.float64))
        points = line()[0]
        vector = points[-1] - points[0]
        differences = points - points[0]
        torch.testing.assert_close(differences[:, 0] * vector[1] - differences[:, 1] * vector[0],
                                   torch.zeros(10, dtype=torch.float64), atol=1e-14, rtol=0)
        assert bool(((points > 0) & (points < 1)).all())
    line = StraightLineParameter([0.54, 0.48], 41, 10, 0.05)
    assert torch.autograd.gradcheck(lambda raw: torch.func.functional_call(line, {"raw": raw}), (line.raw,))


def test_model_matches_production_reference_at_identical_points():
    config = RunConfig()
    h = config.layouts.heat
    model = build_model(config, 1)
    reference = GeneralizedPatchworkGP(model.supports().detach(), model.stitches().detach(), [(0, 1)],
        q_rank=h.q_rank, lengthscale=h.lengthscale, outputscale=h.outputscale, noise_std=h.noise_std,
        numerical=NumericalConfig(weight_epsilon=h.weight_epsilon))
    reference.q.load_state_dict(model.q.state_dict())
    torch.manual_seed(21)
    x = 0.05 + 0.9 * torch.rand(12, 2, dtype=torch.float64)
    y = torch.sin(x[:, 0] * 9)
    expected = reference.elbo_batch(x, y, 12)[0]
    actual = model.elbo_batch(x, y, 12)[0]
    torch.testing.assert_close(actual, expected)
    expected_points_grad = torch.autograd.grad(expected, reference.stitch.raw)[0]
    points = reference.stitches().detach()
    expected_coordinate_grad = expected_points_grad / (points * (1 - points))
    chain_grad = torch.autograd.grad(model.stitches(), model.stitch.raw, expected_coordinate_grad, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, model.stitch.raw)[0]
    torch.testing.assert_close(actual_grad, chain_grad, rtol=1e-7, atol=1e-7)


def test_line_heat_runner_saves_motion_and_scores(tmp_path, monkeypatch):
    base = RunConfig()
    heat = replace(base.layouts.heat, cells=12, time_steps=3, supports_per_patch=8, stitches=4, q_rank=2)
    layouts = replace(base.layouts, heat=heat, line_angles=(90,), line_centers=((0.56, 0.5),), line_names=("Shifted",))
    config = replace(base, layouts=layouts, epochs=2, warmup_epochs=1, observations=40, snapshot_every=1)
    def no_git(*args):
        raise AssertionError("Git must not be called when capture_git=False")

    monkeypatch.setattr("experiments.stitch_line_heat._git", no_git)
    snapshots, completed = [], []
    output = run(config, directory=tmp_path / "run", capture_git=False,
        on_snapshot=lambda folder, epoch: snapshots.append((folder, epoch)),
        on_complete=lambda folder, metrics: completed.append((folder, metrics)))
    assert json.loads((output / "status.json").read_text())["state"] == "complete"
    metrics = json.loads((output / "metrics.json").read_text())[0]
    assert np.isfinite(metrics["nlpd_standardized"])
    assert metrics["mean_stitch_displacement"] > 0
    assert (output / "layout_0/model.pt").is_file()
    assert [epoch for _, epoch in snapshots] == [1, 2]
    assert len(completed) == 1 and completed[0][0] == output / "layout_0"
    assert json.loads((output / "provenance.json").read_text())["git_head"] is None
    kernels = json.loads((output / "layout_0/kernel_history.json").read_text())
    assert [row["epoch"] for row in kernels] == [0, 1, 2]
    np.testing.assert_allclose(kernels[0]["lengthscales"], np.full((2, 2), heat.lengthscale))
    np.testing.assert_equal(kernels[0]["lengthscales"], kernels[1]["lengthscales"])
    assert kernels[0]["noise_std"] == kernels[1]["noise_std"]
    for key in ("lengthscales", "outputscales", "noise_std"):
        np.testing.assert_allclose(kernels[-1][key], metrics[key])
        assert np.isfinite(np.asarray([row[key] for row in kernels])).all()
    assert (output / "layout_0/kernel_evolution.png").stat().st_size > 0


def test_requested_cuda_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        run(RunConfig(device="cuda:0"), directory=tmp_path / "no_cuda")
    assert not (tmp_path / "no_cuda").exists()
