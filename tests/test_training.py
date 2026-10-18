import torch

from pwgp import SVGP, GeneralizedPatchworkGP, fit


def test_shared_training_path_runs_for_both_models() -> None:
    torch.manual_seed(12)
    x = 0.05 + 0.9 * torch.rand(24, 2, dtype=torch.float64)
    y = torch.sin(x[:, 0])

    svgp = SVGP(x[:5].clone())
    svgp_history = fit(svgp, x, y, epochs=1, batch_size=12)

    supports = torch.stack((x[:4], x[4:8]))
    stitches = torch.empty(0, 0, 2, dtype=torch.float64)
    patchwork = GeneralizedPatchworkGP(supports, stitches, [], q_rank=0)
    patchwork_history = fit(patchwork, x, y, epochs=1, batch_size=12)

    assert len(svgp_history.losses) == 1
    assert len(patchwork_history.losses) == 1
    assert torch.isfinite(torch.tensor(svgp_history.losses)).all()
    assert torch.isfinite(torch.tensor(patchwork_history.losses)).all()

