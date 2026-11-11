import torch

from pwgp import SVGP, GeneralizedPatchworkGP, fit
from pwgp.config import OptimizerConfig


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


def test_zero_warmup_updates_patchwork_hyperparameters_immediately() -> None:
    torch.manual_seed(31)
    x = 0.05 + 0.9 * torch.rand(20, 1, dtype=torch.float64)
    y = torch.sin(6 * x[:, 0])
    supports = torch.stack((x[:4], x[4:8]))
    stitches = torch.empty(0, 0, 1, dtype=torch.float64)
    model = GeneralizedPatchworkGP(
        supports,
        stitches,
        [],
        q_rank=0,
        train_supports=False,
    )
    initial = model.hyp.raw_lengthscales.detach().clone()
    optimizer = OptimizerConfig(
        learning_rate_q=2e-3,
        learning_rate_supports=5e-2,
        learning_rate_stitches=5e-3,
        learning_rate_hyperparameters=1e-2,
        warmup_fraction=0.0,
    )

    fit(model, x, y, epochs=1, batch_size=len(x), optimizer=optimizer)

    assert not torch.equal(model.hyp.raw_lengthscales.detach(), initial)


def test_epoch_callback_observes_completed_epochs() -> None:
    x = torch.linspace(0.1, 0.9, 12, dtype=torch.float64)[:, None]
    model = SVGP(x[::3].clone())
    seen = []

    def record(epoch, current_model, history):
        assert current_model is model
        assert len(history.losses) == epoch
        assert history.elapsed_seconds > 0
        seen.append(epoch)

    fit(model, x, x[:, 0], epochs=2, batch_size=12, on_epoch=record)
    assert seen == [1, 2]
