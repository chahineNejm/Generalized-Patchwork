import copy
import io

import pytest
import torch
from tqdm.std import tqdm

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


def test_progress_refreshes_every_twenty_epochs_and_on_early_stop(monkeypatch):
    from pwgp import training

    torch.set_num_threads(2)
    frames = []

    class RecordingProgress(tqdm):
        def __init__(self, *args, **kwargs):
            assert kwargs["maxinterval"] == float("inf")
            super().__init__(*args, file=io.StringIO(), **kwargs)

        def display(self, *args, **kwargs):
            frames.append((self.n, self.desc, self.postfix))
            return super().display(*args, **kwargs)

    monkeypatch.setattr(training, "tqdm", RecordingProgress)
    x = torch.linspace(.1, .9, 12, dtype=torch.float64)[:, None]
    first = SVGP(x[::3].clone())
    second = copy.deepcopy(first)
    optimizer = OptimizerConfig(learning_rate_q=.001, learning_rate_supports=.001,
                                learning_rate_stitches=0., learning_rate_hyperparameters=0.)
    seen = []
    label = "train:oisst / GPW seed31 rank0 stitches-none"

    def callback(epoch, model, history):
        seen.append(epoch)
        return epoch == 43

    sparse = fit(first, x, x[:, 0], epochs=45, batch_size=12, optimizer=optimizer,
                 on_epoch=callback, progress_description=label, progress_every_epochs=20)
    assert [f[0] for f in frames] == [0, 20, 40, 43]
    assert all(f[1] == label for f in frames)
    assert "loss=" in frames[-1][2]
    assert seen == list(range(1, 44)) and len(sparse.losses) == 43
    ordinary = fit(second, x, x[:, 0], epochs=45, batch_size=12, optimizer=optimizer,
                   on_epoch=lambda epoch, *_: epoch == 43)
    assert sparse.losses == ordinary.losses
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_progress_interval_rejects_invalid_values(interval):
    x = torch.linspace(.1, .9, 12, dtype=torch.float64)[:, None]
    with pytest.raises(ValueError, match="progress_every_epochs"):
        fit(SVGP(x[::3]), x, x[:, 0], epochs=1, batch_size=12, progress_every_epochs=interval)
