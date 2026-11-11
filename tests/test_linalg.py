import pytest
import torch

from pwgp import SVGP, GeneralizedPatchworkGP
from pwgp.linalg import cholesky_solve


@pytest.mark.parametrize("batch_shape", [(), (8,), (2, 3)])
def test_triangular_cholesky_solve_matches_native_values_and_gradients(batch_shape) -> None:
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(
        *batch_shape, 5, 5, generator=generator, dtype=torch.float64, requires_grad=True
    )
    rhs = torch.randn(
        *batch_shape, 3, 5, generator=generator, dtype=torch.float64, requires_grad=True
    ).transpose(-2, -1)
    covariance = raw @ raw.transpose(-2, -1) + torch.eye(5, dtype=raw.dtype)
    factor = torch.linalg.cholesky(covariance)
    expected = torch.cholesky_solve(rhs, factor)
    actual = cholesky_solve(rhs, factor)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    expected_grad = torch.autograd.grad(expected.square().sum(), (raw, rhs), retain_graph=True)
    actual_grad = torch.autograd.grad(actual.square().sum(), (raw, rhs))
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, reference, rtol=1e-10, atol=1e-10)


def test_triangular_cholesky_solve_preserves_broadcasting() -> None:
    factor = torch.tensor([[2.0, 0.0], [0.5, 1.0]], dtype=torch.float64)
    rhs = torch.arange(12, dtype=torch.float64).reshape(3, 2, 2)
    torch.testing.assert_close(cholesky_solve(rhs, factor), torch.cholesky_solve(rhs, factor))


def test_models_do_not_call_native_cholesky_solve(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise AssertionError("Native cholesky_solve must not be used")

    monkeypatch.setattr(torch, "cholesky_solve", unavailable)
    generator = torch.Generator().manual_seed(41)
    x = 0.1 + 0.8 * torch.rand(16, 2, generator=generator, dtype=torch.float64)
    supports = torch.stack((x[:4], x[4:8]))
    stitches = torch.tensor([[[0.5, 0.5]]], dtype=x.dtype)
    models = [GeneralizedPatchworkGP(supports, stitches, [(0, 1)], q_rank=1), SVGP(x[:6])]
    for model in models:
        loss = -model.elbo_batch(x, x[:, 0], len(x))[0]
        loss.backward()
        assert torch.isfinite(loss)
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
