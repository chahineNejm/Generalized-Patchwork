import torch

from experiments.residual_weight_order import local_posterior_terms
from pwgp import GeneralizedPatchworkGP


def test_local_covariance_matches_dense_variational_projection():
    torch.manual_seed(31)
    dtype = torch.float64
    model = GeneralizedPatchworkGP(
        0.1 + 0.8 * torch.rand(3, 4, 2, dtype=dtype),
        0.1 + 0.8 * torch.rand(2, 2, 2, dtype=dtype),
        [(0, 1), (1, 2)], q_rank=3,
    )
    x = torch.rand(7, 2, dtype=dtype)
    prior, state = model.prior(), model.q.state()
    a, mean, covariance, residual = local_posterior_terms(model, x, prior, state)
    local = state["local"]
    correction = state["correction"].reshape(12, -1)
    s = torch.block_diag(*(block @ block.T for block in local)) - correction @ correction.T
    flat = a.reshape(7, 3, 12)
    torch.testing.assert_close(covariance, flat @ s @ flat.transpose(-1, -2))
    torch.testing.assert_close(mean, (flat @ model.q.mean.flatten()).squeeze(-1))
    alpha = model.fusion_weights(x, prior)
    w = torch.einsum("bi,bipm->bpm", alpha, a)
    torch.testing.assert_close(w, model.regression_weights(x, prior))
    expected = model.q.variance_from_weights(w, state)
    torch.testing.assert_close(expected, torch.einsum("bi,bij,bj->b", alpha, covariance, alpha))
    assert torch.linalg.eigvalsh(residual).min() >= -1e-12
