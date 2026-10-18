import pytest
import torch

from pwgp.models.patchwork import GeneralizedPatchworkGP, agreement_weights


def _model(rank: int = 6) -> GeneralizedPatchworkGP:
    torch.manual_seed(29)
    supports = 0.1 + 0.8 * torch.rand(3, 4, 2, dtype=torch.float64)
    edges = [(0, 1), (1, 2), (0, 2)]
    stitches = 0.1 + 0.8 * torch.rand(3, 2, 2, dtype=torch.float64)
    lengthscales = torch.tensor(
        [[0.2, 0.35], [0.45, 0.23], [0.3, 0.5]], dtype=torch.float64
    )
    return GeneralizedPatchworkGP(
        supports, stitches, edges, q_rank=rank, lengthscale=lengthscales
    )


def test_structured_prior_matches_dense_reference() -> None:
    model = _model()
    prior = model.prior()
    local = torch.block_diag(*prior["local_covariance"].unbind(0))
    cross = prior["support_constraint"].reshape(12, -1)
    dense = local - cross @ torch.cholesky_solve(
        cross.T, prior["raw_constraint_cholesky"]
    )
    torch.testing.assert_close(
        prior["logdet"], torch.linalg.slogdet(dense)[1], rtol=1e-8, atol=1e-8
    )
    torch.testing.assert_close(
        model.q.kl(prior), torch.zeros((), dtype=torch.float64), rtol=0, atol=1e-7
    )


def test_fusion_and_variational_algebra() -> None:
    model = _model(rank=3)
    x = torch.rand(8, 2, dtype=torch.float64)
    prior = model.prior()
    weights, fusion = model.regression_weights(x, prior, return_fusion=True)
    torch.testing.assert_close(
        fusion.sum(-1), torch.ones(len(x), dtype=torch.float64), rtol=1e-9, atol=1e-9
    )
    assert weights.shape == (8, 3, 4)
    mean, variance = model.latent(x, prior)
    assert mean.shape == variance.shape == (8,)
    assert bool(torch.isfinite(variance).all())

    loss = -model.elbo_batch(x, torch.randn(8, dtype=torch.float64), len(x))[0]
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert bool(torch.isfinite(parameter.grad).all()), name


def test_agreement_weights_rejects_semidefinite_covariance() -> None:
    covariance = torch.ones(2, 3, 3, dtype=torch.float64)
    with pytest.raises(torch.linalg.LinAlgError):
        agreement_weights(covariance)


def test_no_stitch_limit_uses_inverse_residual_variance() -> None:
    torch.manual_seed(4)
    supports = torch.rand(2, 3, 2, dtype=torch.float64)
    stitches = torch.empty(0, 0, 2, dtype=torch.float64)
    model = GeneralizedPatchworkGP(supports, stitches, [], q_rank=0)
    x = torch.rand(5, 2, dtype=torch.float64)
    prior = model.prior()
    residual = model.local_query_terms(x, prior)["residual"]
    inverse = 1 / torch.diagonal(residual, dim1=-2, dim2=-1)
    expected = inverse / inverse.sum(-1, keepdim=True)
    torch.testing.assert_close(model.fusion_weights(x, prior), expected, rtol=1e-8, atol=1e-8)

