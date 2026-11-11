import math

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


def test_agreement_weights_regularizes_semidefinite_covariance() -> None:
    covariance = torch.ones(2, 3, 3, dtype=torch.float64)
    weights = agreement_weights(covariance, epsilon=1e-8)
    assert bool(torch.isfinite(weights).all())
    torch.testing.assert_close(
        weights.sum(-1), torch.ones(2, dtype=torch.float64), rtol=0, atol=1e-12
    )


def test_no_stitch_limit_uses_full_local_posterior_covariance() -> None:
    torch.manual_seed(4)
    supports = torch.rand(2, 3, 2, dtype=torch.float64)
    stitches = torch.empty(0, 0, 2, dtype=torch.float64)
    model = GeneralizedPatchworkGP(supports, stitches, [], q_rank=0)
    x = torch.rand(5, 2, dtype=torch.float64)
    prior = model.prior()
    covariance = model.local_prediction_terms(x, prior, model.q.state())["predictive_covariance"]
    contrasts = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    identity = torch.eye(2, dtype=torch.float64)
    delta = contrasts @ covariance @ contrasts.T + 1e-8
    rhs = contrasts @ covariance[..., :, 0, None]
    coefficients = torch.linalg.solve(delta, rhs).squeeze(-1)
    expected = identity[0] - coefficients @ contrasts
    torch.testing.assert_close(model.fusion_weights(x, prior), expected, rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("rank", [0, 3, 6])
def test_full_posterior_weights_and_residual_match_dense_reference(rank) -> None:
    model = _model(rank=rank)
    with torch.no_grad():
        model.q.raw_inverse_d_cholesky.mul_(0.8)
        model.q.mean.normal_(std=0.2)
    x = torch.rand(7, 2, dtype=torch.float64)
    prior, state = model.prior(), model.q.state()
    terms = model.local_query_terms(x, prior)
    eye = torch.eye(3, dtype=x.dtype)
    a = torch.einsum("ij,bim->bijm", eye, terms["phi0"])
    a = a - torch.einsum("biq,pmq->bipm", terms["solved_cross"], prior["w"])
    a = a.reshape(len(x), 3, 12)
    local = state["local"]
    correction = state["correction"].reshape(12, rank)
    s = torch.block_diag(*(block @ block.T for block in local)) - correction @ correction.T
    inducing = a @ s @ a.transpose(-2, -1)
    covariance = inducing + terms["residual"]
    efficient = model.local_prediction_terms(x, prior, state)
    torch.testing.assert_close(efficient["inducing_covariance"], inducing)
    torch.testing.assert_close(efficient["predictive_covariance"], covariance)
    contrasts = eye[0][None] - eye[1:]
    system = contrasts @ covariance @ contrasts.T + model.numerical.weight_epsilon * torch.eye(2, dtype=x.dtype)
    b = torch.linalg.solve(system, contrasts @ covariance[..., :, 0, None]).squeeze(-1)
    alpha = eye[0] - b @ contrasts
    torch.testing.assert_close(model.fusion_weights(x, prior), alpha)
    local_mean = a @ model.q.mean.flatten()
    expected_mean = (alpha * local_mean).sum(-1)
    expected_variance = torch.einsum("bi,bij,bj->b", alpha, covariance, alpha)
    mean, variance = model.latent(x, prior, state)
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(variance, expected_variance)
    y = torch.sin(x[:, 0])
    objective = model.elbo_batch(x, y, len(x))[0]
    noise = model.hyp.noise_std().square()
    expected = (-0.5 * (torch.log(2 * math.pi * noise)
                       + ((y - expected_mean).square() + expected_variance) / noise)).sum()
    torch.testing.assert_close(objective, expected - model.kl(prior, state))
    predicted_mean, observed_variance = model.predict(x)
    torch.testing.assert_close(predicted_mean, expected_mean)
    torch.testing.assert_close(observed_variance, expected_variance + noise)
    gradient = torch.autograd.grad((alpha[:, 0] * torch.arange(len(x), dtype=x.dtype)).sum(),
                                   model.q.raw_inverse_d_cholesky)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 1e-8


def test_single_patch_retains_conditional_residual_far_from_supports() -> None:
    supports = torch.tensor([[[0.10], [0.15], [0.20]]], dtype=torch.float64)
    model = GeneralizedPatchworkGP(
        supports, torch.empty(0, 0, 1, dtype=torch.float64), [], q_rank=0,
        lengthscale=0.05, outputscale=1.0,
    )
    x = torch.tensor([[0.9]], dtype=torch.float64)
    mean, variance = model.predict(x, observation_noise=False)
    torch.testing.assert_close(variance, torch.ones_like(variance), atol=1e-8, rtol=0)
    assert mean.shape == (1,)

