import math

import pytest
import torch

from pwgp import ProSparGP, fit, predict_batched
from pwgp.config import OptimizerConfig
from pwgp.kernels import ard_rbf
from pwgp.parameters import inverse_softplus


def example(*, prior_jitter=0.0, **options):
    supports = [torch.tensor([[0.12, 0.2], [0.3, 0.6], [0.4, 0.4]], dtype=torch.float64),
                torch.tensor([[0.55, 0.3], [0.7, 0.65], [0.85, 0.8], [0.9, 0.4]], dtype=torch.float64)]
    model = ProSparGP(supports, temperatures=torch.tensor([1.3, 0.7]), exponent=1.2,
                     inducing_nuggets=[torch.tensor([0.02, 0.03, 0.04]), 0.05],
                     prior_jitter=prior_jitter, lengthscale=torch.tensor([[0.2, 0.4], [0.3, 0.15]]),
                     outputscale=torch.tensor([1.2, 0.8]), noise_std=0.12, **options)
    with torch.no_grad():
        for j, q in enumerate(model.q):
            q.mean.copy_(torch.linspace(-0.2, 0.3 + j * 0.1, q.size, dtype=torch.float64))
            q.raw_cholesky.add_(0.1 * torch.eye(q.size, dtype=torch.float64))
    return model


def dense_reference(model, x, y):
    covariances, projections, residuals, q_covariances = [], [], [], []
    for j, z in enumerate(model.supports()):
        kernel = ard_rbf(z, z, model.hyp.lengthscales()[j], model.hyp.outputscales()[j])
        r = kernel + torch.diag(model.hyp.nuggets()[j] + model.prior_jitter)
        cross = ard_rbf(x, z, model.hyp.lengthscales()[j], model.hyp.outputscales()[j])
        a = torch.linalg.solve(r, cross.T).T
        residuals.append(model.hyp.outputscales()[j] - (cross * a).sum(-1))
        projections.append(a)
        covariances.append(r)
        q_covariances.append(model.q[j].cholesky() @ model.q[j].cholesky().T)
    residuals = torch.stack(residuals, dim=1)
    alpha = torch.softmax(-model.hyp.temperatures() * residuals.pow(model.hyp.exponent()), dim=1)
    residual = (alpha / residuals).sum(-1).reciprocal()
    weights = residual[:, None] * alpha / residuals
    design = torch.cat([weights[:, j, None] * a for j, a in enumerate(projections)], dim=1)
    prior = torch.block_diag(*covariances)
    q_covariance = torch.block_diag(*q_covariances)
    q_mean = torch.cat([q.mean for q in model.q])
    mean = design @ q_mean
    propagated = (design @ q_covariance * design).sum(-1)
    variance = residual + propagated
    r_y = torch.diag(residual + model.hyp.noise_std().square())
    q = torch.distributions.MultivariateNormal(q_mean, q_covariance)
    p = torch.distributions.MultivariateNormal(torch.zeros_like(q_mean), prior)
    kl = torch.distributions.kl_divergence(q, p)
    expected = torch.distributions.MultivariateNormal(mean, r_y).log_prob(y)
    expected -= 0.5 * torch.trace(torch.linalg.solve(r_y, design @ q_covariance @ design.T))
    exact = torch.distributions.MultivariateNormal(torch.zeros_like(y), r_y + design @ prior @ design.T)
    return mean, variance, alpha, weights, expected - kl, kl, exact.log_prob(y)


@pytest.mark.parametrize("jitter", [0.0, 1e-4])
def test_prospar_values_and_gradients_match_dense_generative_reference(jitter):
    model = example(prior_jitter=jitter)
    x = torch.tensor([[0.1, 0.25], [0.5, 0.5], [0.8, 0.7]], dtype=torch.float64)
    y = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    elbo, mean, variance, kl = model.elbo_batch(x, y, len(x))
    terms = model.conditional_terms(x)
    reference = dense_reference(model, x, y)
    for actual, expected in zip((mean, variance, terms["alpha"], terms["weights"], elbo, kl),
                                reference[:6], strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
    assert elbo < reference[6]
    torch.testing.assert_close(terms["weights"].sum(-1), torch.ones(len(x), dtype=x.dtype))
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad(elbo, parameters)
    reference_gradients = torch.autograd.grad(reference[4], parameters)
    for actual, expected in zip(actual_gradients, reference_gradients, strict=True):
        assert bool(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected, atol=2e-8, rtol=2e-8)


def test_one_expert_recovers_prior_and_integrated_conditional_objective():
    z = torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64)
    model = ProSparGP([z], temperatures=2.0, exponent=1.0, inducing_nuggets=0.03,
                     prior_jitter=0.0, outputscale=1.3, noise_std=0.1)
    x, y = z.clone(), torch.sin(8 * z[:, 0])
    torch.testing.assert_close(model.kl(), torch.zeros((), dtype=z.dtype), atol=1e-10, rtol=0)
    mean, variance = model.latent(x)
    torch.testing.assert_close(mean, torch.zeros(len(x), dtype=z.dtype))
    torch.testing.assert_close(variance, torch.full_like(mean, 1.3))
    terms = model.conditional_terms(x)
    conditional = terms["residual"] + model.hyp.noise_std().square()
    expected = -0.5 * (torch.log(2 * math.pi * conditional)
                       + (y.square() + variance - terms["residual"]) / conditional).sum()
    torch.testing.assert_close(model.elbo_batch(x, y, len(x))[0], expected)


def test_minibatch_elbo_scaling_subtracts_kl_once():
    model = example()
    x = torch.tensor([[0.1, 0.2], [0.4, 0.6], [0.7, 0.8], [0.9, 0.4]], dtype=torch.float64)
    y = torch.cos(x[:, 0])
    full, _, _, kl = model.elbo_batch(x, y, len(x))
    batches = [model.elbo_batch(a, b, len(x))[0] for a, b in zip(x.split(2), y.split(2), strict=True)]
    torch.testing.assert_close(torch.stack(batches).mean(), full)
    assert kl > 0


@pytest.mark.parametrize("dimensions", [1, 5])
def test_general_dimensions_ragged_counts_and_noise_batching(dimensions):
    generator = torch.Generator().manual_seed(8)
    z = [0.1 + 0.8 * torch.rand(m, dimensions, dtype=torch.float64, generator=generator)
         for m in (2, 3, 5)]
    model = ProSparGP(z, temperatures=1.0, exponent=1.2, inducing_nuggets=0.04,
                     prior_jitter=0.0)
    x = torch.rand(9, dimensions, dtype=torch.float64, generator=generator)
    mean, latent = model.predict(x, observation_noise=False)
    batched_mean, observed = predict_batched(model, x, batch_size=2)
    torch.testing.assert_close(mean, batched_mean)
    torch.testing.assert_close(observed - latent, model.hyp.noise_std().square().expand_as(latent))
    assert model.inducing_counts == (2, 3, 5)


def test_fit_updates_all_parameter_groups_and_frozen_flags_survive():
    model = example()
    x = torch.rand(20, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
    y = torch.sin(7 * x[:, 0])
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    snapshots = []
    history = fit(model, x, y, epochs=2, batch_size=10,
                  optimizer=OptimizerConfig(warmup_fraction=0, alternate_patchwork_updates=False),
                  on_epoch=lambda epoch, current, record: snapshots.append(epoch))
    assert snapshots == [1, 2] and len(history.losses) == 2
    for name in ("support.0.raw", "q.0.mean", "hyperparameters.raw_temperatures",
                 "hyperparameters.raw_exponent", "hyperparameters.raw_nuggets.0"):
        assert not torch.equal(dict(model.named_parameters())[name], initial[name]), name
    frozen = example(train_supports=False, train_hyperparameters=False)
    fit(frozen, x, y, epochs=1, batch_size=10)
    assert not any(p.requires_grad for p in frozen.hyp.parameters())
    assert not list(frozen.support.parameters())


@pytest.mark.parametrize("field,value", [("temperatures", 0), ("exponent", -1),
                                        ("inducing_nuggets", float("nan")),
                                        ("prior_jitter", -1)])
def test_invalid_explicit_parameters_raise(field, value):
    options = {"temperatures": 1.0, "exponent": 1.0, "inducing_nuggets": 0.02, "prior_jitter": 0.0}
    options[field] = value
    with pytest.raises(ValueError):
        ProSparGP([torch.tensor([[0.2], [0.8]], dtype=torch.float64)], **options)


def test_required_parameters_and_numerical_errors_have_no_defaults_or_repairs():
    z = [torch.tensor([[0.2], [0.8]], dtype=torch.float64)]
    with pytest.raises(TypeError, match="prior_jitter"):
        ProSparGP(z, temperatures=1.0, exponent=1.0, inducing_nuggets=0.02)
    model = ProSparGP(z, temperatures=1000.0, exponent=1.0, inducing_nuggets=0.02, prior_jitter=0.0)
    assert bool(torch.isfinite(model.hyp.temperatures()).all())
    bad_prior = [{"covariance": torch.eye(2, dtype=torch.float64),
                  "cholesky": 1e-5 * torch.eye(2, dtype=torch.float64)}]
    with pytest.raises(FloatingPointError, match="conditional residuals"):
        model.latent(z[0], bad_prior)
    with torch.no_grad():
        model.hyp.raw_exponent.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="exponent"):
        model.predict(z[0])


def test_stable_inverse_softplus_preserves_large_positive_initial_values():
    values = torch.tensor([1e-8, 0.1, 10.0, 1000.0], dtype=torch.float64)
    torch.testing.assert_close(torch.nn.functional.softplus(inverse_softplus(values)), values)
