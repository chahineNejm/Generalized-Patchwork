import torch

from pwgp.models.svgp import SVGP


def test_svgp_initializes_at_prior_and_has_finite_gradients() -> None:
    torch.manual_seed(3)
    supports = 0.1 + 0.8 * torch.rand(7, 2, dtype=torch.float64)
    model = SVGP(supports)
    x = torch.rand(11, 2, dtype=torch.float64)
    y = torch.randn(11, dtype=torch.float64)

    torch.testing.assert_close(
        model.kl(), torch.zeros((), dtype=torch.float64), atol=1e-8, rtol=0
    )
    elbo, mean, variance, kl = model.elbo_batch(x, y, len(x))
    assert mean.shape == variance.shape == (len(x),)
    assert torch.isfinite(elbo)
    assert torch.isfinite(kl)
    assert bool((variance > 0).all())

    (-elbo).backward()
    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())


def test_svgp_prediction_can_include_observation_noise() -> None:
    supports = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)[:, None]
    model = SVGP(supports)
    x = torch.tensor([[0.25], [0.75]], dtype=torch.float64)
    _, latent = model.predict(x, observation_noise=False)
    _, observed = model.predict(x, observation_noise=True)
    expected = model.hyp.noise_std().square().expand_as(latent)
    torch.testing.assert_close(observed - latent, expected)

