import pytest
import torch

from pwgp import SVGP
from pwgp.parameters import UnitCubeParameter


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_initial_and_learned_parameters_raise(value):
    supports = torch.full((3, 2), 0.5, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        SVGP(supports, noise_std=value)
    with pytest.raises(ValueError, match="finite"):
        UnitCubeParameter(torch.tensor([[value]], dtype=torch.float64))
    points = UnitCubeParameter(supports)
    with torch.no_grad():
        points.raw.fill_(value)
    with pytest.raises(FloatingPointError, match="finite"):
        points()
    model = SVGP(supports)
    with torch.no_grad():
        model.hyp.raw_noise.fill_(value)
    with pytest.raises(FloatingPointError, match="finite"):
        model.predict(torch.tensor([[0.4, 0.4]], dtype=torch.float64))


def test_svgp_nonfinite_prediction_raises():
    model = SVGP(torch.tensor([[0.2], [0.5], [0.8]], dtype=torch.float64))
    with torch.no_grad():
        model.q.mean.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="finite"):
        model.predict(torch.tensor([[0.4]], dtype=torch.float64))


@pytest.mark.parametrize("parameter", ["lengthscale", "outputscale", "noise_std"])
def test_vnngp_nonfinite_initial_parameters_raise(parameter):
    from pwgp.models.vnngp import VNNGP

    with pytest.raises(ValueError, match="finite"):
        VNNGP(torch.rand(8, 2, dtype=torch.float64), **{parameter: float("nan")})


def test_vnngp_nonfinite_mean_raises():
    from pwgp.models.vnngp import VNNGP

    model = VNNGP(torch.rand(8, 2, dtype=torch.float64), neighbors=2, batch_size=4)
    model(None)
    with torch.no_grad():
        model.variational_strategy._variational_distribution.variational_mean.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="finite"):
        model(None)


@pytest.mark.parametrize("noise", [0.0, -0.1])
def test_initial_observation_noise_must_be_positive(noise):
    from pwgp.models.vnngp import VNNGP

    x = torch.rand(8, 2, dtype=torch.float64)
    with pytest.raises(ValueError):
        SVGP(x, noise_std=noise)
    with pytest.raises(ValueError, match="strictly positive"):
        VNNGP(x, neighbors=2, batch_size=4, noise_std=noise)
