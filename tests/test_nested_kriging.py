import math

import pytest
import torch
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

from pwgp import NestedKriging, predict_batched


@pytest.fixture(scope="module", autouse=True)
def small_matrix_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def example(*, kernel="matern52", noise_std=0.12, jitter=1e-6):
    generator = torch.Generator().manual_seed(32)
    x = 0.05 + 0.9 * torch.rand(12, 3, dtype=torch.float64, generator=generator)
    y = torch.sin(7 * x[:, 0]) + 0.2 * torch.cos(4 * x[:, 1])
    groups = torch.arange(12) // 4
    return NestedKriging(x, y, groups, kernel=kernel, lengthscale=[0.3, 0.4, 0.5],
                         noise_std=noise_std, jitter=jitter)


def dense_reference(model, query, *, exclude=None):
    mask = torch.arange(len(model.train_x)) != exclude if exclude is not None else torch.ones(
        len(model.train_x), dtype=torch.bool,
    )
    x, y, labels = model.train_x[mask], model.train_y[mask], model.groups[mask]
    ky = model.kernel(x, x) + (model.noise_std.square() + model.jitter) * torch.eye(
        len(x), dtype=x.dtype,
    )
    cross = model.kernel(x, query)
    projection = x.new_zeros(len(query), model.patches, len(x))
    for group in range(model.patches):
        indices = (labels == group).nonzero().flatten()
        projection[:, group, indices] = torch.linalg.solve(ky[indices][:, indices], cross[indices]).T
    covariance = projection @ ky @ projection.transpose(-2, -1)
    k_m = (projection @ cross.T[..., None]).squeeze(-1)
    beta = torch.linalg.solve(covariance, k_m[..., None]).squeeze(-1)
    mean = (beta * (projection @ y)).sum(-1)
    variance = model.outputscale() - (beta * k_m).sum(-1)
    return mean, variance, covariance, k_m, beta


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
def test_nested_kriging_matches_dense_projection_values_and_gradients(kernel):
    model = example(kernel=kernel)
    query = model.train_x[:5] + 0.015
    actual = model.components(query)
    expected = dense_reference(model, query)
    for key, value in zip(
        ["mean", "variance", "covariance", "cross_covariance", "weights"], expected, strict=True,
    ):
        torch.testing.assert_close(actual[key], value)
    parameters = tuple(model.parameters())
    loss = actual["mean"].square().sum() + actual["variance"].sum()
    expected_loss = expected[0].square().sum() + expected[1].sum()
    expected_gradients = torch.autograd.grad(expected_loss, parameters)
    gradients = torch.autograd.grad(loss, parameters)
    for actual_gradient, expected_gradient in zip(gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient)
        assert torch.isfinite(actual_gradient).all()


@pytest.mark.parametrize("kernel", ["rbf", "matern52"])
def test_kernel_matches_sklearn_ard_reference(kernel):
    model = example(kernel=kernel)
    reference = RBF([0.3, 0.4, 0.5]) if kernel == "rbf" else Matern([0.3, 0.4, 0.5], nu=2.5)
    reference = ConstantKernel(1.0) * reference
    x = model.train_x
    expected = torch.from_numpy(reference(x.numpy(), x.numpy()))
    torch.testing.assert_close(model.kernel(x, x), expected)


@pytest.mark.parametrize("singleton_experts", [False, True])
def test_one_expert_or_singleton_experts_recover_full_gp(singleton_experts):
    x = torch.linspace(0.1, 0.9, 5, dtype=torch.float64)[:, None]
    groups = torch.arange(len(x)) if singleton_experts else torch.zeros(len(x), dtype=torch.int64)
    model = NestedKriging(x, torch.sin(8 * x[:, 0]), groups, noise_std=0.1)
    query = torch.tensor([[0.23], [0.57], [1.1]], dtype=x.dtype)
    ky = model.kernel(x, x) + 0.01 * torch.eye(len(x), dtype=x.dtype)
    cross = model.kernel(x, query)
    expected_mean = cross.T @ torch.linalg.solve(ky, model.train_y)
    expected_variance = model.outputscale() - (cross * torch.linalg.solve(ky, cross)).sum(0)
    mean, variance = model.predict(query, observation_noise=False)
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(variance, expected_variance)


def test_noise_added_once_and_prediction_batching_matches():
    model = example()
    query = model.train_x[:5] + 0.03
    mean, variance = model.predict(query, observation_noise=False)
    observed_mean, observed_variance = predict_batched(model, query, batch_size=2)
    torch.testing.assert_close(observed_mean, mean)
    torch.testing.assert_close(observed_variance - variance, model.noise_std.square().expand_as(variance))


def test_far_field_recovers_prior_without_unit_sum_constraint():
    model = example(kernel="rbf")
    query = torch.full((1, 3), 2.0, dtype=model.train_x.dtype)
    parts = model.components(query)
    assert (parts["weights"].sum() - 1).abs() > 0.01
    assert parts["mean"].abs().max() < 0.01
    torch.testing.assert_close(parts["variance"], model.outputscale().expand(1), atol=1e-6, rtol=0)


def test_leave_one_out_excludes_target_and_matches_dense_reference():
    model = example()
    indices = torch.tensor([0, 5, 10])
    means, variances = model.leave_one_out(indices)
    for row, index in enumerate(indices):
        expected = dense_reference(model, model.train_x[index:index + 1], exclude=int(index))
        torch.testing.assert_close(means[row], expected[0].squeeze())
        torch.testing.assert_close(variances[row], expected[1].squeeze())
    with torch.no_grad():
        model.train_y[0] += 100
    changed, _ = model.leave_one_out(indices[:1])
    torch.testing.assert_close(changed, means[:1])


def test_paper_outputscale_calibration_matches_equation_18():
    model = example(noise_std=0, jitter=0)
    indices = torch.arange(len(model.train_x))
    mean, variance = model.leave_one_out(indices)
    expected = model.outputscale() * ((model.train_y - mean).square() / variance).mean()
    actual = model.calibrate_outputscale()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model.outputscale(), expected)
    new_mean, new_variance = model.leave_one_out(indices)
    torch.testing.assert_close(((model.train_y - new_mean).square() / new_variance).mean(),
                               model.train_x.new_tensor(1.0))


def test_fit_has_progress_callbacks_and_finite_updates(capsys):
    model = example(noise_std=0, jitter=0)
    initial = model.lengthscales().detach().clone()
    callbacks = []
    history = model.fit(epochs=2, learning_rate=0.01, loo_batch_size=3, seed=7,
                        on_epoch=lambda epoch, current, history: callbacks.append(epoch))
    assert callbacks == [1, 2]
    assert len(history.losses) == 2 and all(math.isfinite(loss) for loss in history.losses)
    assert not torch.equal(initial, model.lengthscales())
    assert "loss=" in capsys.readouterr().err
    assert model.configuration()["group_sizes"] == [4, 4, 4]


def test_singular_aggregation_raises_without_retry_or_general_solve(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Production NK must not use a general solve or pseudoinverse")

    monkeypatch.setattr(torch.linalg, "solve", forbidden)
    monkeypatch.setattr(torch.linalg, "pinv", forbidden)
    model = example()
    model.predict(model.train_x[:2] + 0.01)
    x = torch.tensor([[0.3], [0.7]], dtype=torch.float64)
    singular = NestedKriging(x, x[:, 0], torch.tensor([0, 1]), kernel="rbf")
    with pytest.raises(torch.linalg.LinAlgError):
        singular.predict(torch.tensor([[1000.0]], dtype=x.dtype))


def test_invalid_settings_and_empty_loo_expert_raise():
    model = example()
    x, y, groups = model.train_x, model.train_y, model.groups
    with pytest.raises(ValueError, match="contiguous"):
        NestedKriging(x, y, groups + 1)
    with pytest.raises(ValueError, match="strictly inside"):
        NestedKriging(x, y, groups, lengthscale=0)
    with pytest.raises(ValueError, match="kernel must"):
        NestedKriging(x, y, groups, kernel="other")
    with pytest.raises(ValueError, match="nonnegative"):
        NestedKriging(x, y, groups, jitter=-0.1)
    with pytest.raises(ValueError, match="noise_std=0"):
        model.fit(epochs=1, learning_rate=0.01, loo_batch_size=2)
    with pytest.raises(ValueError, match="training rows"):
        model.leave_one_out(torch.tensor([len(x)]))
    singletons = NestedKriging(x, y, torch.arange(len(x)))
    with pytest.raises(ValueError, match="remain nonempty"):
        singletons.leave_one_out(torch.tensor([0]))
    with pytest.raises(ValueError, match="at least two"):
        singletons.fit(epochs=1, learning_rate=0.01, loo_batch_size=1)


def test_noiseless_paper_configuration_predicts_without_hyperparameter_fitting():
    generator = torch.Generator().manual_seed(21)
    x = torch.rand(30, 1, dtype=torch.float64, generator=generator).sort(dim=0).values
    model = NestedKriging(x, torch.sin(8 * x[:, 0]), torch.arange(30) // 2,
                         lengthscale=0.05, outputscale=1, noise_std=0, jitter=0,
                         train_hyperparameters=False)
    query = torch.linspace(0, 1, 101, dtype=x.dtype)[:, None]
    mean, variance = predict_batched(model, query, batch_size=17, observation_noise=False)
    assert mean.shape == variance.shape == (101,)
    assert torch.isfinite(mean).all() and (variance >= 0).all()
