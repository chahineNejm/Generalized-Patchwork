import torch

from experiments.two_phase_sine import (
    TwoPhaseSineConfig,
    build_model,
    make_training_data,
    prediction_decomposition,
    two_phase_truth,
)
from pwgp import fit
from pwgp.config import OptimizerConfig


def test_two_phase_truth_is_continuous_at_fixed_boundary() -> None:
    epsilon = torch.tensor(1e-10, dtype=torch.float64)
    boundary = two_phase_truth(torch.tensor([0.0], dtype=torch.float64))
    left_limit = two_phase_truth(-epsilon[None])
    right_limit = two_phase_truth(epsilon[None])
    torch.testing.assert_close(boundary, torch.zeros_like(boundary), rtol=0, atol=0)
    torch.testing.assert_close(left_limit, boundary, rtol=0, atol=1e-8)
    torch.testing.assert_close(right_limit, boundary, rtol=0, atol=1e-8)


def test_fixed_stitch_and_variance_decomposition_after_training() -> None:
    config = TwoPhaseSineConfig(
        epochs=1,
        warmup_epochs=0,
        observations_per_phase=6,
        query_points=21,
        supports_per_patch=3,
        q_rank=1,
        torch_threads=1,
    )
    config.validate()
    torch.manual_seed(config.seed)
    training_x, _, _, training_y = make_training_data(config)
    model = build_model(config)
    initial_stitch = model.stitches().detach().clone()
    fit(
        model,
        training_x,
        training_y,
        epochs=config.epochs,
        batch_size=len(training_x),
        seed=config.seed,
        optimizer=OptimizerConfig(
            learning_rate_q=config.learning_rate_q,
            learning_rate_hyperparameters=config.learning_rate_hyperparameters,
            warmup_fraction=0.0,
        ),
    )
    torch.testing.assert_close(model.stitches(), initial_stitch, rtol=0, atol=0)
    torch.testing.assert_close(
        model.stitches(), torch.tensor([[[0.5]]], dtype=torch.float64), rtol=0, atol=0
    )
    query = torch.linspace(0.05, 0.95, config.query_points, dtype=torch.float64)[:, None]
    decomposition = prediction_decomposition(model, query)
    torch.testing.assert_close(
        decomposition["latent_variance"],
        decomposition["inducing_variance"] + decomposition["residual_variance"],
    )
    assert bool((decomposition["residual_variance"] >= 0).all())
