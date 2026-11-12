"""Approved experiment defaults; notebooks may override frozen configurations."""

from dataclasses import replace

from pwgp.config import ProSparConfig


def oisst_defaults(*, device="cuda:0"):
    from experiments.oisst_benchmark import OISSTConfig
    from experiments.ukhousing_benchmark import UKConfig

    common = UKConfig(
        epochs=100, warmup_epochs=20, inducing_points=512, patches=8,
        q_rank=32, patch_neighbors=2, stitches_per_edge=8,
        vnngp_neighbors=32, vnngp_batch_size=512,
        exact_maximum_points=256, exact_stitches_per_edge=5,
        batch_size=512, prediction_batch_size=256,
        lengthscale=0.1, outputscale=1.0, noise_std=0.1,
        learning_rate_q=0.002, learning_rate_structure=0.001,
        learning_rate_hyperparameters=0.001, exact_learning_rate=0.01,
        weight_epsilon=0.01, seed=31, threads=2, snapshot_every=10, device=device,
    )
    prospar = ProSparConfig(temperatures=1.0, exponent=1.0,
                           inducing_nuggets=0.01, prior_jitter=0.0)
    runs = [("SVGP", replace(common, model="svgp")),
            ("GPW", replace(common, model="gpw")),
            ("ProSpar", replace(common, model="prospar", prospar=prospar)),
            ("VNNGP", replace(common, model="vnngp")),
            ("Standard Patchwork", replace(common, model="patchwork", exact_training_points="gpw_initial",
                exact_partition="grid", exact_grid_shape=(4, 2))),
            ("Standard Patchwork (GPW regions)", replace(common, model="patchwork", exact_training_points="gpw_initial",
                exact_partition="gpw_voronoi")),
            ("Standard Patchwork (PC tree)", replace(common, model="patchwork", exact_training_points="gpw_initial",
                exact_partition="pc_tree", exact_tree_patches=8))]
    return OISSTConfig(day="2025-01-15", bounds=(-80.0, -30.0, 20.0, 50.0),
                       seed=31, train_fraction=0.64, validation_fraction=0.16,
                       save_inducing_positions=True), runs


def heaton_defaults(*, device):
    from experiments.heaton_benchmark import HeatonConfig

    return HeatonConfig(
        inducing_points=480, epochs=100, warmup_epochs=20,
        batch_size=2048, prediction_batch_size=2048,
        patches=8, stitches_per_edge=8, neighbors=2, q_rank=32,
        lengthscale=0.1, outputscale=1.0, noise_std=0.1,
        learning_rate_q=0.002, learning_rate_supports=0.001,
        learning_rate_stitches=0.001, learning_rate_hyperparameters=0.001,
        weight_epsilon=0.01, seed=31, threads=2, snapshot_every=10, device=device,
    )


def ukhousing_defaults(*, device):
    from experiments.ukhousing_benchmark import UKConfig

    return UKConfig(
        epochs=100, warmup_epochs=20, inducing_points=1024, patches=8,
        stitches_per_edge=8, q_rank=32, patch_neighbors=2,
        vnngp_neighbors=32, vnngp_batch_size=512,
        exact_maximum_points=512, exact_stitches_per_edge=2,
        batch_size=1024, prediction_batch_size=256,
        lengthscale=0.1, outputscale=1.0, noise_std=0.1,
        learning_rate_q=0.002, learning_rate_structure=0.001,
        learning_rate_hyperparameters=0.001, exact_learning_rate=0.01,
        weight_epsilon=0.01, seed=31, snapshot_every=10, threads=2, device=device,
    )


def periodic_heat_defaults(*, device):
    from experiments.periodic_heat_preview import PreviewConfig
    from experiments.stitch_line_heat import RunConfig
    from experiments.stitch_line_preview import LayoutConfig

    heat = PreviewConfig(
        cells=80, diffusivity=(0.6, 0.025), angle_pairs=((0, 90),),
        periods=(4, 4), baseline=1.0, amplitude=0.8, time=0.015,
        time_steps=40, supports_per_patch=28, support_margin=0.02,
        stitches=10, q_rank=12, lengthscale=0.12, outputscale=1.0,
        noise_std=0.03, weight_epsilon=0.01, seed=31,
    )
    layouts = LayoutConfig(
        heat=heat, line_centers=((0.5, 0.5), (0.56, 0.5), (0.5, 0.5), (0.5, 0.5)),
        line_angles=(90, 90, 45, 0),
        line_names=("Vertical", "Vertical, shifted", "Diagonal", "Horizontal"),
        line_margin=0.05, candidate_points=10000,
    )
    return RunConfig(
        layouts=layouts, observations=500, epochs=350, warmup_epochs=40,
        learning_rate_q=0.008, learning_rate_supports=0.004,
        learning_rate_line=0.003, learning_rate_hyperparameters=0.003,
        snapshot_every=25, prediction_batch_size=256, threads=2, device=device,
    )


def ukhousing_comparison_defaults(*, device):
    """Approved 1,024-point comparison; retain the legacy four-model preset."""
    common = ukhousing_defaults(device=device)
    prospar = ProSparConfig(temperatures=1.0, exponent=1.0,
                           inducing_nuggets=0.01, prior_jitter=0.0)
    runs = [("SVGP", replace(common, model="svgp")),
            ("GPW", replace(common, model="gpw", patches=8)),
            ("GPW 4 x 256", replace(common, model="gpw", patches=4)),
            ("ProSpar 8 x 128", replace(common, model="prospar", patches=8, prospar=prospar)),
            ("ProSpar 4 x 256", replace(common, model="prospar", patches=4, prospar=prospar)),
            ("VNNGP K32", replace(common, model="vnngp", vnngp_neighbors=32)),
            ("VNNGP K256", replace(common, model="vnngp", vnngp_neighbors=256)),
            ("Standard Patchwork", replace(common, model="patchwork", exact_training_points="gpw_initial",
                exact_partition="grid", exact_grid_shape=(4, 2))),
            ("Standard Patchwork (GPW regions)", replace(common, model="patchwork",
                exact_training_points="gpw_initial", exact_partition="gpw_voronoi")),
            ("Standard Patchwork (PC tree)", replace(common, model="patchwork",
                exact_training_points="gpw_initial", exact_partition="pc_tree", exact_tree_patches=8))]
    return runs
