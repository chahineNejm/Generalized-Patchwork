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
    prospar = ProSparConfig(temperatures=0.1, exponent=2.0,
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
    prospar = ProSparConfig(temperatures=0.1, exponent=2.0,
                           inducing_nuggets=0.01, prior_jitter=0.0)
    runs = [("SVGP", replace(common, model="svgp")),
            ("GPW", replace(common, model="gpw", patches=8)),
            ("GPW 4 x 256", replace(common, model="gpw", patches=4)),
            ("ProSpar 8 x 128", replace(common, model="prospar", patches=8, prospar=prospar)),
            ("ProSpar 4 x 256", replace(common, model="prospar", patches=4, prospar=prospar)),
            ("VNNGP K32", replace(common, model="vnngp", vnngp_neighbors=32)),
            ("VNNGP K128", replace(common, model="vnngp", vnngp_neighbors=128)),
            ("Standard Patchwork", replace(common, model="patchwork", exact_training_points="gpw_initial",
                exact_partition="grid", exact_grid_shape=(4, 2))),
            ("Standard Patchwork (GPW regions)", replace(common, model="patchwork",
                exact_training_points="gpw_initial", exact_partition="gpw_voronoi")),
            ("Standard Patchwork (PC tree)", replace(common, model="patchwork",
                exact_training_points="gpw_initial", exact_partition="pc_tree", exact_tree_patches=8))]
    return runs


def multidataset_defaults(*, device, input_root):
    """Approved dataset protocols and model capacities; no training is launched."""
    from pathlib import Path

    from experiments.multidataset_data import DatasetSpec

    paths = Path(input_root)
    data, runs = {}, {}
    oisst, oisst_runs = oisst_defaults(device=device)
    data["oisst"] = DatasetSpec("oisst", "oisst", 2, oisst.train_fraction, oisst.validation_fraction,
                                 oisst.seed, day=oisst.day, bounds=oisst.bounds)
    runs["oisst"] = [*oisst_runs, ("VNNGP K128", replace(dict(oisst_runs)["VNNGP"], vnngp_neighbors=128))]
    data["ukhousing"] = DatasetSpec("ukhousing", "ukhousing", 2, .64, .16, 31)
    runs["ukhousing"] = ukhousing_comparison_defaults(device=device)
    data["modis"] = DatasetSpec("modis", "modis", 2, None, 0., 31)
    heaton = heaton_defaults(device=device)
    common = replace(ukhousing_defaults(device=device), inducing_points=heaton.inducing_points,
        batch_size=heaton.batch_size, prediction_batch_size=heaton.prediction_batch_size)
    runs["modis"] = [("SVGP", replace(common, model="svgp")),
                      ("GPW", replace(common, model="gpw", patches=8)),
                      ("GPW 4 x 120", replace(common, model="gpw", patches=4))]
    specifications = [("california", 8, 512, 512), ("tco", 2, 1024, 1024),
        ("tco_l2", 2, 1024, 1024), ("icethick", 2, 1024, 1024),
        ("protein", 9, 1024, 1024), ("sarcos", 21, 1024, 1024), ("flight", 8, 2048, 2048)]
    for name, dimension, budget, batch in specifications:
        path = str(paths / ("airline.npz" if name == "flight" else "icethk.txt")) if name in ("flight", "icethick") else None
        data[name] = DatasetSpec(name, name, dimension, .8, 0., 31, input_path=path)
        runs[name] = matched_runs(device=device, budget=budget, batch_size=batch,
            primary_patches=16 if name == "flight" else 8,
            alternative_patches=8 if name == "flight" else 4, dimension=dimension)
    for function in ("michalewicz", "g_function"):
        for dimension in (3, 5, 10):
            for count in (50000, 100000):
                name = f"{function}_d{dimension}_n{count}"
                data[name] = DatasetSpec(name, function, dimension, .8, 0., 31, synthetic={
                    "training_points": count, "test_points": count // 4, "noise_variance": 1e-8,
                    "michalewicz_m": 10, "g_coefficients": [i / 2 for i in range(dimension)]})
                runs[name] = matched_runs(device=device, budget=1024, batch_size=1024,
                    primary_patches=8, alternative_patches=4, dimension=dimension)
    deferred = {"precipitation": "User deferred: no verified source file/URL supplied",
                "satellite_drag": "User deferred: no verified source file/URL supplied"}
    return data, runs, deferred


def multidataset_notebook_defaults(*, device, input_root):
    """Approved reduced batch; retain every existing GPW/ProSpar size."""
    _, runs, _ = multidataset_defaults(device=device, input_root=input_root)
    datasets = ["oisst", "ukhousing", "modis", "california", "tco", "tco_l2", "icethick", "protein", "sarcos",
                "michalewicz_d5_n50000", "g_function_d5_n50000"]
    models = {name: [label for label, config in runs[name] if config.model != "vnngp" and
                    (config.model != "patchwork" or config.exact_partition == "pc_tree")] for name in datasets}
    return {"datasets": datasets, "models": models, "epochs": 100, "warmup_epochs": 20,
            "early_stopping": {"patience": 10, "relative_min_improvement": 0.001}}


def benchmark_repeat_defaults(*, device, input_root):
    """Approved second/third model seeds, preserving seed31 data and capacities."""
    comparison = multidataset_notebook_defaults(device=device, input_root=input_root)
    datasets = ["california", "protein", "tco", "oisst"]
    overrides = {name: {label: {"prospar": {"temperatures": 1.0, "exponent": 1.0,
        "inducing_nuggets": 0.01, "prior_jitter": 0.0}}
        for label in comparison["models"][name] if label.startswith("ProSpar")} for name in datasets}
    return {**comparison, "datasets": datasets, "models": {name: comparison["models"][name] for name in datasets},
            "model_overrides": overrides, "seeds": [32, 33], "progress_every_epochs": 20}


def gpw_ablation_defaults(*, device, input_root):
    """Approved four-dataset ablation selection; retain existing data protocols."""
    comparison = multidataset_notebook_defaults(device=device, input_root=input_root)
    return {"datasets": ["california", "protein", "tco", "oisst"], "inducing_points": 1024, "patches": 8,
            "seeds": [31, 32, 33], "ranks": [0, 32],
            "stitch_modes": ["none", "fixed", "learned"],
            "progress_every_epochs": 20,
            "epochs": comparison["epochs"], "warmup_epochs": comparison["warmup_epochs"],
            "early_stopping": comparison["early_stopping"]}


def gpw_flight_ablation_defaults(*, device, input_root):
    """Nine rank64 stitching runs on FLIGHT800k, TCO.L2 and ICETHICK."""
    defaults = gpw_ablation_defaults(device=device, input_root=input_root)
    return {**defaults, "datasets": ["flight", "tco_l2", "icethick"], "seeds": [31], "ranks": [64],
            "data_overrides": {"flight": {"subset_size": 800000}}}


def gpw_shared_kernel_defaults(*, device, input_root):
    """Twelve shared-kernel runs paired to local rank32/learned ablations."""
    defaults = gpw_ablation_defaults(device=device, input_root=input_root)
    return {**{key:value for key,value in defaults.items() if key not in ("ranks", "stitch_modes")},
            "seeds": [31, 32, 33], "q_rank": 32}


def matched_runs(*, device, budget, batch_size, primary_patches, alternative_patches, dimension):
    """Fixed-budget variants; higher-dimensional Standard Patchwork uses its PC tree."""
    common = replace(ukhousing_defaults(device=device), inducing_points=budget, batch_size=batch_size,
                     patches=primary_patches)
    prospar = ProSparConfig(temperatures=.1, exponent=2., inducing_nuggets=.01, prior_jitter=0.)
    runs = [("SVGP", replace(common, model="svgp")), ("GPW", replace(common, model="gpw")),
        ("GPW (alternative)", replace(common, model="gpw", patches=alternative_patches)),
        ("ProSpar", replace(common, model="prospar", prospar=prospar)),
        ("ProSpar (alternative)", replace(common, model="prospar", patches=alternative_patches, prospar=prospar)),
        ("VNNGP K32", replace(common, model="vnngp", vnngp_neighbors=32)),
        ("VNNGP K128", replace(common, model="vnngp", vnngp_neighbors=128))]
    if dimension == 2:
        runs.extend([
            ("Standard Patchwork", replace(common, model="patchwork", exact_training_points="gpw_initial",
                exact_partition="grid", exact_grid_shape=(4, 2))),
            ("Standard Patchwork (GPW regions)", replace(common, model="patchwork",
                exact_training_points="gpw_initial", exact_partition="gpw_voronoi"))])
    runs.append(("Standard Patchwork (PC tree)", replace(common, model="patchwork",
        exact_training_points="gpw_initial", exact_partition="pc_tree", exact_tree_patches=8)))
    return runs
