"""Local approved exponential-GP toy using unchanged production GPW training."""

import argparse
import json
import subprocess
import time
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from experiments.default_hyperparameters import gpw_exponential_toy_defaults, ukhousing_defaults
from experiments.exploratory_analysis import (
    plot_toy_comparison,
    plot_toy_field,
    plot_toy_points,
    plot_toy_predictions,
    toy_prediction_fields,
)
from experiments.gpw_ablation import bind_initialization
from experiments.multidataset_benchmark import prepare_split, run_model, write_json
from pwgp.data.download import file_checksum
from pwgp.data.gp_toy import sample_exponential_gp
from pwgp.run_records import record_failures


def snapshot_source(root, destination):
    paths = sorted([*root.glob("src/**/*.py"), *root.glob("experiments/**/*.py"), root / "pyproject.toml"])
    with zipfile.ZipFile(destination / "source_snapshot.zip", "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            bundle.write(path, path.relative_to(root))
    record = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "mode": "actual working-tree source snapshot; may include uncommitted runner",
        "source_sha256": file_checksum(destination / "source_snapshot.zip"),
        "files": {path.relative_to(root).as_posix(): file_checksum(path) for path in paths}}
    write_json(destination / "source_provenance.json", record)
    return record


def run_toy(*, directory, settings):
    """Run six approved variants; failures stop all subsequent runs."""
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    with record_failures(output, status_key="state"):
        write_json(output / "settings.json", settings)
        snapshot_source(Path(__file__).resolve().parents[1], output)
        dataset, latent = sample_exponential_gp(points=settings["points"], extent=settings["extent"],
            variance=settings["variance"], lengthscale=settings["lengthscale"],
            noise_variance=settings["noise_variance"], seed=settings["data_seed"], threads=settings["threads"])
        split = prepare_split(dataset, train_fraction=settings["train_fraction"], seed=settings["data_seed"])
        np.savez_compressed(output / "data.npz", inputs=dataset.x, latent=latent, observations=dataset.y,
            train=split["train"], test=split["test"])
        write_json(output / "data_manifest.json", {"data_sha256": file_checksum(output / "data.npz"),
            "generator": dataset.metadata, "preprocessing": split["preprocessing"],
            "counts": {key: len(split[key]) for key in ("train", "test")},
            "fitted_kernel": "local ARD RBF; intentionally different from exponential generator"})
        visuals = output / "visuals"
        visuals.mkdir()
        plot_toy_field(dataset.x, latent, destination=visuals / "latent_field.png", extent=settings["extent"])
        rows, geometry = [], {}
        base = replace(ukhousing_defaults(device="cpu"), q_rank=settings["q_rank"],
            epochs=settings["epochs"], warmup_epochs=settings["warmup_epochs"], threads=settings["threads"])
        for setup in settings["configurations"]:
            for seed in settings["seeds"]:
                config = replace(base, seed=seed, inducing_points=setup["inducing_points"], patches=setup["patches"],
                                 stitches_per_edge=setup["stitches_per_edge"])
                reference = bind_initialization(split, config)
                identity = split["gpw_initialization_sha256"]
                geometry[f"{setup['name']}/seed{seed}"] = identity
                completed = []
                for mode in tqdm(settings["stitch_modes"], desc=f"compare:toy/{setup['name']}/seed{seed}", unit="model"):
                    variant = replace(config, stitch_mode=mode)
                    label = f"toy/{setup['name']}/seed{seed}/{mode}/rank{config.q_rank}"
                    print("START", label, flush=True)
                    result = run_model(variant, split, directory=output / setup["name"] / f"seed{seed}_{mode}",
                        save_inducing_positions=True, early_stopping=settings["early_stopping"],
                        progress_description="train:" + label, progress_every_epochs=settings["progress_every_epochs"])
                    predictions = toy_prediction_fields(result, split["target_std"])
                    plot_toy_predictions(predictions, latent[predictions["indices"]],
                        destination=result / "visuals/field_predictions.png", extent=settings["extent"], title=label)
                    plot_toy_points(result, split["preprocessing"], destination=result / "visuals/point_layouts.png",
                                    extent=settings["extent"], title=label)
                    metrics = json.loads((result / "metrics.json").read_text())
                    if json.loads((result / "initialization_identity.json").read_text())["geometry_sha256"] != identity:
                        raise ValueError("Toy stitch modes do not share the declared initial geometry")
                    row = {"configuration": setup["name"], "seed": seed, "stitch_mode": mode,
                        "rank": config.q_rank, "inducing_points": config.inducing_points, "patches": config.patches,
                        "active_stitches": 0 if mode == "none" else int(np.prod(reference["candidate_stitches"].shape[:-1])),
                        "directory": str(result), **metrics}
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
                    write_json(result / "toy_data_reference.json", {"data_sha256": file_checksum(output / "data.npz"),
                        "geometry_sha256": identity, "configuration": setup, "data_seed": settings["data_seed"]})
                    target = output / f"{setup['name']}_seed{seed}_{mode}.zip"
                    with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
                        shared = [output / p for p in ("settings.json", "data_manifest.json", "source_snapshot.zip", "source_provenance.json")]
                        for path in shared + sorted(p for p in result.rglob("*") if p.is_file()):
                            bundle.write(path, path.relative_to(output))
                    print("COMPLETE", label, json.dumps({k: metrics[k] for k in
                        ("rmse_original_target", "nlpd_original_target", "training_seconds")}), "archive:", target, flush=True)
                    completed.append(result)
                combined = visuals / f"{setup['name']}_seed{seed}"
                combined.mkdir()
                plot_toy_comparison(completed, split, destination=combined,
                                    extent=settings["extent"], title=f"{setup['name']}, seed{seed}, rank{config.q_rank}")
        write_json(output / "geometry_references.json", geometry)
        write_json(output / "status.json", {"state": "complete", "completed_runs": len(rows),
                                            "wall_seconds": time.perf_counter() - started})
    print("RESULTS", output, flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    target = args.directory if args.directory is not None else Path("results") / datetime.now(UTC).strftime("exponential_toy_%Y%m%dT%H%M%S%fZ")
    run_toy(directory=target, settings=gpw_exponential_toy_defaults())
