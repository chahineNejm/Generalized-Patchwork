"""Build a GitHub-backed UKHousing Colab launcher."""

import json
from pathlib import Path

from experiments.build_heaton_colab import cell
from experiments.colab_source import github_bootstrap, source_metadata


def build_notebook(root: Path) -> Path:
    unpack = github_bootstrap("/content/gpw_ukhousing")
    cells = [
        cell("markdown", """
        # UKHousing: SVGP, GPW, standard Patchwork and VNNGP

        Upload to Colab, select a GPU runtime and Run all. Restart the runtime
        first if an older copy has already loaded model classes.
        Code loads from the latest main of chahineNejm/Generalized-Patchwork.
        The resolved commit and source archive are saved with the results.

        This reconstructs the 2018 flats task from HM Land Registry transactions
        and FreeMapTools postcode centroids using the HIP-GP authors' region and
        filters. The verified reconstruction contains 185,254 transactions.
        It is NOT the verified original VNNGP dataset snapshot. Do not compare
        these numbers directly with the VNNGP paper's table.
        Downloads are checksum-pinned: changed files raise errors, not substitutions.

        All four models use the same seeded 64/16/20 train/validation/test split,
        Gaussian observation likelihood, ARD RBF kernel family and float64.
        Only training targets determine log-price standardization. Coordinates
        use the authors' fixed geographic ROI. Repeated-coordinate transactions
        are retained, so random-split performance is not spatial extrapolation.
        VNNGP's paper instead uses Matern-5/2: this is a controlled RBF comparison.

        Default runs: SVGP 1,024 inducing points; GPW 8 x 128; standard Patchwork
        conditions directly on all training targets in fixed rectangular regions
        with at most 512 observations each; VNNGP has one inducing point per
        training transaction and 32 preceding neighbors. Budgets are displayed,
        not represented as equal. Standard Patchwork uses more regions to make
        exact local conditioning feasible; this is not a same-partition ablation.

        Each model trains for 100 epochs. Variational models freeze kernel/noise
        parameters for 20 epochs; exact Patchwork has no variational warm-up and
        performs 100 full conditional-likelihood updates. No early stopping or
        test-based selection. All runs have tqdm and live diagnostics every 10
        epochs, then metrics and maps. Final metrics use standardized log prices;
        scores in unstandardized log GBP are saved separately. No GBP-scale claims.
        Training timings include live diagnostics/checkpoints; prediction timings
        cover validation and test inference, excluding plots and scoring.

        GPW weights use C = A S A^T + R, and latent variance includes alpha^T R alpha.
        Its weights depend on S: the expected-likelihood-minus-KL objective is
        posterior-adaptive, without a fixed-model ELBO claim.
        VNNGP uses GPyTorch 1.15.2 with exact CPU FAISS neighbor searches and GPU
        GP calculations. CPU search is an explicit design choice, not a fallback.
        Prediction uses the same mean-field conditional equations with two
        triangular solves, checked against GPyTorch's original predictions.
        The stochastic KL uses the same SPD solver too, replacing the upstream
        batched torch.linalg.solve without changing the KL formula or jitter.
        Numerical failures raise; no retries, jitter escalation, CPU substitution,
        variance clipping or automatic configuration changes are implemented here.
        """),
        cell("markdown", "## 1. GitHub source and pinned VNNGP dependencies"),
        cell("code", unpack),
        cell("code", '''import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--editable", "/content/gpw_ukhousing[ukhousing]"], check=True)'''),
        cell("markdown", "## 2. Settings and GPU checks"),
        cell("code", """
        import gc, json, shutil
        from dataclasses import asdict, replace
        from datetime import UTC, datetime
        import faiss
        import gpytorch
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import Image, Markdown, display
        from experiments.ukhousing_benchmark import UKConfig, archive_run, prepare_split, run_ukhousing
        from pwgp.data.ukhousing import prepare_ukhousing
        from pwgp.linalg import cholesky_solve
        from pwgp.models.vnngp import VNNGP, strict_gpytorch

        if not torch.cuda.is_available():
            raise RuntimeError("Select Runtime > Change runtime type > GPU, restart, then Run all")
        DEVICE = "cuda:0"
        torch.set_num_threads(2)
        faiss.omp_set_num_threads(2)
        print({"GPU": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "CUDA": torch.version.cuda, "gpytorch": gpytorch.__version__,
               "neighbor_search": "exact CPU FAISS", "dtype": "float64"})
        for batch, size, columns in [(8, 128, 128), (512, 32, 1), (1, 1024, 4)]:
            matrix = (torch.eye(size, dtype=torch.float64, device=DEVICE) + 0.01).expand(batch, size, size).clone()
            matrix.requires_grad_(True)
            rhs = torch.ones(batch, size, columns, dtype=torch.float64, device=DEVICE)
            answer = cholesky_solve(rhs, torch.linalg.cholesky(matrix))
            torch.testing.assert_close(matrix @ answer, rhs)
            answer.square().mean().backward()
            assert torch.isfinite(matrix.grad).all()
        del matrix, rhs, answer
        GPU_KL_PROBES = [(4, 8), (32, 512)]
        for neighbors, batch_size in GPU_KL_PROBES:
            generator = torch.Generator(device=DEVICE).manual_seed(31)
            probe_x = 0.1 + 0.8 * torch.rand(neighbors + batch_size, 2, device=DEVICE,
                                             dtype=torch.float64, generator=generator)
            probe = VNNGP(probe_x, neighbors=neighbors, batch_size=batch_size)
            for step in range(2):
                with strict_gpytorch():
                    probe.train()
                    probe.zero_grad()
                    prediction = probe(None)
                    probe_indices = probe.variational_strategy.current_training_indices.to(DEVICE)
                    loss = -gpytorch.mlls.VariationalELBO(probe.likelihood, probe, num_data=len(probe_x))(
                        prediction, torch.sin(probe_x[probe_indices, 0]))
                    loss.backward()
                assert len(probe_indices) == (neighbors if step == 0 else batch_size)
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
            probe_mean, probe_variance = probe.predict(probe_x[:7])
            assert torch.isfinite(probe_mean).all() and (probe_variance > 0).all()
            torch.cuda.synchronize()
            del probe, prediction, loss, probe_mean, probe_variance
        gc.collect()
        torch.cuda.empty_cache()
        print("VNNGP first-K and stochastic KL batches, backward and prediction passed")

        from experiments.ukhousing_benchmark import build_model
        probe_rng = np.random.default_rng(31)
        probe_coordinates = probe_rng.uniform(0.05, 0.95, (80, 2))
        probe_split = prepare_split(probe_coordinates * [7.5, 5.5] + [-5.7, 50],
                                    np.sin(8 * probe_coordinates[:, 0]))
        for name in ["gpw", "svgp", "patchwork"]:
            probe_config = UKConfig(model=name, device=DEVICE, inducing_points=12,
                                    patches=3, q_rank=4, stitches_per_edge=2,
                                    exact_maximum_points=16)
            probe = build_model(probe_config, probe_split)
            probe_x = probe_split["x"][probe_split["train"][:8]].to(DEVICE)
            probe_y = probe_split["y"][probe_split["train"][:8]].to(DEVICE)
            loss = probe.loss() if name == "patchwork" else -probe.elbo_batch(probe_x, probe_y, len(probe_x))[0]
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
            probe_mean, probe_variance = probe.predict(probe_x)
            assert torch.isfinite(probe_mean).all() and (probe_variance > 0).all()
            del probe, loss, probe_mean, probe_variance
        gc.collect()
        torch.cuda.empty_cache()
        print("All four model GPU checks passed")

        from experiments.default_hyperparameters import ukhousing_defaults
        COMMON = replace(ukhousing_defaults(device=DEVICE))  # Add explicit overrides here.
        RUNS = [("SVGP 1024", replace(COMMON, model="svgp")),
                ("GPW 8 x 128", replace(COMMON, model="gpw")),
                ("Standard Patchwork", replace(COMMON, model="patchwork")),
                ("VNNGP K32", replace(COMMON, model="vnngp"))]
        RUN_ROOT = ROOT / "results/ukhousing" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        (RUN_ROOT / "source_snapshot.zip").write_bytes(SOURCE_ZIP)
        (RUN_ROOT / "source_provenance.json").write_text(json.dumps(SOURCE_PROVENANCE, indent=2))
        display(pd.DataFrame([{"run": name, **asdict(config)} for name, config in RUNS]))
        """),
        cell("markdown", "## 3. Download, validate and prepare one shared dataset"),
        cell("code", """
        raw_x, raw_y, manifest = prepare_ukhousing(ROOT / "data/ukhousing")
        split = prepare_split(raw_x, raw_y, seed=COMMON.seed)
        np.savez_compressed(RUN_ROOT / "split.npz", train=split["train"],
                            validation=split["validation"], test=split["test"])
        preprocessing = {key: split[key] for key in
                         ["target_mean", "target_std", "seed", "input_transform", "target_transform"]}
        (RUN_ROOT / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2))
        (RUN_ROOT / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
        display(manifest)
        print({name: len(split[name]) for name in ["train", "validation", "test"]})
        """),
        cell("markdown", "## 4. Train and report each model"),
        cell("code", """
        from google.colab import files

        summaries, completed = [], []
        for name, config in RUNS:
            display(Markdown(f"### {name}"))
            live = display(Markdown("Initializing model..."), display_id=True)

            def show_progress(path, epoch, handle=live):
                handle.update(Image(filename=str(path / "progress.png")))

            result = run_ukhousing(config, split,
                                   directory=RUN_ROOT / name.lower().replace(" ", "_"),
                                   on_snapshot=show_progress)
            metrics = json.loads((result / "metrics.json").read_text())
            summaries.append({"model": name, **metrics})
            completed.append(result)
            display(pd.DataFrame([summaries[-1]]).set_index("model").T)
            display(Image(filename=str(result / "maps.png")))
            comparison = pd.DataFrame(summaries).set_index("model")
            comparison.to_csv(RUN_ROOT / "comparison.csv")
            run_archive = archive_run(result)
            gc.collect()
            torch.cuda.empty_cache()
            print("Saved run:", run_archive)
            print("CUDA allocated/reserved MiB:", torch.cuda.memory_allocated() / 2**20,
                  torch.cuda.memory_reserved() / 2**20)
            files.download(str(run_archive))
        display(Markdown("## Final comparison: standardized log-price metrics"))
        display(comparison)
        """),
        cell("markdown", "## 5. Download all results"),
        cell("code", """
        from google.colab import files

        archive = shutil.make_archive(str(RUN_ROOT), "zip", root_dir=RUN_ROOT)
        print("Results archive:", archive)
        files.download(archive)
        """),
        cell("markdown", """
        ## Sources and attribution
        - [UK flats preparation reference](https://github.com/cunningham-lab/hipgp/blob/master/experiments-hip-gp/uk_housing_data.py)
        - [HM Land Registry Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-yearly-file)
        - [Postcode coordinate source](https://www.freemaptools.com/download-uk-postcode-lat-lng.htm)
        - [Patchwork Kriging paper](https://jmlr.org/papers/v19/17-042.html)
        - [VNNGP paper](https://proceedings.mlr.press/v162/wu22h.html)
        - [GPyTorch VNNGP implementation](https://docs.gpytorch.ai/en/v1.15.2/examples/04_Variational_and_Approximate_GPs/VNNGP.html)

        Contains HM Land Registry data (C) Crown copyright and database right 2021.
        Licensed under the Open Government Licence v3.0.
        Contains Ordnance Survey data (C) Crown copyright and database right 2021.
        Contains Royal Mail data (C) Royal Mail copyright and database right 2021.
        Source: Office for National Statistics licensed under OGL v3.0; via FreeMapTools.
        """),
    ]
    notebook = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"name": "UKHousing_Colab.ipynb", "provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"}, "gpw_source": source_metadata()},
        "nbformat": 4, "nbformat_minor": 5}
    target = root / "notebooks/UKHousing_Colab.ipynb"
    target.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return target


if __name__ == "__main__":
    print(build_notebook(Path(__file__).resolve().parents[1]))
