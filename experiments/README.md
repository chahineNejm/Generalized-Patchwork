# Experiments

This directory contains runnable experiments and supports the Colab entry
points under `notebooks/`.

Experiment code must accept hyperparameters at the call site, use exact declared
values, use common splits for comparisons, save full configuration and data
provenance, show training progress, and write outputs under ignored `results/`.

See `TODO.md` for planned runners and visual studies.

## Defaults, OISST And Analysis

`default_hyperparameters.py` supplies approved experiment defaults. All notebooks
allow explicit overrides through `dataclasses.replace`; OISST exposes per-model
`OVERRIDES` and `DATA_OVERRIDES`. Effective settings are saved, not just defaults.
OISST's `MODELS` list chooses which models run and their order, for example
`MODELS=["GPW", "SVGP"]`. Selection validates names and duplicates without
silently dropping requests; overrides for known deselected models may remain.

`notebooks/OISST_Colab.ipynb` runs SVGP, GPW, ProSpar, VNNGP and standard
Patchwork on the approved OISST ROI/split. It imports `oisst_benchmark.py`, which
calls existing model builders/trainers and the dataset manager. Defaults include
the approved ProSpar prior_jitter=0. No training/data/model code is embedded.

`exploratory_analysis.py` is separate from training: `explore_dataset(split,
directory=...)` saves summaries/maps/histograms; `analyze_run(directory)`
regenerates training/metric/point figures from saved outputs. Each model has a
`visuals/` folder and optional `inducing_positions.npz`; disable position saving
with `DATA_OVERRIDES={"save_inducing_positions": False}`. This does not freeze
positions or alter training. Standard Patchwork has no inducing variables.

All four main Colab launchers load latest main from the configured GitHub repo,
using sparse code-only checkout and recording the resolved commit/source archive.
Source changes take effect after pushing; no notebook rebuild is needed for a
model or default change. Rebuild a launcher only after editing its cells:

```powershell
python -m experiments.build_oisst_colab
python -m experiments.build_heaton_colab
python -m experiments.build_ukhousing_colab
python -m experiments.build_stitch_line_colab
```

Heaton and UKHousing also accept `model="prospar"`, with a required
`pwgp.config.ProSparConfig` specifying initial `temperatures`, `exponent`,
`inducing_nuggets` and explicit `prior_jitter`. No values are inferred and the
existing Colab run lists do not automatically add a ProSpar experiment.
Heaton's command line accepts the four fields as a `--prospar-config` JSON object.
See the root README for the approved generative equations and numerical policy.

`minimal_visual_smoke.py` is the smallest current verification. It fits SVGP
and Patchwork to one synthetic curve and plots loss, predictions, uncertainty,
initial/learned supports, and the learned stitch location.

## Two-phase sine

Run the fixed-boundary diagnostic with one low-frequency phase before zero and
one high-frequency phase after zero:

```powershell
python -m experiments.two_phase_sine
```

The default truth is `sin(2*pi*x)` for `x < 0` and `sin(10*pi*x)` for
`x >= 0`. The stitch and supports are fixed so the diagnostic isolates fitted
variational and kernel uncertainty. Saved metrics compare the production latent
variance `w^T S w + alpha^T R alpha` with the same fit evaluated without the
residual `R` contribution; the latter is an ablation, not another trained model.

## Heaton MODIS

`notebooks/GPW_Heaton_Colab.ipynb` imports source from GitHub. Upload that notebook
to Colab, select a GPU runtime, and Run all. The repository URL is configured
and no extra source upload is needed. It runs GPW 8 x 60,
SVGP 480, then GPW 4 x 120, each for 100 epochs with 20 warm-up epochs.

Rebuild the launcher after changing its cells:

```powershell
python -m experiments.build_heaton_colab
```

Install the `benchmark` extra to read the authors' RData files. The runner pins
the source revision and checksum, preserves the official split, and saves each
run in a new directory under `results/heaton/`.

```powershell
python -m experiments.heaton_benchmark --model patchwork --inducing-points 480 --patches 8 --epochs 100 --warmup-epochs 20 --device cuda:0
python -m experiments.heaton_benchmark --model svgp --inducing-points 480 --epochs 100 --warmup-epochs 20 --device cuda:0
python -m experiments.heaton_benchmark --model patchwork --inducing-points 480 --patches 4 --epochs 100 --warmup-epochs 20 --device cuda:0
```

All remaining settings are explicit `HeatonConfig` fields and command-line
arguments. `config.json` also records the existing numerical settings and
parameter bounds. `status.json`, `history.json`, and `progress.png` update
during training. Final outputs include metrics, predictions, maps, geometry
history, learned hyperparameters, and model state. Prediction intervals include
observation noise; the maps also show latent standard deviation separately.

