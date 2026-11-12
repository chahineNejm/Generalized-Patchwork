# Generalized Patchwork Gaussian Processes

This repository contains the reusable algorithmic foundation for the research
project described in *Structured Gaussian Processes for Scalable Regression*.
It deliberately separates model code from experiment code.

Coding agents must read [AGENTS.md](AGENTS.md), [TODO.md](TODO.md), and
[docs/EVOLUTION.md](docs/EVOLUTION.md) before making changes. They record the
repository rules, planned work, and technical history.

## OISST Comparison

Open `notebooks/OISST_Colab.ipynb` in Colab, select a GPU runtime and Run all.
The notebook loads the latest `main` from
https://github.com/chahineNejm/Generalized-Patchwork and records its resolved
commit. Model/data implementations are imported, not embedded. Code-only sparse
checkout excludes unrelated dataset files. The local airline pickle is ignored.
Changes must be pushed to GitHub before Colab can use them.
To load a source update, rerun setup, dependencies and settings. No kernel
restart, reconnect or runtime deletion is required. Setup verifies the existing
checkout and fast-forwards to latest main; local source edits, a wrong repository
or divergent history raise instead of being overwritten. Data/results stay.
Every package-dependent stage runs in a fresh Python worker, with live progress
and visuals in the notebook. Workers verify the setup commit and source archive.
Live progress and training-history figures update in place without accumulating
one plot per snapshot; saved figures and snapshot frequency are unchanged.
Metrics appear as numeric tables, without single-run metric bar-chart displays.
To reuse earlier prepared data after setup, choose DATA_ACTION="load" and its
exact PREPARED_PATH. Ordinary model-setting changes need only settings/training.
Worker requests/logs/failures are saved under results/worker_jobs. Errors block
dependent stages; there is no automatic skipping, retry or checkpoint resumption.

Approved defaults are in `experiments/default_hyperparameters.py`:
OISST 2025-01-15, ROI (-80,-30,20,50), shared 64/16/20 split, seed 31;
SVGP 512, GPW 8x64, ProSpar 8x64, VNNGP K32 with all training locations,
and two Standard Patchwork baselines using GPW's exact 512 initial observed
locations/targets: `Standard Patchwork` uses a 4x2 checkerboard (longitude x
latitude); `Standard Patchwork (GPW regions)` uses Voronoi cells of GPW's initial
clustering centers. Each has 5 fixed stitches/shared boundary segment. Empty
cells raise; no region is dropped and no point is substituted. GPW still learns
from all training observations: these are reduced-data, not equal-information,
comparisons. Historical all-data Patchwork results are a different protocol.
ProSpar starts with T=1, c=1, nuggets=0.01 and explicit prior_jitter=0.
Variational runs use 100 epochs/20 warm-up, batch 512, q LR 0.002 and other
learned-parameter LR 0.001. Exact Patchwork retains its existing LR 0.01,
100 full-data updates and no variational warm-up. Targets use training-only
standardization; input normalization uses the declared geographic ROI.

The notebook's `OVERRIDES` accepts per-model settings, for example:

```python
MODELS = ["GPW", "SVGP"]  # Only these models, in this order.
OVERRIDES = {"GPW": {"epochs": 150, "warmup_epochs": 30}}
DATA_OVERRIDES = {"save_inducing_positions": True}
```

`MODELS` accepts SVGP, GPW, ProSpar, VNNGP, Standard Patchwork and
Standard Patchwork (GPW regions). Unknown,
duplicate or empty selections raise; deselected models are not initialized.
Overrides for known deselected models may remain in the notebook for later use.

Matched Patchwork controls can be set directly in the notebook:

```python
MODELS = ["GPW", "Standard Patchwork", "Standard Patchwork (GPW regions)"]
OVERRIDES = {
    "GPW": {"inducing_points": 512, "patches": 8},
    "Standard Patchwork": {"exact_grid_shape": (4, 2), "exact_stitches_per_edge": 5,
                           "epochs": 100, "exact_learning_rate": 0.01},
    "Standard Patchwork (GPW regions)": {"exact_stitches_per_edge": 5, "epochs": 100},
}
```

Reference point counts and clustering follow the effective GPW overrides even
if GPW is deselected for a baseline-only rerun. `exact_maximum_points` applies
only to the legacy adaptive partition, not fixed grids/Voronoi cells. Selected
rows and observed targets are in `training_points.npz`; boundaries/labels and
stitches are in `partition.npz`, with `visuals/partition.png`. GPW's matching
initial supports, centers and row indices are in `initialization.npz`.

No override changes the defaults file. To override ProSpar settings, provide a
complete dictionary of `ProSparConfig` fields through that model's `prospar`
field; the worker constructs the configuration. Actual settings,
source/data provenance, shared split, history and checkpoints are saved.
Each completed model is archived/downloaded and its GPU memory released before
the next. A failed run saves its traceback and stops the comparison.

`experiments/exploratory_analysis.py` contains data summaries and plotting,
separate from fitting. Comparison-level `exploratory/` contains target maps,
histograms and split maps. Each run's `visuals/` contains metrics, loss,
hyperparameter evolution, predictions/uncertainty, and initial/final inducing
positions when enabled. GPW/ProSpar also save weight maps. Raw positions and
snapshot epochs are in `inducing_positions.npz`; standard Patchwork has no
inducing positions. `analyze_run(path)` regenerates saved-run summary plots
without retraining. Training, prediction, analysis and model comparisons use tqdm.

### Partial Reruns

Run setup once. The OISST notebook has separate data, training and analysis cells:

- To change models/hyperparameters, rerun settings and **Train only the selected
  models**. Do not rerun data preparation when its settings are unchanged.
  Prepared data are verified, not reacquired or reprocessed. Every training
  execution creates a new batch; earlier results stay intact.
- To retry after an error, correct the input/cause and rerun the failed stage.
  Dependents remain blocked until it succeeds. For a failed model, select only
  that model in MODELS, rerun settings and training. This retrains from scratch;
  it does not resume a checkpoint or silently skip completed models.
- To regenerate scores/figures, rerun analysis with ANALYSIS_SOURCE="latest_batch"
  after a successful batch, or choose ANALYSIS_SOURCE="saved_runs" and set
  SAVED_RUNS to exact completed model-run directories. The saved_runs action
  needs setup only, not data preparation or training. All outputs go to a new
  analysis directory; original metrics/plots are not overwritten.

The data cell also supports DATA_ACTION="load" with an explicit PREPARED_PATH
from a previous preparation. Missing/corrupt files or changed data settings
raise rather than triggering preparation. To retain prepared data across runtime
deletion, download that folder explicitly in the download cell with
DOWNLOAD_SOURCE="saved_directory" and SAVED_DIRECTORY set to its printed path.
Restore the folder before choosing load in a later session. Runtime-local files
are not persistent storage; the notebook does not silently search for backups.

Heaton and UKHousing runs record failures and re-raise the original exception.
Colab notebooks stop downstream cells after a failed stage; restart the runtime
before retrying. Installation/test commands check exit codes. Requested patch
neighbor counts are validated, never capped; one patch requires zero neighbors.

The current foundation provides:

- a full-covariance stochastic variational Gaussian process (SVGP);
- ProSpar-GP with independent full-covariance sparse experts and local ARD RBF;
- standalone two-layer Nested Kriging with correlated exact submodels and shared
  RBF or Matern-5/2 kernels;
- a generalized Patchwork GP with local ARD kernels, trainable supports,
  trainable stitching locations, agreement weights conditional on inducing values, and a
  block-plus-low-rank variational precision;
- shared minibatch ELBO training and batched prediction;
- direct-observation Standard Patchwork trained with the paper's joint
  likelihood p(Y, Delta_S=0), not the stitch-conditioned likelihood;
- hierarchical support and patch-graph initialization;
- a dataset registry with cached downloads, provenance manifests, checksum
  validation, and common `X, y` loaders.

Experiment sweeps, result tables, and paper figures are intentionally out of
scope for this first layer.

## Installation

Python 3.11 or newer is required. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,dev]"
```

PyTorch installation can be replaced with the platform-specific command from
the PyTorch website when CUDA support is required.

## Model API

The variational models expose `elbo_batch`, `latent`, and `predict` with the same input
convention. Inputs and targets should be normalized before fitting.

### ProSpar-GP

`ProSparGP` uses the same `fit` and `predict_batched` APIs as SVGP. Supports can
be `[experts, points, dimensions]` or a sequence of `[points_j, dimensions]`
tensors with different expert sizes. Initialize from real training locations.
Temperatures, exponent and inducing nuggets are learned positive parameters;
their initial values and the extra fixed `prior_jitter` are required explicitly.
No experiment values are chosen here:

```python
from pwgp import ProSparGP

def make_prospar(supports, *, temperatures, exponent, inducing_nuggets,
                 prior_jitter, **svgp_options):
    return ProSparGP(
        supports, temperatures=temperatures, exponent=exponent,
        inducing_nuggets=inducing_nuggets, prior_jitter=prior_jitter,
        **svgp_options,
    )
```

`prior_jitter=0.0` means `R_j=K_j+diag(d_j)`; an explicitly positive value adds
that fixed diagonal term. There is no retry/escalation, and this argument, not
`NumericalConfig.prior_jitter`, controls the ProSpar prior. Kernel/noise bounds
and minimum variational Cholesky diagonal reuse SVGP's constraints.

This implementation follows the generative model (6)-(13) in the supplied
[ProSpar preprint](https://arxiv.org/abs/2311.08752v1), with the approved derived
ELBO/prediction rather than the inconsistent printed (14)/(16). The likelihood
integrates the conditional residual before taking its expectation over q(U).
Predictive latent variance is `lambda + sum_j b_j^2 A_j S_j A_j^T`; observation
noise is added once. No author source implementation has been verified.

Heaton and UKHousing runners accept `model="prospar"` with
`prospar=ProSparConfig(temperatures=..., exponent=..., inducing_nuggets=...,
prior_jitter=...)`, imported from `pwgp.config`. Runner budgets divide equally
across experts; the model class itself also supports unequal budgets. Existing
Colab run lists are unchanged. Records include explicit configuration, learned
parameters, checkpoints, metrics, visuals and model/paper provenance.

### SVGP Example

```python
import torch

from pwgp import SVGP, fit
from pwgp.config import OptimizerConfig
from pwgp.initialization import observed_supports

X = torch.rand(500, 2, dtype=torch.float64)
y = torch.sin(8 * X[:, 0]) + 0.1 * torch.randn(500, dtype=torch.float64)
Z = observed_supports(X, count=32, seed=7)

model = SVGP(
    Z,
    lengthscale=torch.tensor([0.2, 0.3], dtype=torch.float64),
    outputscale=1.0,
    noise_std=0.05,
    train_supports=True,
    train_hyperparameters=True,
)
optimizer = OptimizerConfig(
    learning_rate_q=2e-3,
    learning_rate_supports=2e-2,
    learning_rate_hyperparameters=1e-2,
    warmup_fraction=0.5,
)
history = fit(model, X, y, epochs=50, batch_size=128, optimizer=optimizer)
mean, variance = model.predict(X[:20])
```

For Patchwork, use `build_patchwork` to create the support groups, neighbor
graph, stitch locations, and local ARD initialization consistently:

```python
from pwgp.initialization import build_patchwork

model = build_patchwork(
    X,
    patches=4,
    supports_per_patch=32,
    stitches_per_edge=2,
    q_rank=8,
    seed=7,
)
history = fit(model, X, y, epochs=50, batch_size=128)
```

### Standalone Nested Kriging

`NestedKriging` implements the two-layer shared-prior construction in
[Rulliere et al., equations (3)-(4), (17)-(18)](https://arxiv.org/html/1607.05432v3).
Provide explicit disjoint group labels `0,...,P-1`; inputs can have any dimension.
It uses observed targets directly, with no inducing points, stitches or GPW weights.
Cross-expert covariances are retained and NK coefficients need not sum to one.

```python
from pwgp import NestedKriging, predict_batched

X = torch.linspace(0.01, 0.99, 30, dtype=torch.float64)[:, None]
y = torch.sin(8 * X[:, 0])
model = NestedKriging(
    X, y, groups=torch.arange(30) // 2,
    kernel="matern52", lengthscale=0.05, outputscale=1.0,
    noise_std=0.0, jitter=0.0, train_hyperparameters=False,
)
query = torch.linspace(0, 1, 101, dtype=torch.float64)[:, None]
mean, variance = predict_batched(model, query, batch_size=32, observation_noise=False)
```

For fitting, construct with `train_hyperparameters=True`, then call
`model.fit(epochs=100, learning_rate=0.01, loo_batch_size=10, seed=31)`.
This minimizes the paper's leave-one-out MSE with Adam/autograd, not its SPSA
optimizer, and then applies noiseless variance calibration. Each expert must
retain at least one observation after exclusion. Default calibration requires
`noise_std=jitter=0`; for an explicitly noisy model, set `calibrate_variance=False`
to fit lengthscales with fixed outputscale/noise instead. Outputscale calibration
outside the configured bounds raises rather than clipping. `on_epoch` callbacks
and tqdm report training progress; `model.configuration()` exposes model settings.

This is a correctness-first baseline with pointwise variance, not arbitrary
multi-layer aggregation or the authors' optimized R/C++ implementation. Prediction
uses covariance blocks without storing the full training covariance, but still
accounts for all cross-group covariances. LOO currently recomputes local factors;
large-data performance and CUDA are not verified. Existing Colab notebooks import
the class from GitHub but their default comparison lists are unchanged.

## Dataset manager

`pip install -e ".[heat]"` installs the pinned FEM solver for the corrected setup.
`python -m experiments.stitch_line_preview` previews fixed horizontal/vertical
periodic heat, disjoint uniform support regions and four initial stitch lines.
It saves arrays and a figure only; no training is performed.
`python -m experiments.stitch_line_heat` trains the same four initial layouts
with independently moving stitches, two ARD models and free support motion.
Lines initialize stitches only; no collinearity is imposed. Defaults: 28 supports/model, 500 observations, 350 epochs and
40 kernel warm-up epochs. Full configurations, motion, checkpoints and scores
are saved under `results/stitch_line_heat/`.

`notebooks/GPW_Periodic_Heat_Colab.ipynb` imports the experiment from latest main.
Select a GPU runtime and Run all; no separate source upload is needed.
Change `STITCH_LEARNING_RATES = [0.003]` to one or several values in the settings
cell. Every value repeats the four approved starts with matched seeds/data.
Live figures, per-layout downloads and memory cleanup are included; CPU FEM
generation and float64 CUDA GP training are explicit, without a CPU fallback.
Live figures also track every kernel's x/y ARD lengthscales and output variance,
plus shared noise SD. Each layout saves epoch-0/per-epoch transformed values in
`kernel_history.json` and plots in `kernel_evolution.png`; dashed markers denote
kernel warm-up. Output variance and noise SD are in standardized target units.

`python -m experiments.anisotropic_heat` runs the original two-metal heat experiment locally:
two orthogonal diffusivity tensors, two trainable ARD GPW support clouds, four
matched starting orientations, 350 epochs with 40 kernel warm-up epochs. Results
under `results/anisotropic_heat/` include initial/final points, fitted fields,
held-out scores, checkpoints and complete provenance. Override values through
`run(HeatConfig(...))`; the GP uses temperature observations, not a PDE loss.

`notebooks/UKHousing_Colab.ipynb` compares SVGP 1,024, GPW 8 x 128, direct-observation
standard Patchwork Kriging, and GPyTorch VNNGP (all training locations, K=32).
Select a Colab GPU and Run all; source loads from GitHub and data downloads automatically.
All use ARD RBF/Gaussian models and one seeded 64/16/20 split. The 185,254-row
2018-flats reconstruction follows the HIP-GP authors' documented filters using
checksum-pinned Land Registry and postcode sources, but is NOT the verified
VNNGP paper snapshot. Paper scores are not directly comparable. Target is natural
log GBP, standardized on training targets only; original log-GBP scores also save.
Exact Patchwork has fixed rectangular regions (max 512 training points/region),
conditions on every training target and boundary stitches, and uses conditional
Gaussian marginal likelihood, not inducing variables or a variational substitute.
Its finer partition differs from GPW's eight patches. VNNGP uses exact CPU FAISS
neighbor search explicitly and GPU GP computations, with no backend fallback.
VNNGP prediction and stochastic KL both use Cholesky plus two triangular solves
to avoid incompatible batched CUDA solve kernels. Its preflight tests the first-K
and stochastic-KL batches, including the default K=32 / batch=512 workload.
The notebook saves metrics, checkpoints, source/data provenance, train/validation/
test indices, progress visuals and final maps. After every completed run it frees
the model and CUDA cache, then archives and downloads that run with shared
provenance. The final combined archive is also available.
Local verification of these optional baselines requires `pip install -e ".[dev,ukhousing]"`.

`notebooks/GPW_Heaton_Colab.ipynb` is a GitHub-backed GPU Colab notebook:
upload it, select a GPU runtime, and Run all. It imports the current source and
runs GPW 8 x 60, SVGP 480, and GPW 4 x 120 with 100 epochs / 20 warm-up.
GitHub access is required; no separate source upload is needed.
The notebook registers the checked-out `src/` package in the active kernel, so
setup does not require a runtime restart after installation.
Cholesky solves use two explicit triangular solves. The notebook validates
representative batched solves and gradients on the selected GPU before training.
Restart the runtime when replacing an older notebook with updated model source.

GPW computes agreement weights from `R = Cov(local functions | U, Delta_S=0)`,
retaining stitching-induced cross-covariances. Weights depend on inducing/stitch
locations and kernels, not the variational distribution. Its latent
variance is `w^T S w + alpha^T R alpha`; observation noise is added once for
measurement predictions and NLPD. Both terms enter training, with gradients
through the model-dependent weights. Training uses the fixed-model Gaussian
ELBO. The explicit weight regularizer is unchanged. SVGP is unchanged.
Historical posterior-weighted runs, and older runs omitting residual predictive
variance, describe different implementations and are not recomputed automatically.

For a compact synthetic check of that decomposition, run
`python -m experiments.two_phase_sine`. It fits two sine frequencies separated
at a fixed stitch `x = 0`, saves regional fit and calibration metrics, and plots
the inducing-state and residual `R` variance contributions separately.

The local Heaton MODIS comparison is available through
`python -m experiments.heaton_benchmark --help` after installing the
`benchmark` and `visual` extras. See `experiments/README.md` for the official-split
GPW/SVGP calls and saved visual diagnostics.

The default cache is `data/`, which is ignored by Git.

```powershell
pwgp-data list
pwgp-data download oisst --date 2025-01-15
pwgp-data download flight
pwgp-data prepare flight
pwgp-data download tco-l2
pwgp-data import icethick C:\path\to\icethk.txt
pwgp-data inspect icethick
```

OISST is downloaded from NOAA NCEI. FLIGHT now uses Hensman's author-prepared
`airline.pickle`, as referenced by the VFF airline SVGP experiment, rather than
reconstructing it from Data Expo files. Its published Dropbox link was disabled
when checked on 2026-11-11. Download fails explicitly; no raw-data fallback exists.
A trusted local copy can be used with `pwgp-data import flight C:\path\to\airline.pickle`
followed by `pwgp-data prepare flight`. Only load trusted pickles: the format can
execute code. Preparation keeps all usable rows, applies the author's explicit
HHMM-to-minutes conversion, and records features/removal counts; no filling,
subsampling, split or target scaling occurs. The original 2013 split is not verified.
Nonfinite model parameters and predictions raise errors rather than being replaced.
TCO.L2 uses the data bundle published by the Patchwork Kriging authors.
The ICETHICK archive is hosted behind an interactive CAPTCHA, so it is imported
locally and verified against the official MD5 checksum.

Dataset files are never committed. Each successful acquisition writes a JSON
manifest recording source URLs, checksums, timestamps, and local paths.
Sources and schemas are explicit. Acquisition and loading do not use mirrors,
parser guessing, automatic recovery, or silent parameter substitution.

## Verification

```powershell
python -m pytest
```

The tests compare the structured Patchwork prior, variational covariance, KL,
and fusion algebra against small dense reference calculations. They also check
SVGP gradients and dataset checksum behavior.

For the smallest end-to-end visual check, open
`notebooks/colab_minimal_smoke.ipynb` in Colab. Its cells install the project,
run the algebra tests, and fit both models to a synthetic curve. The final
figure shows training loss, predictive uncertainty, support movement, and the
learned Patchwork stitch. Set the notebook's single `REPOSITORY_URL` value
after this repository is published on GitHub.

## References

- Hensman, Fusi, and Lawrence (2013), *Gaussian Processes for Big Data*.
- Park and Apley (2018), *Patchwork Kriging for Large-scale Gaussian Process Regression*.
- NOAA 1/4-degree Daily Optimum Interpolation Sea Surface Temperature v2.1.

