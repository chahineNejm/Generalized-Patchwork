# Agent Guide

Read this file, `TODO.md`, and `docs/EVOLUTION.md` before changing code.

## Purpose

Build a reproducible research codebase for scalable Gaussian-process regression.
The core models are full-covariance SVGP and generalized Patchwork GP with local
ARD kernels, stitching constraints, Nested-Kriging fusion, and structured
variational precision. The current phase is the algorithm and data foundation.

## Non-Negotiable Rules

- All methodological choices belong to the user. Do not infer approval from
  silence, a paper's defaults, library defaults, or a recommendation. Before
  implementing any unspecified choice affecting the model, objective, kernels,
  uncertainty, initialization, data cleaning/preprocessing, splits, optimization,
  hyperparameters, numerical regularization, or evaluation protocol, ask the user
  and wait for explicit approval. Preserve already approved choices; record new
  decisions in docs/EVOLUTION.md. Unresolved choices must remain explicit and
  must not receive an invented default. Routine implementation details may be
  chosen only when they do not change the method or experimental behavior.
- Fail loudly. Do not add fallbacks, silent substitutions, parser guessing,
  model downgrades, pseudoinverses, alternate URLs, or implicit value changes.
- The user explicitly prefers an error to any fallback. A failure must never
  trigger a different model, data source, solver, device, parameter value, or
  experimental configuration. Do not disguise recovery as robustness.
- Do not use `try/except` to continue after an error. Let the original error
  surface. Validation errors should name the failed invariant.
- Never switch to a base model or change requested values to make a run succeed.
- Runner exceptions must save a full traceback and re-raise; never install a
  process-global exception hook. Colab stages must block downstream execution
  after failure, and subprocess commands must check exit status.
- OISST supports explicit partial reruns: failed stages may be retried by rerunning
  that stage, but their dependents remain invalid until success. Independent saved-
  result analysis is an explicit action, never a fallback for failed training.
  Prepared-data reuse must verify config identity and checksums without rebuilding.
  Model reruns train from scratch in new directories; never auto-skip, resume or
  overwrite earlier runs. Reanalysis reads saved predictions and writes new outputs
  without training, prediction, checkpoint loading or modifying original artifacts.
- Validate out-of-range values. Do not clamp or project requested inputs silently.
- Nonfinite model parameters/predictions must raise. Unusable dataset rows are
  removed, never filled, with counts recorded. FLIGHT defaults to the trusted
  author-prepared airline data, not raw reconstruction; an unavailable source
  raises without substitution. Do not claim a verified original 2013 SVGP split.
  The multi-dataset Kaggle adapter uses only the approved one-time converted
  airline.npz plus pinned conversion provenance, never a pickle compatibility
  fallback. Preserve the original pickle and the approved feature/time mapping.
- Keep code compact. Use brief docstrings for functions and comments only for
  mathematically dense blocks.
- Put tunable values in explicit arguments or configuration dataclasses.
- Every training loop displays `tqdm` progress with the current loss.
- Dataset files, checkpoints, and generated results stay outside Git.
- Preserve complete configuration and provenance with every saved result.
- Colab launchers load the latest main from the user-supplied GitHub repository;
  record the resolved commit and source archive for each comparison. Never embed
  model or dataset implementations in notebooks. Approved experiment defaults
  live in experiments/default_hyperparameters.py; notebook overrides are explicit
  and saved as the effective configuration, never silently merged with replacements.
- Keep exploratory analysis in experiments/exploratory_analysis.py. OISST runs
  save plots under visuals/ and optionally record actual inducing positions;
  standard Patchwork has no inducing variables and must not be labeled as such.
- OISST setup reuses only a verified clean checkout and fast-forwards to latest
  main, preserving data/results. Run package-dependent stages in fresh Python
  workers; never reload models into a stale notebook kernel or require reconnects.
  Check source commit/archive identity and worker exit status; stream progress
  and snapshot events, save full failures, and terminate workers on interruption.

## Mathematical Invariants

- Trainable support and stitch points receive inputs normalized to `[0,1]^D`.
- Dataset-based inducing initialization samples distinct actual training locations
  without replacement, globally or within clustered patches; never snap centers.
  Open-unit-cube parameters require interior observations; do not move boundary
  rows inward. Insufficient eligible locations raise, never reduce the budget.
- Initial learned observation-noise SD must be positive; reject invalid user
  values before squaring. Preserve explicit noiseless Nested-Kriging references.
- GPW/SVGP kernels are ARD RBF. SVGP has one global ARD vector. Patchwork has one ARD
  vector and output scale per patch, plus one global observation noise.
- SVGP uses `q(U) = N(m, LL^T)` and includes conditional residual variance.
- ProSpar uses independent full-covariance q(U_j), local ARD RBF kernels, learned
  positive inducing nuggets, temperatures and exponent. Their initial values and
  the extra fixed prior_jitter are required arguments, with no defaults. The
  explicit prior_jitter overrides NumericalConfig.prior_jitter for this model.
  Use the approved generative derivation from arXiv:2311.08752v1 (6)-(13), not
  literal printed (14)/(16): integrate f before the expected log likelihood;
  prediction propagates q variance with squared b_j, not squared lambda_j.
  This is an independent paper-derived implementation. Core mean/variance/ELBO
  algebra was checked against the official author supplement after converting
  whitened coordinates; optimization and parameter constraints are not identical.
  Approved experiment starts are T_j=0.1 for every expert and c=2. Temperatures
  remain independently learned, not tied. Nuggets0.01/prior_jitter0 and all
  other approved settings remain unchanged; this is not full author replication.
- Patchwork uses `q(U) = N(m, (D + VV^T)^-1)` with block-diagonal `D`.
- Stitching conditions exactly on `Delta_S = 0` using one coherent regularized
  prior. Do not add independent jitter to the Schur complement.
- Agreement weights use R = Cov(local functions | U, Delta_S=0), independent
  of q(U), and sum to one. Preserve the explicit weight_epsilon regularization;
  singular regularized systems raise an error.
- Patchwork latent variance is w^T S w + alpha^T R alpha, with the full stitched
  local conditional residual covariance R. Observation noise is added only once.
- Weights depend on inducing/stitch locations and local kernels, not q(U).
  Training uses the Gaussian ELBO with gradients through these model parameters
  and the variational predictive moments. Preserve both variance terms.
- Algebra changes require small dense-reference tests.
- Use `pwgp.linalg.cholesky_solve` for two explicit triangular solves; do not
  restore the native batched POTRS call that fails on some Colab CUDA builds.
- Standard Patchwork baseline conditions directly on observed noisy targets in
  fixed linear-boundary regions and boundary stitch differences, without variational
  inducing variables. Train with the paper's joint likelihood p(Y, Delta_S=0),
  not p(Y | Delta_S=0); values and gradients must match a dense joint reference.
- OISST Standard Patchwork variants use exactly GPW's initial selected rows and
  their observed targets, not all observations or GPW's optimized locations.
  Provide a fixed (longitude, latitude) grid, default (4,2), or clipped Voronoi
  cells of the initial clustering centers, or an 8-region balanced PC tree.
  PC-tree splits use the first local PC and equal observation counts, with
  seeded uniform stitches on shared leaf boundaries. Require power-of-two
  region counts and divisible budgets; unsplittable projections raise. Save
  fixed tree routing with the geometry. Preserve every requested region;
  empty cells raise, never drop/merge cells or substitute points. Resolve the
  reference from GPW's effective initialization settings, including when GPW is
  deselected. Save row indices, centers, geometry and fixed training-point plots.
  GPW still trains on all observations: label these as reduced-data baselines,
  not equal-information benchmarks. Legacy UKHousing adaptive/all-data settings
  remain explicit and unchanged.
- UKHousing is a verified 2018-flats reconstruction, not a verified VNNGP snapshot.
  Preserve this distinction, pinned raw checksums and training-only target scaling.
- The separate UKHousing size-matched comparison uses RBF, 100 epochs / 20
  variational kernel-noise warm-up; GPW/ProSpar 8x128 and 4x256, SVGP 1024,
  VNNGP K32/K128, and the three reduced-data Standard Patchwork partitions.
  Those baselines reference the named eight-patch GPW initialization explicitly,
  even with both GPW sizes selected. Use fresh workers and verified prepared
  splits without reconnects; preserve the legacy four-model UKHousing preset.
- The multi-dataset Kaggle launcher is gated off until RUN_TRAINING=True, prints
  the complete finite plan, and defaults to 100 epochs maximum / 20 warm-up. It uses fresh
  workers, explicit selection/overrides, immutable prepared-data reuse, per-run
  archives and a collection archive under /kaggle/working/gpw_results.
  The reduced notebook excludes VNNGP and FLIGHT, retains nine real datasets
  and two synthetic cases (each function D5/train50k), keeps every existing
  GPW/ProSpar size, and selects PC-tree-only Standard Patchwork. Keep all
  full catalog presets available and reject conflicting selections. Archive and save
  summary metrics after each completed model, before starting the next, with an
  immediate download link. Session-local files are not an external backup.
  Approved early stopping monitors epoch training loss only after warm-up:
  patience10, minimum relative improvement0.001*abs(significant_best_loss).
  Equal losses never reset patience, even at zero. Epoch21 sets the baseline
  under20 warm-up. Evaluate the last trained model, never restore a checkpoint.
  Record control settings, actual epochs and stopping reason; force final plots
  and checkpoints on early exit. Preserve fixed-length behavior when disabled.
  New real/synthetic datasets use approved seed31, 80/20, all-row input min-max
  scaling and training-only target statistics. Existing geographic protocols
  remain unchanged. Precipitation and satellite drag are explicitly deferred.
  Synthetic dimension/sample-size/noise/function parameters remain explicit.
  ProSpar capacities must be labeled against the paper's 5000-point configuration.
- Higher-dimensional Standard Patchwork uses all features and balanced first-PC
  trees, with linear-program bounds/shared-face detection and exact uniform
  rejection sampling on shared faces. Never substitute an approximate sampler.
  Save full-dimensional geometry/positions; label 2-feature plots as projections.
- The separate GPW ablation notebook selects11 existing datasets,1024 supports
  in8 patches, seeds31-35, ranks0/32 and none/fixed/learned stitches (330 runs).
  Preserve data/synthetic seed31 and existing preprocessing/splits. Model seeds
  vary initialization/training; six variants within each dataset/seed must share
  source rows, initial supports/centers, candidate graph and candidate stitches,
  verified by a saved geometry fingerprint before training. Fixed freezes only
  stitches; none removes constraints, never the GPW model. Preserve existing
  rank/prior-dependent q initialization, uncertainty and optimization settings.
  Shared-kernel and powered-product studies are not part of this batch. Explicit
  dataset/seed selection creates fresh finite batches; never auto-resume/skip.
  Save per-run ZIPs before proceeding and descriptive seed summaries with actual
  counts; a single-seed sample SD is undefined, not zero or silently filled.
- VNNGP uses pinned GPyTorch with explicitly selected CPU FAISS searches and
  no numerical retry/jitter escalation. Prediction equations match its official
  mean-field conditional, using two triangular solves for CUDA compatibility.
- VNNGP stochastic KL also uses Cholesky plus two triangular solves on its SPD
  neighbor covariance; do not restore upstream batched torch.linalg.solve.
  Verify values and gradients against the pinned upstream KL implementation.
- Standalone Nested Kriging is the two-layer shared-prior exact-submodel baseline,
  with explicit disjoint groups in arbitrary input dimension and shared ARD RBF
  or Matern-5/2 kernels. Its coefficients are C_M^-1 k_M, not GPW's unit-sum
  agreement weights; latent variance is k(x,x)-k_M^T C_M^-1 k_M. Preserve full
  expert cross-covariances, add observation noise once, and never regularize the
  aggregation matrix implicitly. Singular Cholesky systems raise.
- NK lengthscale fitting minimizes LOO MSE using Adam/autograd (not the paper's
  SPSA optimizer). Equation (18) outputscale calibration requires zero observation
  noise and zero training jitter. LOO must not drop emptied experts silently.

## Repository Map

- `src/pwgp/models/`: model classes.
- `src/pwgp/data/`: acquisition, preparation, loading, and provenance.
- `src/pwgp/initialization.py`: supports, graph, stitches, and ARD setup.
- `src/pwgp/training.py`: optimization and prediction.
- `experiments/`: future experiment definitions and Colab entry points.
- `tests/`: unit and dense-reference tests.
- `data/`: local datasets, ignored by Git.
- `results/`: generated run records, ignored by Git.

## Validation

```powershell
python -m compileall -q src tests
python -m pytest
python -m ruff check src tests
```

A model is not working merely because it trains. Model changes require the full
suite and the relevant dense-reference algebra test.

## Result Contract

Every future result must store the model class and package version; dataset
checksums, split, and preprocessing; complete model, optimizer, and runtime
configuration; seeds; hardware and software environment; training history;
metrics; artifact paths; and full error information for failed runs.

## Evolution Procedure

For each material change:

1. update `TODO.md`;
2. add a dated entry to `docs/EVOLUTION.md` with decisions and tests;
3. update this file when an invariant, directory, or standard changes;
4. update `README.md` when user-facing commands change.

Do not mark work complete without implementation and verification.

