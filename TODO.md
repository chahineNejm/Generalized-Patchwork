# Development Roadmap

Status: `[ ]` planned, `[~]` in progress, `[x]` implemented and verified.

## Foundation

- [x] Fix OISST cold-CUDA allocator reset order; verify requested-device
  initialization and loud initialization failures before publishing the fix.
- [x] Make user ownership of every methodological choice explicit in AGENTS.md;
  require approval for unspecified choices and reinforce the no-fallback policy.
- [x] Convert Colab notebooks to GitHub-backed launchers, using the user-approved
  latest main from https://github.com/chahineNejm/Generalized-Patchwork and
  recording the resolved commit for each run; preserve experiment settings.
- [x] Verify fail-fast runner records, Colab stage blocking, checked subprocesses,
  and uncapped graph-neighbor validation.
- [x] Implement approved finite-value errors and FLIGHT row removal without
  imputation, recording discarded-row counts and source-reconstruction limits.
- [x] Default to author-prepared airline data, initialize from distinct observed
  locations and reject nonpositive initial learned noise; verified and rebuilt Colab.
- [ ] Obtain a trusted author airline pickle: the published Dropbox link is disabled.
- [ ] Discuss any further VNNGP strictness changes; do not edit third-party libraries.

- [x] Working SVGP model class with explicit hyperparameters.
- [x] Working generalized Patchwork model class with explicit hyperparameters.
- [x] Shared training API with visible `tqdm` progress.
- [x] Dataset acquisition, preparation, cache, checksum, and provenance system.
- [x] Keep datasets, results, and checkpoints out of Git.
- [x] Dedicated model package in `src/pwgp/models/`.
- [x] Dedicated experiment directory in `experiments/`.
- [x] Fail-loudly policy with no fallback or silent value changes.

## Model Evaluation

- [x] OISST five-model runner, centralized default hyperparameters with notebook
  overrides, separate exploratory analysis and per-run visuals/optional positions.
- [x] Approve OISST ProSpar prior_jitter=0: learned nuggets alone, no extra fixed
  prior regularizer; retain an explicit required choice in the model class.

- [x] Implement paper-derived ProSpar-GP using the SVGP foundation; verify dense
  values/gradients, explicit initialization/jitter, and Heaton/UKHousing runners.
- [ ] Choose ProSpar experiment temperatures, exponent, nuggets and whether to
  add fixed prior jitter; these remain required call-time settings, without defaults.

- [x] Implement standalone two-layer Nested Kriging, dense-reference algebra,
  explicit partitions, shared RBF/Matern-5/2 kernels and noiseless LOO fitting.

- [x] Record benchmark selections and paper/code references in target.txt,
  distinguishing paper-linked, author-associated, unavailable and unverified repositories.
- [x] Inventory benchmark hyperparameters, current Colab defaults and reported
  overlapping-dataset configurations/sweeps; flag missing values and paper/code mismatches.
- [ ] Resolve unreported settings, exact external splits and the gp2Scale
  California Housing paper/code VNNGP discrepancy before claiming reproduction.

- [x] Add live/saved per-kernel ARD lengthscale and output-variance evolution,
  shared noise SD and epoch-zero/per-epoch hyperparameter records to heat Colab.

- [x] Restore independent production GPW stitches in periodic heat and Colab;
  initialization lines only. Historical straight-line runs are a separate variant.

- [x] Self-contained GPU periodic-heat Colab with call-time stitch learning-rate
  sweep, live snapshots, per-layout archives and no-Git provenance.

- [x] Preview fixed 0/90-degree periodic heat with four starting stitch lines and
  disjoint, temperature-independent uniform support regions; no training.
- [x] Implement line-constrained trainable position/angle and train the approved
  periodic heat layouts, with matching shared data and initial layouts.

- [x] Joined anisotropic heat experiment: two trainable ARD GPW support clouds,
  matched orientation starts, initial/final point maps and held-out predictions.

- [x] Fixed-boundary two-frequency sine diagnostic with full
  `w^T S w + alpha^T R alpha` uncertainty decomposition and a no-R scoring ablation.

- [x] Save/download a provenance-complete archive and release model/GPU memory
  after every UKHousing run, before starting the next model.

- [x] Replace VNNGP stochastic-KL batched general solve with equivalent Cholesky
  triangular solves; verify upstream KL values/gradients and rebuild Colab snapshots.

- [x] Package and verify self-contained UKHousing Colab: SVGP, full-covariance GPW,
  direct-observation standard Patchwork, and GPyTorch VNNGP with explicit provenance.
- [ ] Execute the default four-model UKHousing comparison on Colab GPU; local
  synthetic smoke checks and real-data preparation are verified, not full training.

- [x] Update Heaton Colab GPW to full posterior covariance weighting (A S A^T + R)
  and residual-inclusive training/prediction; verify dense algebra and gradients.

- [x] Three-patch diagnostic: weights from C_U versus C_U+R, with full residual
  variance propagated in both cases on the same fitted variational distribution.

- [x] Run a small 2D central-gap diagnostic comparing current and residual-inclusive
  uncertainty on the same fitted GPW, without changing production equations.

- [x] Add a saved-prediction NLPD noise-sensitivity cell without retraining;
  distinguish observation noise from the missing GPW conditional residual.

- [ ] Execute the self-contained Heaton Colab notebook: GPW 8 x 60, SVGP 480,
  GPW 4 x 120, each 100 epochs / 20 warm-up. User now requests code, not local runs.
- [x] Package and validate the self-contained Heaton Colab notebook with live visuals.
- [x] Verify active-kernel imports resolve the embedded source without a restart.
- [x] Remove native batched POTRS calls and verify equivalent solves/gradients;
  strengthen Colab GPU preflight without changing experiment settings.
- [ ] Common protocol for testing external model classes.
- [ ] Single-model benchmark runner with call-time hyperparameters.
- [ ] Multi-model comparison runner with identical splits and metrics.
- [ ] Hyperparameter test grids without hidden defaults or changed values.
- [ ] Point/support dynamics studies across configurations.
- [ ] Visual diagnostics for support and stitch movement during training.

## Results

- [ ] Immutable record per model, dataset, split, seed, and configuration.
- [ ] Complete model, optimizer, data, runtime, and environment configuration.
- [ ] Training history, metrics, timings, failures, and artifact references.
- [ ] Deterministic run IDs derived from canonical configuration.
- [ ] Result readers for model and dataset comparisons.

## Colab Entry Points

- [x] Install and environment validation cell.
- [ ] Dataset acquisition and integrity-check cell.
- [x] Unit/algebra test cell with visible output.
- [x] Minimal two-model visual smoke test with inline hyperparameters.
- [ ] Single-model training call with inline hyperparameters.
- [ ] Model comparison call with inline hyperparameters.
- [ ] Point-dynamics visualization call.

## User Preferences

- Supply hyperparameters where a model or test is called.
- Always show training progress.
- Prefer explicit errors to automatic recovery.
- Keep functions and comments brief.
- Never silently modify a model, dataset, or requested configuration.

