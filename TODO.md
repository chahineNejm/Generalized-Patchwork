# Development Roadmap

Status: `[ ]` planned, `[~]` in progress, `[x]` implemented and verified.

## Foundation

- [x] Label ablation training progress with dataset/model/seed/rank/stitches and
  refresh logs every20 epochs plus final state, retaining per-epoch histories.

- [x] Verify and explicitly repin revised official UKHousing source files;
  preserve prior caches and fail-loud checksum validation with actual/expected hashes.

- [x] Update OISST training history in place; omit single-run metric bar charts.
- [x] OISST without reconnects: verified checkout reuse and fresh stage workers,
  live progress/visuals, preserved user settings/data/results and loud failures.
- [x] Use agreement weights conditional on U and stitching, independent of q(U),
  preserving full predictive variance; dense references and all 152 tests pass.
- [x] OISST partial reruns: explicit stage retry, verified prepared-data reuse,
  selected-model fresh batches and saved-prediction reanalysis without training.
- [x] Expose explicit notebook model selection/order separately from per-model
  overrides, with errors for unknown, duplicate or empty selections.
- [x] Validate warm-up only for models that use it; preserve short standard
  Patchwork runs and include model/value details in OISST validation errors.
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
- [x] Use the user's trusted author airline pickle; explicitly convert its legacy
  format in an isolated environment to a pinned NPZ with conversion provenance.
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

- [x] Add FLIGHT-only six-run Kaggle ablations: seed31,ranks0/64,none/fixed/
  learned stitches,1024 supports/8 patches,existing100/20 and stopping controls.
  Add a local ZIP inbox and tracked experiment index;321 tests, Ruff and
  compileall pass. Real FLIGHT GPU training remains user-run.

- [x] Add12 shared-kernel GPW runs: California/PROTEIN/TCO/OISST, seeds31-33,
  1024 supports/8 patches,rank32,learned stitches. Share learned ARD/output
  variance only; leave q trainable/untied. Save local-reference manifests;
  no automatic loading/rerunning local results. Verify dense algebra and push.

- [x] Add a separate44-run Kaggle comparison for seeds32/33 on California,
  PROTEIN, TCO and OISST with fixed seed31 data; preserve audited configurations,
  including explicit historical ProSpar T=1/c=1. Verify, commit and push.

- [x] Add evidence-based experiment completion categories and distinct seed counts
  to the Excel plan; retain actual run configurations and add dataset-specific
  GPW scaling studies with unapproved protocol/grid choices left UNDECIDED.

- [x] Change the four-dataset ablation defaults to seeds31-33 (72 runs),
  preserving all other controls and explicit seed/dataset selection; verify
  generated notebook and settings in fresh workers;28 ablation tests, Ruff,
  compileall and whitespace checks pass on CPU.

- [x] Narrow default GPW ablation selection to California Housing, PROTEIN,
  TCO and OISST (120 runs), preserving all factors/protocols and explicit
  access to the original11-dataset study;27 ablation checks, notebook/settings
  reproducibility, isolated workers, Ruff and compileall pass on CPU.

- [x] Verify the approved330-run GPW ablation notebook: stitches none/fixed/
  learned, V ranks0/32, seeds31-35,1024 supports in8 patches, matched initial
  geometry per dataset/model seed and preserved existing data seed/splits;
  per-run archives, live plots and descriptive seed means/sample SDs. Full
  regression suite and final ablation/fresh-worker checks pass on CPU; Kaggle
  GPU execution remains user-run, not locally verified.
- [ ] Add the requested kernel-hyperparameter comparison; clarify shared learned
  kernels versus frozen common values and its relationship to the factorial study.

- [x] Set future ProSpar experiment starts to author T_j=0.1 and c=2, preserving
  independent temperatures and all other settings; verify defaults and weighting.
- [x] Reduced61-run Kaggle batch: no FLIGHT/VNNGP, PC-tree-only Standard
  Patchwork, both existing GPW/ProSpar sizes, D5/train50k synthetic pair;
  100/20 schedule with approved training-loss early stopping, patience10,
  relative improvement0.001, last-model evaluation and final artifacts; all248 tests pass.
- [x] Kaggle default selection excludes VNNGP, puts FLIGHT last and exposes
  per-completion ZIP download links; verify saved results survive later failures.
- [x] Multi-dataset Kaggle notebook: explicit selections/overrides, 3/1 schedule,
  reviewed training gate, full outputs and archives. All 22 configurations and
  source/preprocessing/capacity choices approved; all 227 tests pass locally.
- [x] Obtain explicit approval to commit/push the verified Kaggle changes;
  test execution on Kaggle GPU remains user-run, not locally verified.
- [x] Extend Standard Patchwork PC-tree partitions/ARD kernels beyond 2D and
  add a dimension-independent comparison runner; dense tests and full suite pass.
- [ ] Precipitation and satellite drag are explicitly deferred by the user;
  obtain verified source files before implementing/running those adapters.

- [x] Approved UKHousing ten-configuration Colab: RBF, 100/20, SVGP 1024,
  GPW/ProSpar 8x128 and 4x256, VNNGP K32/K128, three reduced-data partitions;
  fresh workers, explicit selection/overrides, immutable verified data and archives.

- [x] Add approved 8-region balanced PC-tree Standard Patchwork with the same
  GPW initial observations, seeded uniform stitches, saved tree and dense tests.
- [x] Match Standard Patchwork equation (8) joint-likelihood training; dense
  objective/gradient checks pass. Retain separate regional kernels as approved.
- [x] Preserve the Colab-saved OISST notebook in results/notebook_backups before
  regenerating its output-free launcher, retaining explicit notebook settings.
- [x] Add OISST Standard Patchwork on GPW's exact initial selected rows: a 4x2
  geometric grid or initial-cluster Voronoi regions, empty cells raise; verify
  geometry, dense algebra, saved row identity, notebook controls and full suite.
- [ ] Clarify whether the user's decreasing RMSE refers to epochs, points or
  patches before attributing the trend to a specific mechanism.
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

