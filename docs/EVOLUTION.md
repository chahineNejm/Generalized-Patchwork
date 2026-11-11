# Evolution Log

## 2026-11-11 - OISST comparison, GitHub launchers and centralized defaults

- User approved ProSpar prior_jitter=0 for the OISST comparison: learned nuggets
  alone, without an extra fixed prior term. The model API still requires this
  argument explicitly; no numerical fallback or global default was introduced.
- Added default_hyperparameters.py, explicit notebook overrides and an OISST
  runner calling the existing five model implementations. Protocol: daily
  2025-01-15 ROI (-80,-30,20,50), seed 31, 64/16/20 random split, declared-ROI
  input normalization and training-only target standardization. Proposed model
  settings are recorded in the defaults file, not paper-reproduction claims.
- Added separate exploratory_analysis.py: dataset summary/maps/histogram,
  saved-run metrics, loss/kernel evolution and optional initial/final inducing
  positions. Per-model visuals/ and raw motion snapshots are separate from code.
  Standard Patchwork is correctly labeled as having no inducing variables.
- Four main Colab notebooks now import latest main from the user's GitHub repo,
  record resolved commit and source archive, and use explicit sparse code-only
  checkout to exclude unrelated dataset files. Kept historical
  Heaton/UKHousing/heat experiment budgets and GPU checks unchanged.
- Comparisons stop after errors with tracebacks; each completed OISST run is
  archived with shared provenance and model/GPU memory released. Added tqdm for
  prediction, weighting, exploration and sequential comparison, alongside training.
- User authorized committing/pushing this work to main after verification.
- User separately authorized removing airline.pickle from the prior unpublished
  commit, preserving its local file and ignoring it, before normal main push.
  No published history is rewritten; all other prior commit content is preserved.
- Verification: 123 tests pass; Ruff, compileall and diff whitespace checks pass.
  Includes real local Git clone/import/latest-main checks, five-model synthetic
  end-to-end artifacts, explicit overrides, optional positions, memory release
  and fail-fast comparison. Cached OISST preparation verified 21,370 usable rows:
  13,676 training / 3,419 validation / 4,275 test; SHA-256 recorded.
  Four dependency warnings remain. No full OISST training or Colab GPU run claimed.

## 2026-11-11 - User-owned methodology and GitHub notebook decision

- Explicitly require user approval before implementing any unspecified
  methodological choice, including data protocols, model equations, training,
  hyperparameters and numerical regularization. Recommendations, paper/library
  defaults and silence do not constitute approval. Unresolved choices must not
  receive invented defaults; approved decisions must be preserved and recorded.
- Reinforced the user's preference for errors over fallbacks, including hidden
  changes of solver, device, data source, model or configuration after failure.
- User selected GitHub installation for thin notebooks and latest main from
  https://github.com/chahineNejm/Generalized-Patchwork. Conversion remains
  pending; resolved commits must be recorded rather than claiming a fixed pin.
- Verification: reviewed documentation changes and git diff --check. No model,
  dataset, numerical setting or experiment behavior changed in this rules update.

## 2026-11-11 - ProSpar-GP implementation and benchmark adapters

- Added ProSparGP on the SVGP foundation: local ARD RBF experts, independent
  full-covariance q(U_j), arbitrary input dimension and unequal inducing budgets.
  Learned positive temperatures, exponent and per-inducing nuggets require
  explicit initial values. Reused approved SVGP bounds and Cholesky constraints.
- Kept the unresolved extra-jitter decision at the call site: prior_jitter is
  required, nonnegative and never escalated; zero uses K_j + D_j alone. This
  model does not inherit NumericalConfig.prior_jitter implicitly.
- Followed the approved generative derivation (6)-(13) of arXiv:2311.08752v1:
  E_q(U) log p(y|U) integrates f first, and prediction uses b_j squared for
  propagated q variance. Did not copy the inconsistent printed (14)/(16).
  This is paper-derived code, not an author-source reproduction.
- Connected shared tqdm training, Heaton and UKHousing, with explicit config,
  model/paper provenance, learned parameters, checkpoints and existing visuals.
  Existing experiment lists/settings are unchanged; CPoE is not implemented.
- Used an algebraically equivalent stable inverse-softplus to support large
  positive temperatures without overflow, clipping or parameter substitution.
- Verification: all 112 tests pass, including dense values/gradients,
  integrated-ELBO bound, minibatch scaling, unequal budgets, noise, freezing,
  fail-loud validation and synthetic runner records. Ruff, compileall and
  git diff --check pass. Rebuilt all three embedded Colab source snapshots;
  snapshot equality and cell compilation pass. Four dependency deprecation
  warnings remain. No full real-data or Colab GPU run claimed.

## 2026-11-11 - Observed inducing initialization and author airline source

- Replaced global k-means support centers with observed_supports: seeded sampling
  of distinct actual training locations. Hierarchical patches still cluster the
  domain, then sample real rows within each patch. Removed UK nearest-row snapping.
  The same shared routines serve Heaton/UKHousing and arbitrary input dimensions.
- The existing logit parameterization needs interior coordinates. Only distinct
  interior training rows are eligible; coordinates are never moved/clamped and
  insufficient counts raise. Explicit synthetic layout experiments are unchanged.
- VNNGP now rejects nonpositive initial SD before squaring; GPW/SVGP already
  reject these through their configured positive bounds. No learned-noise transform,
  variance floor, retry behavior or third-party library file was changed.
- FLIGHT download/prepare now targets the author airline.pickle archive referenced
  by Hensman's VFF SVGP experiment. The published Dropbox URL currently returns
  shared_link_disabled HTML; reject it, with no raw reconstruction fallback.
  A trusted local pickle can be explicitly imported. Real archive/schema loading
  cannot be verified until a valid copy is available; fixture tests are separate.
- Author preparation maps the eight known predictors, converts HHMM times to
  minutes as in airline_svigp_rbf.py, retains all usable rows without imputation,
  and records source/feature/time/cleaning metadata. No automatic split, sampling
  or scaling; do not identify this with a verified original 2013 paper split.
- Cache behavior/manifest checking and no-jitter-escalation settings unchanged.
  All 96 tests pass, including 20 focused initialization/airline/finite-input
  tests and existing dense-reference/VNNGP gradient comparisons; four dependency
  deprecation warnings remain. Ruff and compileall passed; all three embedded
  Colab notebooks rebuilt. Real author-archive download remains blocked by its
  disabled source link; no full benchmark or CUDA run was performed.

## 2026-11-11 - Approved finite checks and FLIGHT row removal

- Added errors for nonfinite initial/raw parameters and nonfinite predictions;
  no imputation, flooring, constraint or numerical-retry changes were made.
- FLIGHT removes malformed/missing/nonbinary status flags instead of filling
  them with zero. Nonfinite/malformed predictors and targets remove the row.
  Genuine zero values remain unchanged; existing age filters/sampling remain.
- Save explicit removal counts and cleaning policy in a preparation JSON file
  and the dataset manifest. NaN in model state is an error; unusable raw dataset
  rows are removed under this separate, user-approved preprocessing policy.
- Checked Hensman's later VFF airline SVGP code: it consumes airline.pickle;
  its download script provides that prepared archive, not raw cleaning logic.
  Our Data Expo reconstruction is therefore not claimed identical to the
  original 2013 benchmark cleaning or split. No new time conversion or split
  was silently adopted from that later experiment.
- Noise research only: GPy Gaussian uses Logexp for variance; GPyTorch's default
  Gaussian likelihood uses a transformed variance bounded below by 1e-4.
  Our existing SD interval and VNNGP squaring behavior remain unchanged.
- Cache replacement, manifest verification, inducing initialization, Heaton
  preprocessing and VNNGP strictness/flooring are unchanged.
- Verified 11 focused tests for nonfinite parameters/predictions, row removal,
  no imputation, empty-data failure and cleaning provenance. All 87 tests pass,
  including existing dense-reference and VNNGP upstream gradient comparisons;
  four dependency deprecation warnings remain. Compileall and Ruff passed.
  Rebuilt all three embedded Colab notebooks; no full benchmark or GPU run.

## 2026-11-11 - Revert unrequested audit fixes

- At the user's request, reverted the additional parameter/prediction guards,
  FLIGHT parser changes, cache/manifest checks, VNNGP noise/numerical changes and
  Heaton trend-rank check, together with their tests and current-state docs.
- Retained only the explicitly requested failure recording/propagation,
  downstream Colab blocking, checked commands and removal of the neighbor cap.
- The earlier entry below records the superseded intermediate state, not the
  current implementation. Other audit findings remain for explanation/discussion.
- Rebuilt all three embedded notebooks from restored source. All 76 tests pass,
  including eight retained execution-safety tests and the dense-reference suite;
  four dependency deprecation warnings remain. Compileall and Ruff passed.

## 2026-11-11 - Fail-fast experiment execution

- Replaced Heaton's process-global exception hook with scoped failure recording;
  UKHousing now also records full tracebacks and failed status. Both re-raise the
  original exception, including interrupts, so subsequent models do not run.
- Colab code cells latch a failed stage and refuse subsequent cells until a
  runtime restart. Package installation and the legacy smoke notebook's Git/test
  commands use checked subprocesses. Preserve final expression display values.
- Removed the graph-neighbor cap: impossible requests raise rather than reducing
  the requested count. A single patch requires an explicit zero neighbors.
- Corrupt checksum-pinned caches now raise instead of silently redownloading.
- Reject nonfinite initial/raw parameters and SVGP predictions, malformed FLIGHT
  numeric fields and unknown cancellation/diversion flags. Genuine missing-row
  filtering remains an explicit preprocessing policy, not parser coercion.
- Verify dataset manifest checksums/sizes before loading, preparing or reusing
  registered downloads. Missing manifest members now raise. Explicit load paths
  must be registered; unpinned first acquisitions remain locally fingerprinted,
  not independently authenticated against an upstream checksum.
- VNNGP rejects invalid noise before squaring, enforces strict Cholesky settings
  inside calls, turns numerical correction warnings into errors and rejects
  outputs that would trigger GPyTorch's variance floor. No flooring replacement
  or adaptive jitter was added. Heaton detrending now checks full design rank.
- No model equations, inducing initialization, hyperparameter bounds or approved
  numerical regularization changed. Author-code comparison confirms explicit
  missing-coordinate filtering in HIP-GP; GPy SVGP accepts caller-supplied Z and
  the official GPyTorch example learns positions from 500 training rows, neither
  shown path imposing a uniqueness check.
- Verified 18 focused failure/validation tests and all 86 repository tests,
  including dense-reference algebra and VNNGP upstream value/gradient parity;
  four dependency deprecation warnings remain. Compileall, Ruff and diff
  whitespace checks passed. Rebuilt all three embedded Colab snapshots.
  No full benchmark run or CUDA verification performed.

## 2026-11-11 - Standalone Nested Kriging baseline

- Added NestedKriging with explicit disjoint groups in arbitrary dimension,
  shared ARD RBF/Matern-5/2 kernels and exact noisy/noiseless GP submodels.
  Aggregation retains full prior cross-expert covariance, uses unconstrained
  C_M^-1 k_M coefficients and k(x,x)-k_M^T C_M^-1 k_M latent variance; observation
  noise is added once. No fusion jitter, unit-sum constraint or pseudoinverse.
- Added explicit lengthscale fitting through leave-one-out MSE, tqdm/callbacks
  and the paper's noiseless outputscale calibration. Adam/autograd is explicitly
  distinguished from the original SPSA optimizer. Empty LOO experts, singular
  systems and out-of-bounds calibrated values raise. No global training matrix
  is stored; LOO recomputes local factors and is not performance-optimized.
- Documented the API and scope; existing GPW, standard Patchwork, VNNGP and
  benchmark run lists are unchanged. Rebuilt all three embedded Colab snapshots.
- Verified 14 focused tests: dense projection means/variances/cross-covariances
  and gradients for both kernels, sklearn kernel parity, full-GP special cases,
  prior recovery without a unit-sum constraint, noise/batch parity, LOO exclusion,
  equation (18), progress callbacks, fail-loud singular systems and the 30-point /
  15-expert / 101-query noiseless configuration. Full suite: 68 passed, four
  dependency deprecation warnings; Ruff, compileall and patch whitespace passed.
  No full benchmark experiment executed; CUDA/large-data performance unverified.

## 2026-11-11 - Benchmark hyperparameter inventory

- Added compact model-control/current-Colab inventories and source-referenced
  configurations for overlapping target datasets to target.txt. Preserved user
  edits, separated fitted parameters from settings, and recorded sweeps/repeats
  without inventing validation-selected winners or unspecified values.
- Checked supplied papers, VNNGP/SWSGP primary publications, the published CPoE
  author version, and gp2Scale California notebooks at commit
  b968d975bd80fe8e23dca3a2ad62e398eeb008d4. Its CA_nngp notebook is an exact
  local-GP predictor (25 neighbors), not the paper's 50-neighbor VNNGP. The
  SVGP notebook uses isotropic Matern-3/2, 500 full-batch updates, and excludes
  likelihood noise from its optimizer. Recorded both discrepancies explicitly.
- Marked ProSpar settings as supplied-preprint values, TCO/TCO.L2 and airline
  classification/regression as distinct tasks, and current UKHousing as a
  reconstruction. CPoE table/split-formula inconsistency remains unresolved.
- Documentation only; checked local configuration sources, source tables and
  patch whitespace. No models changed, external code executed, or training run.

## 2026-11-11 - Benchmark references in the user's target list

- Shortened the benchmark section at the user's request; retained model selection,
  paper/repository URLs and verification caveats. Existing experiment notes unchanged.
- Preserved target.txt experiment/dataset selections and added SVGP, standard
  Patchwork, ProSpar-GP, VNNGP and Nested Kriging references. Kept aggregation
  diagnostics, optional CPoE/SWSGP and the small-data full-GP reference separate.
- Verified GPy's paper citation, GPyTorch's VNNGP example and the author-associated
  nestedKriging repository. Recorded the CPoE paper's explicit GitHub URL and its
  current 404 response. ProSpar-GP, Patchwork and SWSGP author repositories remain
  unverified; no unrelated code was substituted or claimed to be official.
- Documentation only: reviewed target contents and patch whitespace. No model,
  notebook, training or test execution changes; full numerical suite not run.

## 2026-10-29 - Live kernel hyperparameter evolution

- Added diagnostic recording of transformed x/y ARD lengthscales, per-kernel
  output variance and shared observation noise SD from epoch zero and each epoch.
- Heat training live view now combines point positions, objective and four
  hyperparameter panels. Each layout saves kernel_history.json and standalone
  kernel_evolution.png; Colab displays and downloads them after completion.
- Warm-up markers are explicit. Parameters are in normalized-input/standardized
  target model units. No model, optimizer or training-budget changes.
- Verified epoch-zero values, exact warm-up invariance, finite traces and final
  values matching saved metrics; inspected combined live and standalone figures
  from a two-epoch CPU smoke test. Full suite: 54 passed; Ruff and compileall
  passed. All three Colab snapshots rebuilt. No full experiments executed and
  CUDA diagnostics remain unverified locally.

## 2026-10-29 - Restore independently trainable GPW stitches

- User rejected the experimental collinearity constraint as not their method.
  Removed the StraightLineParameter override from build_model; production GPW
  retains its independent UnitCubeParameter stitch coordinates. The legacy class
  remains only for reproducing the already-saved straight-line variant.
- Lines now specify initialization only. Removed final-line angle/geometry
  reporting and save actual per-stitch trajectories; plots do not connect final
  stitches as if constrained to a line. Colab learning-rate control is unchanged
  and now acts on every stitch coordinate. Inducing-point side constraints are
  not added: separation is still initialization-only, explicitly recorded.
- Historical line-constrained runs are not results for the restored method.
  Verification: 54 tests passed, including independent-coordinate motion,
  production objective/gradient parity, smoke callbacks and embedded-source
  imports. Ruff and compileall passed; all three notebooks rebuilt. No new
  full experiment run. CUDA remains unverified locally.

## 2026-10-29 - GPU Colab for the stitch learning-rate experiment

- User requested a self-contained Colab and explicitly approved GPU. Added a
  source-embedded notebook with pinned FEM dependency, checksum verification,
  one settings cell and STITCH_LEARNING_RATES sweep over all four approved starts.
- Runner now accepts explicit device selection, optional live/completion callbacks
  and capture_git=False for source-only Colab. CPU remains its library default;
  notebook explicitly selects CUDA with no fallback. Learning-rate baseline 0.003,
  seeds, model algebra, starts and training budgets remain unchanged.
- Saved CPU heat generation and CUDA GP runtime distinction, live progress,
  per-layout model/artifact downloads and device cleanup before the next layout.
- Verified embedded source checksums/equality, code-cell compilation and imports
  in a fresh isolated interpreter; CPU smoke test exercises live/completion
  callbacks and disabled Git capture. Requested unavailable CUDA raises without
  CPU substitution. Full suite: 53 passed; Ruff and compileall passed.
- CUDA forward/backward preflight is included but unverified locally (CPU-only
  PyTorch). No new full experiment requested or executed locally. All three
  embedded-source notebooks rebuilt after runner/documentation changes.

## 2026-10-29 - Training periodic heat with a straight movable stitch line

- User approved the corrected four layouts for training. Added experiment-local
  StraightLineParameter: angle plus normal offset, stitches at fixed fractions
  of the line/square intersection. No line-target attraction or temperature-based
  point initialization; no independent stitch motion or projection/clamping.
- Reuses production GPW and shared fit without changing covariance/weight algebra;
  fit already accepts its two packed stitch parameters through stitch.raw.
- All four approved starts use 28 supports/model, 10 stitches, 500 shared noisy
  observations, 350 epochs / 40 kernel warm-up. Points are unconstrained by side
  after initialization; kernels remain two axis-aligned ARD RBFs.
- Verification: initial geometry equals preview, points remain collinear/in-domain,
  autograd gradcheck passes, objective and chain-rule derivatives match an
  unconstrained production GPW at identical points; end-to-end saves line motion.
  Full suite: 50 passed; Ruff and compileall passed; both source snapshots rebuilt.
- Completed all four runs under results/stitch_line_heat/20261029T102422821574Z.
  Standardized held-out latent RMSE: vertical 0.11063, shifted 0.12817, diagonal
  0.19341, horizontal 0.28911. NLPD: -0.75316, -0.63921, -0.23926, 0.35999.
  Final angles (mod 180): 90.1077, 97.8377, 47.7940, 6.6527 degrees.
- Initial/final points and stitch lines, fitted/error/full-residual SD maps and
  training curves inspected visually. Losses still decrease at 350 epochs;
  no convergence or material-interface recovery is claimed. Initial separation
  is preserved in saved starts but is not imposed as a training constraint.

## 2026-10-29 - Corrected periodic heat and stitch-line preview

- User clarified: heat pattern stays horizontal on the left and vertical on the
  right; only the starting stitch line changes. Supports start on separate sides
  of each initial line, not mixed across it and not concentrated near heat sources.
- Added stitch_line_preview: aligned vertical, x=0.56 shifted vertical, diagonal
  and horizontal lines. Each model has 28 temperature-independent uniformly
  sampled supports in its half-domain; ten collinear stitches per layout.
- Fixed periodic heat uses repeated cosine initial temperature, full-tensor P1
  finite elements through pinned scikit-fem 12.0.2, implicit Euler and insulated
  outer boundaries. This is spatial repetition, not periodic boundary conditions.
- Preview only: saved exact heat/support/stitch arrays, configs and hashes; no
  optimization or GP scores. Line-constrained trainable angle/position remains
  to implement; independent trainable stitch points do not guarantee collinearity.
- Full suite: 47 tests passed, including placement/collinearity; Ruff and
  compileall passed. Figure visually inspected and both Colab snapshots rebuilt.

## 2026-10-29 - Orthogonal anisotropic heat orientation experiment

- Simulate dimensionless heat diffusion in joined left/right plates with swapped
  diagonal diffusivity, insulated exterior and harmonic face fluxes at the interface.
  SciPy matrix-exponential evolution; conservative finite-volume spatial discretization.
- Compare four rotations of the same two ellipse-shaped support clouds using two
  ARD RBF submodels. Same training data/noise, initial isotropic ARD parameters,
  optimizer and seed; support positions train globally, stitches fixed at x=0.5.
- Save point motion, prediction/error/full residual SD figures, per-run checkpoints,
  configurations, data/split, scaling, source hashes, histories and failure traceback.
- Completed four 350-epoch / 40 kernel-warm-up CPU runs, seed 31, 500 shared noisy
  observations, 28 supports per patch, rank 12 and 10 fixed interface stitches.
  Results: results/anisotropic_heat/20261029T100807013725Z. Standardized held-out
  latent RMSE: aligned 0.3521, swapped 0.3695, diagonal 0.3251, horizontal 0.3313.
  NLPD: 0.5100, 0.6185, 0.7926, 0.6866. Losses remain nonstationary; these are
  fixed-budget results, not a converged or initialization-independent claim.
- Verified initial/final point maps and fitted/error/full-residual SD figures
  visually. Heat operator symmetry, negative semidefiniteness, mass conservation,
  orthogonal coefficients and dense matrix-exponential reference pass. Full suite:
  46 passed; Ruff and compileall passed; both embedded notebooks rebuilt.
- This is not a physics-informed GP: only targets come from the heat equation;
  no heat-flux constraint is added. Dimensional material constants are not claimed.

## 2026-10-22 - Fixed-boundary two-frequency sine diagnostic

- Added `experiments/two_phase_sine.py` for balanced observations from
  `sin(2*pi*x)` on `x < 0` and `sin(10*pi*x)` on `x >= 0`, with a fixed stitch
  exactly at the continuous phase boundary `x = 0`.
- The runner preserves the full latent variance as `w^T S w + alpha^T R alpha`,
  verifies the decomposition against production prediction, and reports an
  evaluation-only no-R ablation for NLPD and 95% coverage on fresh noisy data.
- Results include regional regression/calibration metrics, learned parameters,
  predictions, the model state, training history, figures, data checksums, source
  hashes, Git state, software/runtime provenance, and full failure information.
- Added a regression test checking the fixed stitch, sum-to-one
  fusion, nonnegative residual contribution, and exact variance decomposition.
- Verification: the seeded 500-epoch CPU run completed and both figures were
  inspected; 43 tests passed, Ruff passed for `src`, `tests`, and the new runner,
  and compileall passed for `src`, `tests`, and `experiments`. Rebuilt both Colab
  source snapshots after the README change.

## 2026-10-21 - Save and release each UKHousing run

- Save all existing artifacts before deleting the model and snapshot closure,
  collecting Python cycles and releasing unused CUDA cache on the selected device.
- Colab archives/downloads each completed run with shared source, split, data
  manifest, preprocessing and current comparison before advancing to the next run.
  Shared CPU dataset and small metric summaries remain available between runs.
- Verification: full suite 41 tests passed, including weak-reference checks that
  all four model objects are released after saving, archive contents/overwrite
  protection and incomplete-run rejection. Ruff and compileall passed. CUDA cache
  release and browser downloads require Colab verification; local tests are CPU.
- No training/model configuration changes. Both embedded notebooks rebuilt.

## 2026-10-21 - VNNGP stochastic KL CUDA compatibility

- User reported CUDA no-kernel-image at GPyTorch NNVariationalStrategy's batched
  torch.linalg.solve during the first VNNGP epoch, after the neighbor graph built.
- Override only the stochastic mean-field KL helper in the existing adapter:
  evaluate the same Gaussian conditional KL with Cholesky plus two triangular
  solves. No catch/retry, backend fallback, model downgrade or extra jitter.
- Training budgets, split, seeds, likelihood, kernels and optimizer settings are
  unchanged. User's Colab trace shows a 60-epoch local configuration; no supplied
  executed notebook is available to overwrite or infer its other settings.
- GPU preflight now checks both first-K and stochastic-KL batches at K=4/B=8
  and production K=32/B=512, including backward and prediction.
- Preserved pinned upstream default-dtype jitter arithmetic for numerical parity.
- Verified KL values and gradients against upstream at K=4/B=8 and K=32/B=512;
  training regression forbids torch.linalg.solve through first-K and stochastic
  batches. Rebuilt both embedded Colab snapshots and verified source equality.
- Validation: 40 tests passed; Ruff and compileall passed. Local PyTorch is CPU
  only; the updated GPU preflight and full Colab training remain unverified here.

## 2026-10-19 - Four-model UKHousing Colab

- User requested SVGP, current GPW, standard Patchwork and VNNGP in a ready-to-run
  Colab notebook. Embedded source, automatic checksum-pinned downloads, shared
  64/16/20 split, training-only target scaling, RBF/Gaussian models, float64 GPU.
- Could not verify the original VNNGP prepared UKHousing snapshot. Reconstructed
  2018 flats from the HIP-GP authors' documented Land Registry/FreeMapTools sources:
  185,254 transactions, 118,562 train / 29,640 validation / 37,052 test, seed 31.
  Explicitly NOT a paper-table reproduction; coordinate/target snapshots, kernel
  and normalization differ. Raw data hashes are pinned; no alternate sources.
- SVGP M=1024; GPW 8 x 128, q rank 32, eight stitches/edge; VNNGP M=N_train,
  K=32 with exact CPU FAISS neighbor graph and GPU GP calculations. Budgets shown.
  All 100 epochs; variational kernel/noise warm-up 20 epochs. Exact Patchwork has
  no q warm-up and runs 100 conditional-likelihood updates at lr 0.01.
- Direct Patchwork uses all training targets, fixed rectangular regions with max
  512 points, and two boundary stitches/edge. Actual partition: 260 patches,
  max 507 targets, 1,472 constraints. It is not a same-partition GPW ablation.
- Standard Patchwork factors local observation covariances and a global stitch
  Schur complement; no inducing/variational surrogate. Dense likelihood and
  posterior checks added. VNNGP uses GPyTorch 1.15.2/linear_operator 0.6.1,
  FAISS CPU 1.13.2; explicit CPU search adapter, numerical retries disabled.
- VNNGP prediction uses equivalent mean-field conditional equations via two
  triangular solves to avoid batched POTRS; compared with upstream predictions.
- Saved configs, environment/source hashes, split, preprocessing, metrics in
  standardized log-price and original log GBP, checkpoints, live/final visuals.
  Full default Colab GPU run remains unverified.
- Verification: real source acquisition/schema/counts and partition limits checked;
  all four CPU models complete two-epoch synthetic end-to-end runs with metrics,
  checkpoints, progress/final PNGs, inspected visually. Dense direct-Patchwork
  likelihood/prediction/gradient reference and upstream VNNGP prediction/neighbor
  equivalence pass. Notebook cells compile, embedded sources match current files,
  and the embedded package imports in a fresh isolated interpreter. Corrupt cached
  sources fail before network access. Final verification: 37 tests pass, Ruff and
  compileall pass; both UKHousing and Heaton embedded-source notebooks rebuilt.

## 2026-10-19 - Full posterior covariance GPW notebook update

- User approved new weights using A S A^T + R, including R before agreement and
  propagating alpha^T R alpha in the final latent variance, training and prediction.
- R comes from the existing regularized stitched local prior, with cross-patch
  residual covariance retained; no independent residual parameters, no clipping,
  fallback or new jitter. Existing epsilon, bounds, hyperparameters and schedules
  are unchanged. SVGP is unchanged.
- Weights depend on S and gradients flow through this dependence. The training
  criterion is Gaussian expected log likelihood minus KL; it is not asserted to
  be an ELBO for a fixed generative model with posterior-dependent weights.
- Replaces the previous R_X=0 production assumption, which did not match the
  report's retained residual. Old diagnostics/results remain historical artifacts.
- Rebuild the self-contained notebook from current source. The user's executed
  Downloads notebook is read-only input and remains untouched.
- Verification: all 27 tests pass, including dense covariance/weights/mean/variance
  and training-criterion references at q ranks 0/3/6, gradients through weights,
  single-patch prior variance recovery far from supports, plus embedded-package
  import/source checks. Ruff passes. GPU preflight now also runs the revised GPW
  forward/backward. Full Heaton training on Colab GPU has not been run locally.
- Historical diagnostic scripts now use the revised production model if rerun;
  their already-saved configurations and results retain the old equations.

## 2026-10-19 - Three-patch residual order diagnostic

- User requested a before/after weighting comparison, distinct from the previous
  omission-of-residual comparison. Both outputs now retain full C_U+R covariance.
- Same three-patch model, 48 inducing points, fixed geometry/kernel/noise, 100 q-only
  epochs. Compare agreement_weights(C_U) followed by residual propagation versus
  agreement_weights(C_U+R) followed by full propagation; epsilon stays 0.01.
- Production weights use R alone: neither alternative is labeled the unchanged
  production weighting rule. These are evaluation-time rules on one fitted q,
  not separate end-to-end training runs. No production model equations changed.
- Executed results/residual_weight_order/20261019T105919877267Z. In the gap,
  R-after weights: latent RMSE 0.4950, NLPD 0.8906, mean latent SD 0.8266;
  R-before weights: latent RMSE 0.1769, NLPD 0.2460, mean latent SD 0.4635.
  Both have 100% observation coverage, so neither demonstrates nominal calibration.
- Verified sum-to-one weights and full-variance decomposition during execution;
  figures inspected with shared scales. Added a dense covariance projection test.

## 2026-10-19 - Isolated 2D residual-propagation diagnostic

- User requested a quick visual comparison. Added an isolated experiment with a
  circular observation gap, four patches of 12 inducing points, 100 q-only epochs,
  fixed supports/stitches/kernel/noise, float64 CPU, and seed 31.
- Evaluates one fitted model with wSw versus wSw + alpha^T R alpha. The full local
  residual matrix includes stitching cross-covariances. Mean and weights are unchanged.
- This is a post-hoc diagnostic, not training a residual-inclusive model or changing
  the production GPW construction. Saves settings, checkpoint, predictions, metrics,
  training history and visuals.
- Executed successfully: results/residual_2d/20261019T105240855259Z. In the withheld
  gap, mean latent SD increases from 0.1049 to 0.4353, observation NLPD decreases
  from 5.6544 to 0.5285 and 95% observation coverage increases from 38.09% to 100%.
  Mean/RMSE are identical. The 100% coverage is conservative, not proof of calibration.
- Verified alpha sums to one, aggregated residual is nonnegative without clipping,
  and baseline mean/variance match production prediction. Ruff passes; inspected the
  saved figure with matched SD scales. No production model equations were changed.

## 2026-10-19 - Batched CUDA solve compatibility

- Colab failed at the native batched `torch.cholesky_solve` after passing the
  original single-matrix GPU check. The trace points to a CUDA kernel/build
  compatibility issue; the exact remote GPU/build combination is not available.
- GPW and SVGP now use the same Cholesky solution via two explicit triangular
  solves. No model equations, dtype, jitter, epsilon, or hyperparameters changed.
- Added value, gradient, broadcasting, and native-kernel-exclusion regressions.
- GPU preflight now covers 8 x 60 and 4 x 120 local blocks, batched weight
  systems, and the 480-variable global solve with forward/backward checks.
  GPU model, architecture, PyTorch, and CUDA build details print before checking.
- Restart a previously used Colab runtime before loading the updated snapshot.
- Full remote GPU execution is unverified; failures remain explicit without
  CPU fallback, retries, or automatic numerical changes.
- Verification: all 21 local tests and Ruff pass, including existing dense GPW
  references, native-solver value/gradient comparisons, broadcasting, and
  forward/backward model evaluation with the native solver disabled in tests.

## 2026-10-19 - Colab active-kernel import correction

- Fixed the standalone notebook bootstrap to register both the source-layout
  `src/` directory and project root, then invalidate import caches.
- Editable-install startup hooks are not assumed to run again in an already
  active notebook kernel. No runtime restart or model-setting change is needed.
- Added a subprocess regression that extracts the actual embedded snapshot and
  verifies that the imported models and runner resolve to it, not an installed
  workspace package. The regression failed before the bootstrap correction.
- Verification: the rebuilt notebook passes that import regression; all 15 local
  tests and Ruff pass. Full Colab GPU execution remains unverified.

## 2026-10-19 - Standalone Colab delivery

- User changed the training schedule to 100 total epochs / 20 warm-up, approved
  GPW 8 x 60, SVGP 480, and GPW 4 x 120, then requested ready-to-run Colab code.
- Stopped the initial CPU attempt at 3 completed epochs; it is explicitly marked
  stopped, not a completed benchmark. Stopped the local CUDA package download.
- The local machine has a GTX 1650 Max-Q (4 GB), but its existing PyTorch build
  is CPU-only. No GPU driver or existing CPU environment was replaced.
- Added explicit CUDA selection with no CPU fallback; model equations unchanged.
- Standalone mode records source hashes instead of assuming a Git checkout.
- Notebook embeds the current source archive, installs dependencies, verifies a
  GPU/float64 Cholesky and backward pass, downloads the pinned official data,
  runs the fixed configurations, and displays live and final visual diagnostics.
- No GitHub repository, separate source upload, or manual code edits are needed.
- Colab runtime must have a GPU; this is an explicit requirement, not a fallback.
- Verification: 14 local tests pass, including embedded-source byte equality,
  source SHA-256, Python cell/source compilation, and exact 100 / 20 schedule.
  The official data schema, split, training-only standardization, and Gaussian
  scoring formulas were checked locally. Package wheel build, Ruff, and source
  compilation pass. Full Colab GPU execution is unverified.

## 2026-10-19 - Heaton MODIS comparison in progress

- Added an explicit single-model runner using the authors' pinned RData file,
  SHA-256 verification, and official 105,569 / 42,740 train/test split.
- Training-only OLS longitude/latitude trend and residual standardization;
  test targets are used for final scoring only.
- Requested settings: GPW 8 x 60 and SVGP 480 inducing points, 300 epochs,
  75 epochs before kernel/noise updates, batch size 2,048, seed 31, float64.
- Added an optional shared-trainer epoch callback for saved progress and
  inducing/stitch trajectories without changing model equations.
- Runner records configuration, existing bounds, environment, source hashes,
  checkpoints, metrics, Gaussian observation intervals, and diagnostic maps.
- No fallback, model replacement, retry, or automatic hyperparameter change.
- Verification and run outcomes will be recorded after completion.

## 2026-10-18 - Local foundation created

Implemented:

- Installable `src/` package with SVGP and generalized Patchwork classes.
- Patchwork structured prior, Nested-Kriging fusion, and block-plus-low-rank
  variational precision extracted from the reference notebook.
- Shared initialization, training, batched prediction, and algebra tests.
- OISST, FLIGHT, TCO.L2, and verified ICETHICK data workflows.
- Checksums and source provenance manifests.

Decisions:

- Paper-scale sweeps remain outside the current phase.
- ICETHICK is a verified import because its official host requires a CAPTCHA.
- Recovery behavior is forbidden. Singular algebra, schema mismatches, missing
  dependencies, and unavailable sources raise errors.
- Every training loop uses `tqdm`.

Verification:

- Created `.venv` with Python 3.12 and installed all declared dependencies.
- Ten tests pass, including dense Patchwork references, SVGP gradients, data
  provenance, strict error behavior, and one-epoch training for both models.
- Ruff and source compilation pass.
- Live NOAA acquisition passed for OISST 2025-01-15. The strict Gulf Stream
  loader returned 33,522 finite observations and wrote a SHA-256 manifest.
- The minimal visual smoke run passed for both models at 120 epochs. SVGP loss
  moved from 134.215 to 0.256 and Patchwork loss from 120.562 to 3.293.

## 2026-10-18 - Reference parity correction

- Verified the reusable Patchwork class against `test-map-fin-sep (3).ipynb` on
  identical tensors. Prior, fusion, variational moments, KL, and ELBO agree to
  numerical precision.
- Corrected the shared trainer so a declared zero warm-up is actually zero.
- Aligned the visual smoke test with the notebook's separate SVGP and PW-NK
  optimizer settings, PW-NK rank, and latent predictive variance.

## 2026-10-19 - Post-hoc noise NLPD diagnostic

- Added a Colab cell reusing saved test means, latent variances, and learned
  observation noise to report NLPD at explicit noise-variance multipliers.
- Multiplier 1 reproduces the saved observation variance; 0 excludes noise.
  Saves a table and plot; no training, hyperparameter updates, or model changes.
- This does not restore the report's conditional residual r(x,x). The current
  implementation's R_X=0 differs from the report's retained diagonal residual.
  Previous descriptions treating R_X=0 as the user's construction were incorrect.

