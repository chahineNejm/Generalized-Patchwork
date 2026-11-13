# Evolution Log

## 2026-11-13 - Explicit FLIGHT800k subset

- User approved800,000 uniformly sampled FLIGHT rows and retained80/20 rather
  than the paper's90/10 split. Use existing data seed31,sampling without
  replacement before preprocessing/splitting. Thus640,000 train/160,000 test,
  shared across all six seed31 rank0/64 stitching ablations. Kernels,model size,
  optimization,numerics and100/20 training schedule remain unchanged.
- Sampling is explicit in notebook DATA_OVERRIDES; full-data FLIGHT presets are
  unchanged. Input scaling uses all selected rows and targets training rows only.
  Save selected source-row indices/checksums in prepared data and each run ZIP.
  This is size-matched,not a verified reproduction of the paper's exact rows.
  Subset/full-data prepared configurations cannot be interchanged. Preserve
  existing full-data prepared identities when no subset is requested.
- Verification:329 tests pass with four existing dependency warnings; Ruff,
  compileall and whitespace checks pass. Tests cover exact seeded selection,
  distinct source rows,selected-row scaling,80/20 counts,prepared identity and
  checksum failures,unsupported/invalid requests,and unchanged full-data identity.
  Notebook generation and the six-run plan pass. No real FLIGHT training was
  launched locally. User approved committing/pushing after verification.

## 2026-11-13 - FLIGHT-only stitching/rank batch and result inbox

- User requested FLIGHT only for the next batch, one initial seed and ranks0/64
  crossed with none/fixed/learned stitches. Retain existing ablation seed31,
  1024 supports/8 patches,100/20 schedule and early stopping. Existing FLIGHT
  full-data source,80/20 split,preprocessing,batch2048 and other settings remain.
  No subsampling, extra seeds, rank128 or increased stitch count was introduced.
- Separate source-backed notebook reuses the existing worker, matched geometry,
  per-run archives, loud failures and progress controls. Original notebooks and
  model equations remain unchanged. Real GPU training is for the user on Kaggle.
- Added result_archives/ as a ZIP inbox with a tracked INDEX.csv and instructions.
  Existing root ZIPs stay untouched; index distinguishes audited outputs from
  planned experiments. New ZIP contents are excluded from Git.
- Verification:321 tests pass with four existing dependency warnings; Ruff,
  compileall and whitespace checks pass. Added six-condition plan/protocol and
  reproducible output-free notebook checks, exclusive-study validation, and
  rank64 matched-geometry/finite-gradient tests in every stitch mode. Archive
  ignore rules preserve tracked index/guide while excluding new ZIP contents.
  No full FLIGHT training was launched locally. User approved committing and
  pushing the verified notebook and archive index to main.

## 2026-11-13 - Approved shared-kernel GPW study

- User approved12 shared-only runs across California Housing, PROTEIN, TCO and
  OISST with seeds31,32,33,1024 observed-row supports/8 patches,rank32 and learned
  stitches. Data/preprocessing retain seed31 and the existing protocols. Reuse
  matching local-kernel ablation results later, not by automatic import/skip.
- User clarified one lengthscale per input direction across all patches; after
  explaining output variance, explicitly approved sharing that too. Implement
  one learned ARD vector and one learned output variance. Keep one global noise
  SD as before; no parameter tying, freezing or family change for q(U). Means,
  block factors and rank32 V remain independently trainable. Supports/stitches,
  bounds, numerics, warm-up, optimization and full uncertainty are unchanged.
- Preserve default local kernels for all previous launchers. Save effective
  kernel_mode/parameter count, geometry identity and local-reference config per
  new run. A future reuse must check config except kernel_mode, numerical/source
  algebra, data/preprocessing/splits and initial geometry; no local completion
  is inferred. Keep per-run archives and20-epoch labeled logs in fresh workers.
- User approved commit/push after verification. Full suite:317 passed with four
  existing dependency warnings. Ruff, compileall and whitespace checks pass.
  Dense references verify stitched prior, conditional residual, weights, both
  variance terms and ELBO with shared kernels. Gradient checks match the sum of
  local gradients at identical values; q means/block factors/V remain untied,
  trainable and change after updates. Tiny three-seed workers verify matched
  local geometry, saved configs/manifests/archives, summaries and loud failure
  handling. Notebook generation and fresh source workers pass. No real benchmark
  was launched locally; Kaggle GPU execution remains user-run.

## 2026-11-13 - Approved44-run comparison repeats

- User approved seeds32/33 on California Housing, PROTEIN, TCO and OISST,
  keeping the tested model sizes and100/20 schedule/early stopping. Six models
  per seed on California/PROTEIN/TCO, four on OISST:44 runs. Exclude VNNGP/FLIGHT.
  Data splits/preprocessing remain seed31; model initialization/training vary.
- Found that archived ProSpar starts were T=1/c=1, unlike current T=0.1/c=2.
  Asked explicitly; user approved preserving the historical starts for these
  repetitions. Keep all newer general defaults unchanged. Explicit notebook
  ProSpar overrides are saved in effective configs. Do not silently pool starts.
- Separate repeat-study dispatch preserves generic comparison/ablation modes.
  Rebind reduced-data PC-tree references to the same model seed's primary GPW
  initialization. Keep per-run archives, loud failures, fresh workers and20-epoch
  progress labels; save both model/data seeds and original model names in CSV.
  Existing seed31 results are neither rerun nor automatically loaded/pooled.
- User approved commit/push after verification. Full suite:295 passed with four
  existing dependency warnings; Ruff, compileall and whitespace checks pass.
  Tests cover44-run plan/config identity, notebook reproducibility, fresh source
  workers, unchanged prepared split, matched GPW/PC-tree rows for both seeds,
  per-run archives and fail-fast preservation of completed outputs. Independently
  compared all saved seed31 configuration keys against the archive audit: no
  differences after the explicit ProSpar overrides. No real benchmark launched.

## 2026-11-13 - Experiment execution tracking and planned GPW scaling

- User requested completed/not-completed categories, completed seed counts and
  GPW scaling experiments across datasets. Update the existing Excel plan using
  the verified61-run archive audit; every archived configuration uses seed31.
  Distinguish distinct seeds from alternative model sizes and partial model
  coverage. Preserve original plan/paper settings and record saved effective
  run parameters separately rather than silently rewriting historical presets.
- Treat the current72-run ablation batch as reported in progress, not completed
  from a notebook plan. The screenshot establishes one saved OISST seed31,
  rank0/no-stitch condition; full local ablation evidence is not yet available.
- Add12 dataset-specific GPW scaling planning rows, including currently excluded
  FLIGHT. Scaling axes, grids, controls, seeds, hardware and training/timing
  protocols remain UNDECIDED. No model, hyperparameter, preprocessing or runner
  change and no new experiments launched. Verified61 inventory rows,4158 saved
  parameter rows,12 scaling studies and696 explicit scaling parameter fields.
  Reopened the exported XLSX, inspected status/counts/links, scanned formula
  errors and rendered all changed views. Original plan rows, configuration rows,
  reference links and unrelated sheet values were checked unchanged. Whitespace
  validation passes; no model tests needed for this planning-only update.

## 2026-11-13 - Identifiable, sparse ablation progress logs

- User requested dataset/model labels and training tqdm updates every20 epochs.
  Ablation labels also identify seed, rank and stitch mode. Expose the interval
  in notebook settings and save effective progress controls with each run.
- Suppress the extra postfix refresh; use explicit epoch progress updates so
  final counts remain correct on early stopping. Keep all loss histories and
  callbacks per epoch, snapshot schedules and optimization unchanged. Existing
  running Kaggle jobs are unaffected. Other multi-dataset jobs gain labels but
  retain their existing per-epoch refresh interval.
- Focused tests:62 passed, including exact training-state/loss equivalence,
  refresh counts, early-stop display and notebook/worker propagation. Disable
  timed background refreshes too;24 training/control tests pass after this check,
  including unchanged nonzero-learning-rate parameter updates. Full suite:
  284 passed with four existing dependency warnings. Ruff, compileall and
  whitespace checks pass. No real benchmark was started.

## 2026-11-13 - Three-seed default ablation study

- User approved reducing model/training seeds to31,32,33. The four selected
  datasets now yield72 runs (4 x3 x2 ranks x3 stitch modes),7200 maximum epochs.
  Keep data seed31,1024 supports/eight patches, rank0/32, stitch modes,100/20
  schedule, early stopping, geometry matching and output contracts unchanged.
- The original11 datasets remain explicitly selectable:198 runs with three
  seeds, or330 when five seeds are explicitly requested. Regenerate the launcher
  and update count/seed assertions and user-facing documentation. All28 ablation
  tests, Ruff, compileall and whitespace checks pass on CPU; four existing
  dependency warnings remain. No real benchmark is run or existing result modified.

## 2026-11-13 - Four-dataset default ablation selection

- User selected California Housing, PROTEIN, TCO and OISST; TCO was repeated
  in the message and is included once. Default order follows that selection.
  Four datasets x five seeds x two ranks x three stitch modes gives120 runs;
  one selected model seed gives24. Original11 datasets remain explicitly
  selectable for330 runs; no dataset adapter or full-catalog preset is removed.
- Preserve1024 points/eight patches, seeds31-35, ranks0/32, stitch modes,
  data splits/preprocessing,100/20 schedule, stopping, geometry matching and
  output contracts. Only the notebook's default dataset selection changes.
- Regenerate the launcher and test default120-run settings, retained330-run
  selection, notebook reproducibility and isolated workers. All27 ablation tests,
  Ruff, compileall and whitespace checks pass; four existing dependency warnings
  remain. No real datasets or full benchmarks are run for this selection change.

## 2026-11-13 - Approved 330-run GPW stitching/rank notebook

- User explicitly approved1024 inducing points in8 patches of128, seeds31-35,
  V ranks0/32, none/fixed/learned stitches across the11 existing batch datasets.
  Keep the shared-kernel study planned, not implemented/crossed into these runs.
  User approved commit/push after verification. No real benchmarks were launched.
- Preserve existing seed31 data splits, synthetic realizations and preprocessing;
  the five model seeds change initialization and training only. Reuse exact
  selected rows/supports/centers across six conditions; candidate graph/stitches
  also match. Verify the SHA256 geometry identity before training each variant.
- Expose stitch_mode in UKConfig, with learned as the unchanged legacy default.
  Fixed freezes only stitches; none removes constraints, not the GPW model.
  Preserve local learned kernels, q initialization, prior/variance equations,
  rank-dependent q family,100/20 schedule and approved early-stopping settings.
- Add GPW_Ablations_Kaggle.ipynb using latest-main source and fresh workers,
  editable finite dataset/seed selections, effective overrides, explicit training
  gate, verified prepared-array reuse, in-place history updates and immediate
  per-run ZIP links. Every completed run is saved before the next begins;
  failures re-raise with traceback. No resume/skip, reconnect or fallback added.
- Save all seed metrics and descriptive means/sample SDs with completed counts;
  SD for one seed is explicitly undefined. Final error bars are sample SDs, not
  confidence intervals. Verification:271-test full regression suite passes;
  final25 ablation checks and a fresh-worker test pass, including dense moments/
  gradients in all six conditions, matched geometry and mismatch errors,
  fixed/moving/absent stitches, archive/failure behavior, notebook reproducibility,
  invalid/irrelevant controls and two consecutive isolated settings workers.
  Ruff, compileall and whitespace checks pass. Four existing dependency warnings
  remain. CPU-only verification; no full benchmark or Kaggle GPU run performed.

## 2026-11-13 - Planned matched-initialization GPW ablations

- User selected no/fixed/learned stitches, V ranks0/32, five seeds and the same
  inducing-point count, with matched initial locations across variants, for the
  requested datasets. The common budget value and seed IDs remain undecided.
  Match locations within each dataset/seed; different datasets have different
  domains/dimensions. No new initialization or seed-generation rule is adopted.
- User also requested a same-kernel-hyperparameter experiment. Whether this
  means tying learned local kernels or freezing common values remains unresolved,
  as does whether to cross this factor with the stitching/rank study.
- Updated target.txt and TODO.md only as planning records. No model, defaults,
  notebook, data protocol or training code changed; no experiments were run.
  Verification: documentation diff and whitespace review only.

## 2026-11-13 - ProSpar author-code audit and approved weighting initialization

- Located the official journal supplement at Figshare article28765094, code
  file53548667 (prospargp_code.zip); MD5 035199fac400d54a22d4fc8d96a3e4ca,
  SHA256 4967ac9a89ab582b79dbace617116cd4a22b3e2a916d6c6ea0b124c4b86b047e.
  Read models.py and the reproduction notebook without executing author code.
  Mean, variance and ELBO match algebraically under whitening conversion; a
  fixed-parameter numeric check agrees below3e-17. No author GitHub was verified.
- Author fitc code names the shared exponent T_tens[0] and expert temperatures
  Tc_tens; the checked reproduction initializes exponent2 and temperatures0.1.
  User approved only these initial-value changes for future ProSpar presets.
- Update OISST, UKHousing and multi-dataset ProSpar presets to T_j=0.1 and c=2.
  Equal initialization does not tie the separately learned temperatures or
  cancel their effect on the softmax. General model constructors still require
  explicit initialization; existing user overrides and saved runs are unchanged.
- Preserve unwhitened inference, positivity/bounds, nuggets0.01, fixed jitter0,
  supports/initialization, learning rates, batch sizes, budgets,100/20 schedule,
  early stopping and all data choices. This is not full author reproduction.
- Update launcher description and reference tests. All57 focused ProSpar,
  OISST, UKHousing comparison and multi-dataset launcher tests pass, including
  dense algebra/gradients, complete preset coverage, independent temperature
  parameters and non-cancellation of a common temperature. Four dependency
  warnings remain; Ruff, compileall and whitespace checks pass. No real
  benchmarks were run. User explicitly approved commit/push after verification.

## 2026-11-12 - Reduced batch and approved training-loss stopping

- User approved retaining all real datasets except FLIGHT, excluding VNNGP,
  reducing each synthetic function to D5/train50k and keeping both existing
  GPW/ProSpar sizes. Standard Patchwork uses only PC-tree variants. The reduced
  selection has11 datasets/61 model configurations; full presets remain available.
- User explicitly changed the maximum schedule to100 epochs/20 warm-up and
  approved early stopping on epoch training loss: patience10, minimum relative
  significant improvement0.001*abs(best_loss), monitored only after warm-up.
  Epoch21 initializes the reference; ten subsequent non-improving epochs stop
  at31 in the constant-loss case. Tiny improvements accumulate relative to the
  last significant reference. Equal losses, including zero, never reset patience.
  Positive/negative/zero losses use the same explicit rule. Nonfinite loss raises.
- User selected evaluation of the last trained model, not checkpoint restoration.
  Save early-stopping config, actual epochs, reason and evaluated epoch per run;
  force final histories, plots, locations and checkpoint on early exit. Preserve
  per-model archives/summary updates and fail-stop behavior. No external backup
  or implicit resume is added. Fixed-length behavior remains when stopping is None.
- Shared trainer and exact/VNNGP loops honor an explicit True epoch-callback
  return; existing None callbacks are unchanged. No model equations, gradient
  formulas, optimizer values, preprocessing or split choices change.
- Verification:47 targeted tests and all248 regression tests pass, including
  dense-reference model checks, all five trainer stop callbacks, threshold
  handling at positive/negative/zero loss, exact warm-up/patience counts,
  last-state evaluation, forced final snapshots, archive retention and unchanged
  retained configurations. Four dependency warnings remain. Compileall, Ruff
  and whitespace checks pass. No real benchmark or Kaggle GPU run was performed.
  User explicitly approved committing and pushing this verified update to main.

## 2026-11-12 - No-VNNGP batch with FLIGHT last

- User requested all datasets/models except VNNGP, with FLIGHT last, and saving
  completed results before GPU time expires. User approved commit/push after checks.
- Notebook explicitly sets excluded_model_types=[vnngp], keeps all 22 datasets,
  moves FLIGHT after the synthetic designs, and selects 137 model configurations.
  The 3/1 schedule and all retained model/data settings are unchanged. The full
  179-run catalog remains available; exclusions are saved in the signed plan.
  Unknown/duplicate exclusions and conflicting explicit model selections raise.
- Preserve sequential per-model metrics, comparison tables and completed ZIPs
  before the next model. Add immediate archive download links. No silent skip,
  checkpoint resume, external upload or persistent-storage guarantee is added.
  Session-local archives must be downloaded/exported to protect against loss
  of the session filesystem. Use the user's confirmed Kaggle input mount path.
- Verification: all22 launcher tests pass, including exclusion/order/config
  identity, notebook regeneration, fresh-worker compatibility and preservation
  of first-run ZIP/metrics after a deliberate second-model failure. Compileall,
  Ruff and whitespace checks pass. No model equations changed; no real
  benchmarks are launched locally.

## 2026-11-12 - Approved multi-dataset Kaggle comparison

- All remaining choices were explicitly approved: new-data 80/20, seed31,
  all-row input min-max and training-only target scaling; canonical sklearn
  California/UCI CASP/GPML SARCOS sources, author TCO/TCO.L2 files, official
  ICETHICK and trusted author FLIGHT. SARCOS combines supplied files, first torque.
  Preserve OISST/UKHousing and official MODIS protocols. Source bytes were acquired
  and pinned; preparation of six new real datasets and all 12 full-size synthetic
  designs succeeded. Precipitation/satellite drag remain explicitly deferred.
- Approved M512 California; M1024 ozone/ice/protein/sarcos/synthetics; M2048 FLIGHT,
  GPW/ProSpar 8/4 patches (FLIGHT16/8), VNNGP K32/K128, eight-region reduced-data
  PC tree. Retain approved RBF initialization, learning rates, bounds and numerics.
  Existing OISST/MODIS sizes remain unchanged. The UK comparison now uses K128
  instead of K256. ProSpar actual versus paper5000 capacities are saved explicitly.
- Approved all d3/5/10 and training50k/100k configurations for both synthetic
  functions: Latin hypercube, Gaussian noise variance1e-8, Michalewicz m10,
  G a_i=(i-1)/2. Test counts12.5k/25k preserve80/20, unlike paper fixed25k.
- User explicitly approved one-time airline conversion after modern pandas could
  not read pandas.core.index. Isolated Python3.6.8/pandas0.23.4/NumPy1.16.6 read
  the trusted pickle; the existing approved feature/time mapping produced
  5,929,413 rows with zero removals. Original SHA256 f111b35372eb2798cd43ab186203d5bb8733653260faa65c6307c87a933ef083
  is unchanged. Converted NPZ SHA256 90ba687d9b4c7fef8d39cd76572a5a46df4b0232cac9fc0e96664e2c4aff5b3a;
  provenance SHA256 0a2ae27c3d8681ffd572ec714ed28ffbe7213addd349f97e71f3c6f52754c4be.
  Kaggle accepts only this NPZ/provenance, with no runtime unpickling or fallback.
  The private input ZIP also contains official icethk.txt; no data enter Git.
- Launcher prints 22 configurations/179 preset model runs and all effective
  settings, defaults to3epochs/1warm-up, and requires RUN_TRAINING=True. Selection
  is explicit; no budgets/devices/values are reduced automatically. Fresh workers
  save full failures, stop downstream work, and terminate on interruption.
- Save immutable checksummed prepared data, all-dimensional EDA/positions,
  configs, source archives, checkpoints, predictions, histories, metric scales,
  timings, completed-run ZIPs and collection ZIP under Kaggle working outputs.
  Reuse is explicit and verified; training never silently resumes/skips/overwrites.
- Verification: all227 tests pass (four external dependency warnings), including
  dense high-dimensional objective/gradient/prediction references, five-model
  3/1 smoke runs, fresh-worker isolation, immutable prepared-data reuse,
  training authorization, complete archives and failure-stop behavior.
  Compileall, Ruff and whitespace checks pass. The full converted FLIGHT loads
  and splits into4,743,530 train/1,185,883 test rows; input ZIP integrity and
  manual-file checksums pass. Plots were visually inspected.
  No real benchmark training or Kaggle GPU run has been performed in this change.
  User explicitly approved committing and pushing the verified code to main;
  dataset files, the private input ZIP and unrelated local files stay outside Git.

## 2026-11-12 - Multi-dataset foundation and higher-dimensional Patchwork

- User chose 80/20 for new dataset protocols, a 3-epoch/1-warm-up short run,
  paper-style PC-tree Standard Patchwork in higher dimensions, and explicit
  deferral of precipitation and satellite drag because source data are unavailable.
- Extend balanced first-PC trees above two dimensions, retaining every feature.
  Retain all regions and equal-count/positive-variance/projection-tie invariants.
  Compute boundary feasibility and exact region bounds with linear programming;
  uniform draws in free coordinates accepted inside the shared face have constant
  surface Jacobian, giving uniform boundary stitches without MCMC or projection.
  No failed solver, infeasible face or poor acceptance triggers a different method.
  Existing 2D tree generation and its seeded boundary draws are preserved.
- Standard Patchwork now uses one ARD coordinate per actual input dimension.
  Prediction routing accepts the closed unit cube, including normalized maxima;
  internal cell boundaries retain their half-open routing. Training support
  initialization still requires actual interior observations, never moved points.
- Add dimension-independent comparison primitives with explicit split fraction
  and seed; target scaling uses training rows only. Dataset adoption of the input
  transform, presets/sources and synthetic generation still awaits user approval.
  No new real dataset was acquired or assigned an invented configuration.
- Shared runner saves full-dimensional locations, checkpoints, hyperparameter
  histories, configurations, scores and labeled two-feature projection plots.
  ProSpar metadata records actual capacity versus the paper's 5000 points.
- Targeted geometry/dense/runner tests pass; full regression verification running.
  Full multi-dataset acquisition/launcher is not complete or published yet.

## 2026-11-12 - Multi-dataset launcher requested

- Record the user-requested capacity caveat: current matched-budget ProSpar
  has 1024 inducing points (8x128 or 4x256), versus 5000 (200x25) in the supplied
  paper. Add the distinction to README; future notebook/report labels must retain it.
- User selected VNNGP K32/K128 instead of K32/K256 for upcoming comparisons.
  Existing published UKHousing launcher remains unchanged pending the new work.
- User approved the full dataset list and a short-run schedule of 3 epochs with
  1 warm-up epoch, both editable independently. New-dataset preprocessing/splits,
  presets, unavailable sources and unsupported model/likelihood combinations
  remain unresolved; do not invent defaults for them.
- Documentation only at this step; no training or model configuration changed.

## 2026-11-12 - Explicit UKHousing source refresh

- After a Colab source-checksum failure, user approved verifying and repinning
  current official files, without bypassing checks or changing methodology.
- Official listing retains the same price URL. Current CSV has 1,037,647 rows:
  30 added IDs, 49 removed IDs, 63 revised records relative to the old local file.
  Exact 16-field CSV schema, unique IDs, 2018 dates and positive prices verified.
  The postcode archive is byte-identical; ZIP integrity/schema/key checks pass.
- Pin prices to dbf8e95e719630fa8b8d0c90bea52e9e282e7bb6c938951737d4ed65c7655eda;
  retain the postcode hash. Record source_snapshot=2026-11-12 in new manifests.
  Detailed source/history notes: docs/UKHOUSING_SNAPSHOTS.md. This is a locally
  observed HTTPS snapshot hash, not a publisher-authenticated checksum.
- Actual unchanged preparation succeeds with 185,257 retained flats (previously
  185,254). Preserve old local files/results. No preprocessing, split procedure,
  model, optimizer, hyperparameter or numerical-setting change. Snapshot changes
  can change actual split membership, and remain distinguishable by hashes.
- Checksum errors now name expected/actual hashes and the local download/cache
  path. Failures still raise with no replacement download, alternate source or
  automatic repinning. Download mismatches retain their .part file for inspection.
- Two complete HTTPS downloads produce identical price hashes; observed server
  Last-Modified is 2026-10-28. The postcode archive has 1,809,110 unique keys.
- Verification: actual full data preparation succeeds; all 187 tests pass,
  including exact download acceptance, expected/actual error details, preserved
  rejected files, and no replacement/retry for stale caches. Ruff, compileall and
  whitespace checks pass. Four dependency warnings remain; no model training.

## 2026-11-12 - Approved UKHousing size-matched comparison

- User approved RBF kernels and existing 100 epochs / 20 kernel-noise warm-up,
  rather than changing to Matern or the VNNGP paper's sweep. Preserve existing
  UKHousing initialization, learning rates, batch sizes, numerical constraints,
  seed 31 and 64/16/20 split. This is not a reproduction of the paper.
- Compare SVGP 1024; GPW and ProSpar each 8x128 and 4x256; VNNGP at all
  training locations with K32/K256; Standard Patchwork grid 4x2, initial-center
  Voronoi and PC-tree 8, using exactly the eight-patch GPW initial 1024 targets.
  These standard baselines retain the existing UKHousing two-stitches/edge
  setting and use reduced data, while the other models train on all observations.
- Approved ProSpar starts: temperatures 1, exponent 1, nuggets 0.01, extra
  prior_jitter 0. Legacy UKHousing defaults and notebook remain unchanged.
- New launcher uses latest main and fresh workers, immutable prepared split
  files verified by checksum/config identity, explicit selected-model reruns,
  an updating live figure, tqdm and per-completion archives/downloads. No
  automatic resume/skip/recovery. Saved failures re-raise and block dependents.
- Initialize CUDA before resetting allocator statistics, matching the verified
  OISST ordering; no device or numerical substitutions.
- Verification: all 184 tests pass, including all ten tiny synthetic runs,
  prepared-data corruption/config rejection, exact preset budgets/references,
  notebook regeneration/syntax, fresh-worker setup reuse, archived completions
  and loud failures blocking later runs. Ruff and compileall pass. Four existing
  dependency warnings remain. No full UKHousing training or Colab GPU run.

## 2026-11-12 - Balanced PC-tree Standard Patchwork

- User approved an additional 8-patch PC-tree baseline following the paper's
  spatial partition, alongside checkerboard and initial-GPW-center Voronoi.
  All three retain exactly the same selected observations and observed targets.
- Split each region along its first local PC, at the midpoint between the two
  central projections, into equal-count children. Keep the full tree fixed.
  Require a power-of-two patch count and a divisible observation budget; ties
  preventing a geometric equal-count split raise, never round/drop/substitute.
- Use the existing stitch budget per shared final-region boundary, drawn
  uniformly with the declared experiment seed, as in the paper. Preserve
  existing evenly spaced grid/Voronoi stitches and all likelihood/kernel settings.
- Save tree routing alongside polygons, labels, stitches and source row indices.
  Existing UKHousing adaptive/all-data settings remain unchanged.
- Verification: all 177 tests pass, including balanced oblique geometry, seeded
  stitch reproducibility, fixed routing, preserved source rows, invalid-budget
  errors, dense joint-likelihood values/gradients and conditional predictions.
  The seven-configuration runner saves/reanalyzes geometry. A 512-point preview
  confirms eight regions of 64 points. Ruff, compileall and whitespace pass;
  four dependency warnings remain. No real-data training or Colab GPU run.
- Consolidated the local planning workbook into five experiment families, keeping
  per-dataset readiness separate. List checkerboard, GPW-center Voronoi and PC
  tree together wherever Standard Patchwork is proposed. Existing hyperparameter
  cells/overrides and unrelated sheets preserved; visuals and XLSX integrity
  checked. Generated workbook stays outside Git.

## 2026-11-12 - In-place OISST history display

- User requested updating the training-history plot instead of appending it at
  each snapshot. Give history its own persistent display handle, like progress.
  Training, snapshot frequency, saved figures and final-result displays unchanged.
- Remove single-run metric bar-chart displays from training completion and
  saved-result analysis as requested; retain numeric tables and saved artifacts.
- Verification: 13 focused notebook/partial-rerun tests pass; after bar-chart
  removal, all 3 display regression tests pass. Ruff, compileall and whitespace
  checks pass. No model changes or real-data training; full suite not rerun.

## 2026-11-11 - Matched-initial-location Standard Patchwork comparisons

- User approved two OISST baselines using only observed targets at GPW's initial
  inducing locations: a fixed (4 longitude, 2 latitude) checkerboard, and Voronoi
  cells of GPW's initial clustering centers. Boundaries are straight segments
  clipped to [0,1]^2. Empty selected-data cells explicitly raise; never merge or
  drop regions, reduce budgets or replace points. Regions/stitches stay fixed.
- Extract the existing initialization's row indices and centers without changing
  its clustering, sampling, seed or support positions. Save those identities for
  GPW and matched baselines. Effective GPW initialization settings drive both
  baselines, including baseline-only reruns; conflicting references raise.
- Each comparison computes that initialization once under GPW's requested
  thread setting and shares the exact arrays across models. Save its reference
  and timing separately; never silently recluster if a shared reference differs.
- Existing Standard Patchwork name now selects the grid; the additional name
  'Standard Patchwork (GPW regions)' selects initial-cluster Voronoi regions.
  Notebook controls expose grid shape, stitches, epochs and learning rate; the
  reference point budget/cluster count are set in the GPW overrides. Update the
  former full default model selection to include both; retain custom selections.
- Preserve joint-likelihood training, local ARD RBFs, separate regional kernel
  parameters, numerical settings, initialization and optimization. Fixed grids
  use their requested geometry, not the adaptive maximum-points subdivision.
  Legacy UKHousing adaptive/all-training-data settings are unchanged.
- Save training row indices/targets, initial centers/supports, polygon boundaries,
  stitches, region counts and partition figures. Reanalysis regenerates the fixed
  geometry plot from saved arrays and includes those arrays in provenance.
- Shared preprocessing and held-out splits are unchanged. GPW uses all training
  observations while these baselines use only selected targets: this is a
  reduced-data support-location comparison, not an equal-information benchmark.
- Verification: all 169 tests pass, including both partitions' dense joint
  likelihood values/gradients and conditional predictions, exact shared row
  identity, empty-cell failures, saved artifacts and fresh-worker notebook tests.
  Partition figures inspected locally; Ruff, compileall and whitespace checks
  pass. Four dependency warnings remain. No real-data benchmark or Colab GPU
  run launched. GPW's model/weights and VNNGP are unchanged.

## 2026-11-11 - OISST reruns without reconnecting

- User explicitly requested no kernel restart or reconnect. OISST setup now
  verifies an existing checkout's root, remote, branch and clean tracked source,
  rejects untracked Python source, fetches latest main and fast-forwards only.
  No reset, deletion, fallback to old code or module-reload guessing.
- Settings, preparation/reuse, training, saved-result analysis and archives use
  fresh Python workers. The parent notebook imports no experiment/model/Torch
  modules. Existing model equations, settings, data splits and runners are reused.
  Old imports from a previous notebook execution do not affect new workers.
- Each worker verifies the setup commit and source archive checksum, records its
  JSON request, PID, streamed log and full failures. Parent checks exit status,
  streams tqdm and snapshot/completion events, and terminates on interruption.
  Failed stages still block downstream work; retries remain explicit actions.
- Notebook regeneration backs up its original bytes and saved outputs under
  ignored results/notebook_backups, preserving explicit model/data/rerun controls.
  Other launchers retain their existing in-kernel execution policy.
- Verification: all 160 tests pass. Real local Git fixtures verify repeated setup
  with loaded parent modules, latest-source fast-forward, preserved data/results,
  distinct worker processes and new defaults despite stale parent imports.
  Damaged/wrong checkouts and stale source identities fail loudly. A small CPU
  worker training run streams snapshots/completion and reproduces saved metrics
  in a separate analysis worker. Callback acknowledgement preserves synchronous
  snapshot handling. Ruff, compileall and whitespace checks pass; four dependency
  warnings remain. No real-data benchmark or Colab GPU run launched.
- The user additionally requested Standard Patchwork use GPW's initial observed
  points instead of all targets. Its grid dimensions and the meaning of the RMSE
  trend are pending clarification; this execution-only change does not alter it.

## 2026-11-11 - Standard Patchwork paper training objective

- User requested the paper's training after review identified conditional versus
  joint likelihood. Implement equation (8): -log p(Y, Delta_S=0), retaining the
  existing per-observation scaling in loss(). Remove the stitch-marginal
  normalization subtraction and include the Gaussian stitch dimension constant.
- Predictions remain conditional on the same observations and stitch differences.
  Fixed jitter, bounds, initial values, Adam settings and partition are unchanged.
  User explicitly chose to retain separate regional kernel parameters and change
  only the likelihood, rather than reproduce the real-data shared-kernel setting.
- User requested a predefined regular geometric partition for future work;
  do not invent a patch count, grid aspect ratio or empty-cell policy.
- Verification: dense joint objective values/gradients and conditional predictions
  pass with stitches and in the single-region limit. Full suite: 152 passed,
  one failed because the Colab-saved OISST notebook already contains execution
  outputs, violating the existing output-free launcher check. Confirmed those
  outputs exist in HEAD; did not delete user content or weaken the test. Ruff,
  compileall and whitespace checks pass. No benchmark training or GPU run.

## 2026-11-11 - Agreement weights independent of the variational posterior

- User explicitly selected Cov(local functions | U, Delta_S=0) for agreement:
  inducing locations, local kernels and stitches define the weights before q(U).
  This supersedes the earlier posterior-adaptive A S A^T + R weight rule.
- Production fusion, regression maps and prediction now use the full stitched
  conditional residual covariance for weights. Retain the existing explicit
  weight_epsilon, prior jitter, initialization, optimization and noise settings.
- Predictive variance still includes w^T S w + alpha^T R alpha; observation
  noise is added once. q(U) affects predictions, not weights. The expected
  Gaussian log likelihood minus KL is now the fixed-model Gaussian ELBO.
- Keep historical experiment records unchanged. Update active runner provenance,
  launcher descriptions and documentation to distinguish the new construction.
- Verification: all 152 tests pass, including independent dense residual
  conditioning, weighted prediction/objective references at q ranks 0/3/6,
  q-independence, retained support/stitch/kernel gradients and saved provenance.
  Notebook generators match saved launchers; notebook execution settings are
  unchanged. Ruff, compileall and whitespace checks pass. Four third-party
  deprecation warnings remain; no benchmark training or Colab GPU run performed.

## 2026-11-11 - Explicit OISST partial reruns

- User requested all three workflows: retry a failed stage, retrain selected
  models while keeping data/results, and regenerate metrics/plots without training.
- OISST notebook now separates setup, settings, data, training, analysis and
  downloads. Stage-local failure states block dependent work but permit explicit
  retry; no global exception hook or automatic recovery. Legacy notebook guards
  remain unchanged. Settings changes invalidate training but retain prepared data;
  reuse validates the data-relevant configuration before any model is built.
- Added checksummed prepared arrays/splits and explicit prepare/load actions.
  Changed data settings, missing files or checksum failures raise; never rebuild,
  impute or substitute data. Position-saving changes do not change the split.
- Selected-model reruns create fresh batch directories and train from scratch,
  preserving failed batches and earlier completed models. No checkpoint resumption,
  automatic skipping, overwrite, budget/seed changes or model-equation changes.
- Reanalysis explicitly reads completed model-run predictions, recomputes existing
  Gaussian metrics, regenerates figures in a new directory and records input
  checksums. No model construction/training/prediction/checkpoint loading; original
  metrics and artifacts remain untouched. Latest failed batches are rejected;
  choosing earlier completed run paths is a separate explicit user action.
- Notebook rebuilding preserves user settings, model selections and overrides.
- Verification: all 150 tests pass, including real small CPU preparation/train/
  reanalysis round trip reproducing the original scores, explicit retries, data
  identity/checksum rejection, failed-stage blocking, unchanged original artifacts,
  archive exclusivity and notebook override preservation. Two focused failure-
  record checks also pass after ensuring rejected requests retain their inputs.
  Ruff, compileall and whitespace checks pass; four dependency warnings remain.
  Actual Colab GPU execution remains unverified locally.

## 2026-11-11 - Standard Patchwork short-run warm-up validation

- The saved OISST notebook requests two epochs for Standard Patchwork without
  overriding its unused warmup_epochs=20 field. Generic validation incorrectly
  rejected that configuration, despite exact Patchwork having no warm-up phase.
- OISST and UKHousing now validate warm-up only for variational models. Positive
  epochs and snapshot intervals remain required. No configured value is changed,
  clamped or replaced. OISST validation errors now name the model and values.
- OISST exact-baseline optimizer provenance now records its actual full-data
  Adam rate and warmup=not_used rather than an unused variational configuration.
- Verification: full 132-test suite passes after the validation fix. Short-run
  end-to-end tests retain the ignored value of 20; invalid active warm-up,
  epochs and snapshot controls still raise without changing requested values.
- At the user's request, added explicit notebook MODELS selection/order,
  independent of OVERRIDES. Unknown/duplicate/empty selections raise. Known
  deselected-model overrides can remain for later use; deselected models are
  never initialized. Preserved the saved two-epoch settings and all defaults.
- Model-selection and notebook checks: 14 focused tests pass, including seven
  new selection tests; Ruff, compileall and whitespace checks pass. No full
  Colab GPU execution is claimed.

## 2026-11-11 - OISST cold-CUDA memory-statistics initialization

- Colab reported Invalid device argument from reset_peak_memory_stats before
  the first model initialized CUDA. PyTorch's reset function does not call lazy
  initialization. Added synchronize(requested_device) immediately before reset,
  matching the shared trainer, which initializes CUDA and validates that device.
- No model, data, hyperparameter, numerical regularizer or device changes.
  CUDA initialization errors still propagate, save tracebacks and stop the run;
  there is no CPU substitution, retry or skipped memory measurement.
- Verification: 126 tests pass, including mocked cold-start/requested-device
  ordering and loud initialization failure. Ruff, compileall and whitespace
  checks pass. Actual Colab GPU execution remains unverified locally.
- Preserved the newer Colab save on GitHub, including the user's two-epoch /
  one-warm-up overrides, by merging rather than overwriting it. Notebook tests
  compare executable structure while allowing explicit override values and
  Colab JSON formatting; defaults and model code are unchanged.

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

