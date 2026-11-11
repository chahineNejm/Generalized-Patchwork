# Agent Guide

Read this file, `TODO.md`, and `docs/EVOLUTION.md` before changing code.

## Purpose

Build a reproducible research codebase for scalable Gaussian-process regression.
The core models are full-covariance SVGP and generalized Patchwork GP with local
ARD kernels, stitching constraints, Nested-Kriging fusion, and structured
variational precision. The current phase is the algorithm and data foundation.

## Non-Negotiable Rules

- Fail loudly. Do not add fallbacks, silent substitutions, parser guessing,
  model downgrades, pseudoinverses, alternate URLs, or implicit value changes.
- Do not use `try/except` to continue after an error. Let the original error
  surface. Validation errors should name the failed invariant.
- Never switch to a base model or change requested values to make a run succeed.
- Runner exceptions must save a full traceback and re-raise; never install a
  process-global exception hook. Colab stages must block downstream execution
  after failure, and subprocess commands must check exit status.
- Validate out-of-range values. Do not clamp or project requested inputs silently.
- Nonfinite model parameters/predictions must raise. Unusable dataset rows are
  removed, never filled, with counts recorded. FLIGHT defaults to the trusted
  author-prepared airline pickle, not raw reconstruction; an unavailable source
  raises without substitution. Do not claim a verified original 2013 SVGP split.
- Keep code compact. Use brief docstrings for functions and comments only for
  mathematically dense blocks.
- Put tunable values in explicit arguments or configuration dataclasses.
- Every training loop displays `tqdm` progress with the current loss.
- Dataset files, checkpoints, and generated results stay outside Git.
- Preserve complete configuration and provenance with every saved result.

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
  This is a paper-derived implementation, not verified author code.
- Patchwork uses `q(U) = N(m, (D + VV^T)^-1)` with block-diagonal `D`.
- Stitching conditions exactly on `Delta_S = 0` using one coherent regularized
  prior. Do not add independent jitter to the Schur complement.
- Posterior-adaptive agreement weights use A S A^T + R and sum to one.
  Singular systems raise an error.
- Patchwork latent variance is w^T S w + alpha^T R alpha, with the full stitched
  local conditional residual covariance R. Observation noise is added only once.
- Weights depend on q through S; training uses the Gaussian expected-likelihood
  minus KL objective with gradients through weights. Do not claim a fixed-model
  ELBO guarantee for this posterior-adaptive objective.
- Algebra changes require small dense-reference tests.
- Use `pwgp.linalg.cholesky_solve` for two explicit triangular solves; do not
  restore the native batched POTRS call that fails on some Colab CUDA builds.
- Standard Patchwork baseline conditions directly on observed noisy targets in
  fixed rectangular regions and boundary stitch differences, without variational
  inducing variables. Its joint conditional likelihood must match a dense reference.
- UKHousing is a verified 2018-flats reconstruction, not a verified VNNGP snapshot.
  Preserve this distinction, pinned raw checksums and training-only target scaling.
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

