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
- Validate out-of-range values. Do not clamp or project requested inputs silently.
- Keep code compact. Use brief docstrings for functions and comments only for
  mathematically dense blocks.
- Put tunable values in explicit arguments or configuration dataclasses.
- Every training loop displays `tqdm` progress with the current loss.
- Dataset files, checkpoints, and generated results stay outside Git.
- Preserve complete configuration and provenance with every saved result.

## Mathematical Invariants

- Trainable support and stitch points receive inputs normalized to `[0,1]^D`.
- Kernels are ARD RBF. SVGP has one global ARD vector. Patchwork has one ARD
  vector and output scale per patch, plus one global observation noise.
- SVGP uses `q(U) = N(m, LL^T)` and includes conditional residual variance.
- Patchwork uses `q(U) = N(m, (D + VV^T)^-1)` with block-diagonal `D`.
- Stitching conditions exactly on `Delta_S = 0` using one coherent regularized
  prior. Do not add independent jitter to the Schur complement.
- Nested-Kriging weights come from local conditional covariance and sum to one.
  Singular systems raise an error.
- The current global Patchwork representation has `R_X = 0`. Do not add local
  residual variance to its predictive variance.
- Algebra changes require small dense-reference tests.

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

