# Development Roadmap

Status: `[ ]` planned, `[~]` in progress, `[x]` implemented and verified.

## Foundation

- [x] Working SVGP model class with explicit hyperparameters.
- [x] Working generalized Patchwork model class with explicit hyperparameters.
- [x] Shared training API with visible `tqdm` progress.
- [x] Dataset acquisition, preparation, cache, checksum, and provenance system.
- [x] Keep datasets, results, and checkpoints out of Git.
- [x] Dedicated model package in `src/pwgp/models/`.
- [x] Dedicated experiment directory in `experiments/`.
- [x] Fail-loudly policy with no fallback or silent value changes.

## Model Evaluation

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

- [ ] Install and environment validation cell.
- [ ] Dataset acquisition and integrity-check cell.
- [ ] Unit/algebra test cell with visible output.
- [ ] Single-model training call with inline hyperparameters.
- [ ] Model comparison call with inline hyperparameters.
- [ ] Point-dynamics visualization call.

## User Preferences

- Supply hyperparameters where a model or test is called.
- Always show training progress.
- Prefer explicit errors to automatic recovery.
- Keep functions and comments brief.
- Never silently modify a model, dataset, or requested configuration.

