# Evolution Log

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

