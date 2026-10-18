# Generalized Patchwork Gaussian Processes

This repository contains the reusable algorithmic foundation for the research
project described in *Structured Gaussian Processes for Scalable Regression*.
It deliberately separates model code from experiment code.

Coding agents must read [AGENTS.md](AGENTS.md), [TODO.md](TODO.md), and
[docs/EVOLUTION.md](docs/EVOLUTION.md) before making changes. They record the
repository rules, planned work, and technical history.

The current foundation provides:

- a full-covariance stochastic variational Gaussian process (SVGP);
- a generalized Patchwork GP with local ARD kernels, trainable supports,
  trainable stitching locations, exact Nested-Kriging agreement weights, and a
  block-plus-low-rank variational precision;
- shared minibatch ELBO training and batched prediction;
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

Both models expose `elbo_batch`, `latent`, and `predict` with the same input
convention. Inputs and targets should be normalized before fitting.

```python
import torch

from pwgp import SVGP, fit
from pwgp.config import OptimizerConfig
from pwgp.initialization import kmeans_supports

X = torch.rand(500, 2, dtype=torch.float64)
y = torch.sin(8 * X[:, 0]) + 0.1 * torch.randn(500, dtype=torch.float64)
Z = kmeans_supports(X, count=32, seed=7)

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

## Dataset manager

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

OISST is downloaded from NOAA NCEI. FLIGHT uses the 2009 Data Expo files.
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

## References

- Hensman, Fusi, and Lawrence (2013), *Gaussian Processes for Big Data*.
- Park and Apley (2018), *Patchwork Kriging for Large-scale Gaussian Process Regression*.
- NOAA 1/4-degree Daily Optimum Interpolation Sea Surface Temperature v2.1.

