"""Exact finite-dimensional exponential-GP sampling for the approved toy study."""

import numpy as np
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

from pwgp.data.types import RegressionDataset


def exponential_covariance(x, *, variance, lengthscale):
    """Isotropic exponential kernel; no squared distance or added jitter."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Exponential covariance requires finite nonempty x[N,D]")
    if not np.isfinite([variance, lengthscale]).all() or min(variance, lengthscale) <= 0:
        raise ValueError("Exponential variance and lengthscale must be finite and positive")
    covariance = cdist(x, x)
    covariance *= -1 / lengthscale
    np.exp(covariance, out=covariance)
    covariance *= variance
    return covariance


def sample_exponential_gp(*, points, extent, variance, lengthscale, noise_variance, seed, threads):
    """Sample one joint 2D latent field and independent Gaussian observation noise."""
    if type(points) is not int or points < 2 or type(seed) is not int or seed < 0:
        raise ValueError("Require integer points>=2 and nonnegative integer seed")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    if not np.isfinite([extent, noise_variance]).all() or extent <= 0 or noise_variance < 0:
        raise ValueError("Require finite positive extent and nonnegative noise variance")
    rng = np.random.default_rng(seed)
    with tqdm(total=3, desc="sample:exponential-GP", unit="stage") as progress:
        x = extent * rng.random((points, 2))
        if len(np.unique(x, axis=0)) != points:
            raise ValueError("Sampled duplicate GP locations; no jitter or location replacement")
        progress.update()
        with threadpool_limits(limits=threads):
            covariance = exponential_covariance(x, variance=variance, lengthscale=lengthscale)
            lower = cholesky(covariance, lower=True, overwrite_a=True, check_finite=True)
            del covariance
            progress.update()
            latent = lower @ rng.standard_normal(points)
        y = latent + np.sqrt(noise_variance) * rng.standard_normal(points)
        if not np.isfinite(latent).all() or not np.isfinite(y).all():
            raise FloatingPointError("Nonfinite exact exponential-GP sample")
        progress.update()
    dataset = RegressionDataset("exponential_gp_2d", x, y, ("x1", "x2"), "noisy GP response",
        metadata={"generator": "exact joint zero-mean exponential GP",
            "covariance": "variance*exp(-Euclidean_distance/lengthscale)", "points": points,
            "extent": extent, "variance": variance, "lengthscale": lengthscale,
            "noise_variance": noise_variance, "seed": seed, "sampling_jitter": 0.,
            "paper": "https://jmlr.org/papers/volume19/17-042/17-042.pdf#page=13",
            "replication_status": "similar approved design; not original R RandomField realization"})
    return dataset, latent
