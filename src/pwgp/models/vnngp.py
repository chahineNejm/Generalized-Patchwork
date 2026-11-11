"""GPyTorch VNNGP with explicitly selected exact CPU FAISS neighbor search."""

from contextlib import ExitStack

import faiss
import gpytorch
import numpy as np
import torch
from tqdm.auto import tqdm

from pwgp.config import DEFAULT_HYPERPARAMETER_BOUNDS
from pwgp.linalg import cholesky_solve


def strict_gpytorch():
    stack = ExitStack()
    stack.enter_context(gpytorch.settings.cholesky_max_tries(1))
    stack.enter_context(gpytorch.settings.cholesky_jitter(float_value=0.0, double_value=0.0))
    return stack


class CPUFaissNeighbors(torch.nn.Module):
    def __init__(self, x, k):
        super().__init__()
        self.k = k
        self.values = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        self.index = faiss.IndexFlatL2(x.shape[-1])
        self.index.add(self.values)

    def find_nn_idx(self, x, k=None):
        k = self.k if k is None else k
        values = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Only unbatched two-dimensional neighbor input is supported")
        indices = self.index.search(values, k)[1]
        if (indices < 0).any():
            raise ValueError("FAISS returned missing neighbors")
        return torch.as_tensor(indices, dtype=torch.long, device=x.device)

    def sequential(self):
        index = faiss.IndexFlatL2(self.values.shape[-1])
        index.add(self.values[:self.k])
        indices = np.empty((len(self.values) - self.k, self.k), dtype=np.int64)
        for i in tqdm(range(self.k, len(self.values)), desc="VNNGP exact CPU neighbor graph", unit="point"):
            indices[i - self.k] = index.search(self.values[i:i + 1], self.k)[1][0]
            index.add(self.values[i:i + 1])
        return torch.from_numpy(indices)


class ExplicitNNStrategy(gpytorch.variational.NNVariationalStrategy):
    def _compute_nn(self):
        self.nn_util = CPUFaissNeighbors(self.inducing_points, self.k)
        self.register_buffer("nn_xinduce_idx", self.nn_util.sequential())
        return self

    def _stochastic_kl_helper(self, kl_indices):
        """Evaluate the upstream mean-field KL using SPD triangular solves."""
        if self._batch_shape:
            raise ValueError("VNNGP KL adapter requires an unbatched inducing distribution")
        neighbors = self.nn_xinduce_idx[kl_indices - self.k].to(self.inducing_points.device)
        inputs = torch.cat((self.inducing_points[neighbors],
                            self.inducing_points[kl_indices, None]), dim=1)
        prior = self.model.forward(inputs)
        covariance, prior_mean = prior.covariance_matrix, prior.mean
        # Preserve the pinned upstream jitter arithmetic, including its default dtype.
        identity = torch.eye(self.k, device=covariance.device)
        factor = torch.linalg.cholesky(covariance[:, :self.k, :self.k] + self.jitter_val * identity)
        cross = covariance[:, :self.k, self.k:]
        weights = cholesky_solve(cross, factor).squeeze(-1)
        residual = covariance[:, self.k, self.k] - (weights * cross.squeeze(-1)).sum(-1)
        residual = residual + self.jitter_val
        if not bool(torch.isfinite(residual).all()) or bool((residual <= 0).any()):
            raise FloatingPointError("Invalid VNNGP KL conditional variance")
        q = self._variational_distribution
        variance = q._variational_stddev.square()
        mean_error = q.variational_mean[kl_indices] - prior_mean[:, self.k]
        mean_error = mean_error - (weights * (
            q.variational_mean[neighbors] - prior_mean[:, :self.k])).sum(-1)
        trace = variance[kl_indices] + (weights.square() * variance[neighbors]).sum(-1)
        return 0.5 * (residual.log() - variance[kl_indices].log() - 1
                      + (trace + mean_error.square()) / residual).sum()


class VNNGP(gpytorch.models.ApproximateGP):
    def __init__(self, train_x, *, neighbors=32, batch_size=512, lengthscale=0.1,
                 outputscale=1.0, noise_std=0.1, jitter=1e-4):
        for name, value in (("train_x", train_x), ("lengthscale", lengthscale),
                            ("outputscale", outputscale), ("noise_std", noise_std)):
            if not bool(torch.isfinite(torch.as_tensor(value)).all()):
                raise ValueError(f"VNNGP {name} must be finite")
        if noise_std <= 0:
            raise ValueError("Initial noise_std must be strictly positive")
        cpu_x = train_x.detach().cpu()
        distribution = gpytorch.variational.MeanFieldVariationalDistribution(len(train_x)).to(cpu_x)
        strategy = ExplicitNNStrategy(self, cpu_x, distribution, k=neighbors,
                                      training_batch_size=batch_size, jitter_val=jitter)
        super().__init__(strategy)
        bounds = DEFAULT_HYPERPARAMETER_BOUNDS
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1],
                                      lengthscale_constraint=gpytorch.constraints.Interval(*bounds.lengthscale)),
            outputscale_constraint=gpytorch.constraints.Interval(*bounds.outputscale),
        ).to(train_x)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(bounds.noise_std[0]**2,
                                                           bounds.noise_std[1]**2),
        ).to(train_x)
        self.covar_module.base_kernel.lengthscale = lengthscale
        self.covar_module.outputscale = outputscale
        self.likelihood.noise = noise_std**2
        self.to(train_x)
        strategy._set_training_iterator()

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

    def __call__(self, x, **kwargs):
        result = self.variational_strategy(x=x, **kwargs)
        variance = result.lazy_covariance_matrix.diagonal(dim1=-1, dim2=-2)
        if not bool(torch.isfinite(result.mean).all() & torch.isfinite(variance).all()):
            raise FloatingPointError("VNNGP output mean and variance must be finite")
        return result

    @torch.no_grad()
    def predict(self, x, *, observation_noise=True):
        self.eval()
        strategy = self.variational_strategy
        indices = strategy.nn_util.find_nn_idx(x.float())
        neighbors = strategy.inducing_points[indices]
        inputs = torch.cat((neighbors, x[:, None]), dim=1)
        covariance = self.covar_module(inputs).to_dense()
        k = strategy.k
        factor = torch.linalg.cholesky(covariance[:, :k, :k] + strategy.jitter_val * torch.eye(
            k, dtype=x.dtype, device=x.device))
        cross = covariance[:, :k, k:]
        weights = cholesky_solve(cross, factor).squeeze(-1)
        q = strategy._variational_distribution
        mean = (weights * q.variational_mean[indices]).sum(-1)
        residual = covariance[:, k, k] - (weights * cross.squeeze(-1)).sum(-1)
        variance = residual + (weights.square() * q._variational_stddev[indices].square()).sum(-1)
        if bool((residual < 0).any()):
            raise FloatingPointError("Negative VNNGP conditional residual")
        if not bool(torch.isfinite(variance).all()) or bool((variance <= 0).any()):
            raise FloatingPointError("Invalid VNNGP latent predictive variance")
        if observation_noise:
            variance = variance + self.likelihood.noise.squeeze()
        if not bool(torch.isfinite(mean).all() & torch.isfinite(variance).all()):
            raise FloatingPointError("VNNGP predictive mean and variance must be finite")
        return mean, variance
