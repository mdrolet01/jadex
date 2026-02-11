from typing import Any

import jax
import jax.numpy as jnp
from chex import PRNGKey, Shape
from flax import struct

from jadex.distributions.base_distribution import BaseDistribution, DistParams, Sample


@struct.dataclass
class DiagonalGaussianSample(Sample):
    """A sample from a diagonal Gaussian distribution."""

    @classmethod
    def create(cls, leading_shape: Shape, dim_shape: Shape, init_val: int):
        """Create a new sample with given shape."""
        value = jnp.ones((*leading_shape, *dim_shape)) * init_val
        return cls(value=value)


@struct.dataclass
class DiagonalGaussianParams(DistParams):
    """Parameters for a diagonal Gaussian distribution.

    Attributes:
        mean: Mean values of the Gaussian distribution
        variance: Variance values of the Gaussian distribution
    """

    mean: jnp.ndarray
    variance: jnp.ndarray


@struct.dataclass
class DiagonalGaussian(BaseDistribution):
    """A diagonal Gaussian (normal) distribution.

    This class implements a multivariate Gaussian distribution with diagonal covariance matrix.
    It provides functionality for sampling from the distribution, computing log probabilities,
    KL divergence, and entropy.
    """

    @property
    def dtype(self):
        return jnp.float32

    @property
    def param_dtype(self):
        return jnp.float32

    # Core distribution operations
    def sample(self, params: DiagonalGaussianParams, leading_shape: Shape, rng_key: PRNGKey):
        """Draw samples from the diagonal Gaussian distribution.

        Args:
            params: Gaussian parameters defining mean and variances
            num_samples: Number of independent samples to draw
            rng_key: Random number generator key

        Returns:
            samples: Real-valued samples from the Gaussian distribution
        """
        self._check_params(params)

        stds = jnp.sqrt(params.variance)
        eps = jax.random.normal(rng_key, leading_shape + params.mean.shape)
        return self.package_sample(params.mean + eps * stds)

    def log_prob(self, params: DiagonalGaussianParams, sample: DiagonalGaussianSample):
        """Compute log probability density of Gaussian samples.

        Args:
            params: Gaussian parameters defining mean and variances
            sample: Real-valued samples to evaluate

        Returns:
            log_probs: Log probability density of each sample under the Gaussian distribution
        """
        self._check_params(params)
        self._check_sample(sample)

        log_normalizer = jnp.log(2 * jnp.pi * params.variance)
        quadratic = jnp.square(sample.value - params.mean) / params.variance
        log_probs = (log_normalizer + quadratic) / -2.0

        return jnp.sum(log_probs, axis=self.param_dims)

    def kl(self, p_params: DiagonalGaussianParams, q_params: DiagonalGaussianParams):
        """Compute KL divergence between two diagonal Gaussian distributions.

        Args:
            p_params: Parameters for distribution P
            q_params: Parameters for distribution Q

        Returns:
            kl: KL(P||Q) for each batch element
        """
        self._check_params(p_params)
        self._check_params(q_params)

        kls = (
            0.5 * (jnp.log(q_params.variance) - jnp.log(p_params.variance))
            + (p_params.variance + jnp.square(p_params.mean - q_params.mean)) / (2 * q_params.variance)
            - 0.5
        )
        return jnp.sum(kls, axis=self.param_dims).clip(0.0)

    def entropy(self, params: DiagonalGaussianParams):
        """Compute entropy of the diagonal Gaussian distribution.

        Args:
            params: Gaussian parameters defining mean and variances

        Returns:
            entropy: Differential entropy of each Gaussian distribution
        """
        self._check_params(params)

        entropy = 0.5 + 0.5 * jnp.log(2 * jnp.pi * params.variance)
        return jnp.sum(entropy, axis=self.param_dims)

    # Reparameterization
    def reparameterize(self, params: DiagonalGaussianParams, rng_key: PRNGKey):
        """Reparameterize samples from the diagonal Gaussian distribution.

        Args:
            params: Gaussian parameters defining mean and variances
            rng_key: Random number generator key

        Returns:
            samples: Reparameterized samples
        """
        self._check_params(params)

        stds = jnp.sqrt(params.variance)
        eps = jax.random.normal(rng_key, params.mean.shape)
        samples = params.mean + eps * stds

        return self.package_sample(samples)

    # Expected values and regularization
    def get_expected_value(self, params: DiagonalGaussianParams):
        """Get expected values (mean) of the Gaussian distribution."""
        self._check_params(params)
        return params.mean

    def get_prior_params(self) -> DiagonalGaussianParams:
        """Get default parameters with zero mean and unit variance."""
        mean = jnp.zeros(self.shape)
        variance = jnp.ones(self.shape)
        return DiagonalGaussianParams(mean, variance)

    def sample_from_prior(self, rng_key: PRNGKey, leading_shape: Shape = ()):
        """Sample from a uniform categorical distribution."""
        mean = jnp.zeros(self.shape)
        logvar = jnp.zeros(self.shape)
        prior_params = DiagonalGaussian.package_params(self, mean, logvar)
        return DiagonalGaussian.sample(self, prior_params, leading_shape, rng_key)

    # Sample creation and packaging
    def create_sample(self, leading_shape: Shape, init_val: int = 0):
        """Create an initialized sample."""
        return DiagonalGaussianSample.create(leading_shape, self.shape, init_val)

    def package_sample(self, value: jnp.ndarray) -> DiagonalGaussianSample:
        """Package value into Gaussian sample."""
        self._check_value(value)
        return DiagonalGaussianSample(value)

    def package_params(self, mean: jnp.ndarray, logvar: jnp.ndarray) -> DiagonalGaussianParams:
        """Package mean and log variances into Gaussian parameters."""
        self._check_param_value(mean)
        self._check_param_value(logvar)

        min_x_var = self.cfg.get("min_x_var", jnp.finfo(logvar.dtype).eps)
        max_x_var = self.cfg.get("max_x_var", jnp.inf)
        clipped_logvar = jnp.clip(logvar, jnp.log(min_x_var), jnp.log(max_x_var))
        variance = jnp.exp(clipped_logvar)
        return DiagonalGaussianParams(mean, variance)

    def _check_params(self, params: Any):
        assert isinstance(params, DiagonalGaussianParams), "params must be DiagonalGaussianParams"
        self._check_param_value(params.mean)
        self._check_param_value(params.variance)

    def _check_sample(self, sample: Any):
        assert isinstance(sample, DiagonalGaussianSample), "params must be DiagonalGaussianParams"
        self._check_value(sample.value)


class DiagonalGaussianConstantVariance(DiagonalGaussian):
    """A diagonal Gaussian distribution with constant variance.

    This class implements a Gaussian where the variance is fixed and not learned.
    """

    @property
    def x_var_prior(self) -> jnp.ndarray:
        """Get the fixed prior variance values."""
        if self.cfg.get("x_var_prior", None) is None:
            return jnp.array(jnp.nan)
        else:
            return jnp.array(self.cfg.x_var_prior)

    def package_params(self, mean: jnp.ndarray) -> DiagonalGaussianParams:
        """Package mean with fixed variance into Gaussian parameters."""
        self._check_param_value(mean)

        _, x_vars = jnp.broadcast_arrays(mean, self.x_var_prior)
        return DiagonalGaussianParams(mean, x_vars)
