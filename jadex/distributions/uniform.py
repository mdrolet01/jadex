from typing import Any

import jax
import jax.numpy as jnp
from chex import PRNGKey, Shape
from flax import struct

from jadex.distributions.base_distribution import BaseDistribution, DistParams, Sample


@struct.dataclass
class UniformSample(Sample):
    """A sample from a continuous Uniform distribution."""

    @classmethod
    def create(cls, leading_shape: Shape, dim_shape: Shape, init_val: float = 0.0):
        """Create a new sample with given shape."""
        value = jnp.ones((*leading_shape, *dim_shape)) * init_val
        return cls(value=value)


@struct.dataclass
class UniformParams(DistParams):
    """Parameters for a continuous Uniform distribution.

    Attributes:
        low: Lower bound(s)
        high: Upper bound(s)
    """

    low: jnp.ndarray
    high: jnp.ndarray

    @property
    def range(self):
        return self.high - self.low


@struct.dataclass
class Uniform(BaseDistribution):
    """A continuous Uniform distribution U(low, high)."""

    @property
    def dtype(self):
        return jnp.float32

    @property
    def param_dtype(self):
        return jnp.float32

    # Core distribution operations
    def sample(self, params: UniformParams, leading_shape: Shape, rng_key: PRNGKey):
        """Draw samples from the uniform distribution."""
        self._check_params(params)
        eps = jax.random.uniform(rng_key, leading_shape + params.low.shape, dtype=self.dtype)
        samples = params.low + eps * params.range
        return self.package_sample(samples)

    def log_prob(self, params: UniformParams, sample: UniformSample):
        """Compute log probability of uniform samples."""
        self._check_params(params)
        self._check_sample(sample)

        inside = jnp.logical_and(sample.value >= params.low, sample.value <= params.high)
        log_probs = -jnp.log(params.range)
        log_probs = jnp.where(inside, log_probs, -jnp.inf)

        return jnp.sum(log_probs, axis=self.param_dims)

    def kl(self, p_params: UniformParams, q_params: UniformParams):
        """KL divergence is infinite unless q covers p entirely."""
        self._check_params(p_params)
        self._check_params(q_params)

        covers = jnp.logical_and(q_params.low <= p_params.low, q_params.high >= p_params.high)
        kl_val = jnp.log(q_params.range) - jnp.log(p_params.range)
        return jnp.where(covers, kl_val, jnp.inf)

    def entropy(self, params: UniformParams):
        """Compute differential entropy of the uniform distribution."""
        self._check_params(params)
        return jnp.sum(jnp.log(params.range), axis=self.param_dims)

    # Expected values
    def get_expected_value(self, params: UniformParams):
        self._check_params(params)
        return (params.low + params.high) / 2.0

    def get_prior_params(self) -> UniformParams:
        low = jnp.zeros(self.shape)
        high = jnp.ones(self.shape)
        return UniformParams(low, high)

    def sample_from_prior(self, rng_key: PRNGKey, leading_shape: Shape = ()):
        prior_params = self.get_prior_params()
        return self.sample(prior_params, leading_shape, rng_key)

    # Sample creation and packaging
    def create_sample(self, leading_shape: Shape, init_val: float = 0.0):
        return UniformSample.create(leading_shape, self.shape, init_val)

    def package_sample(self, value: jnp.ndarray):
        self._check_value(value)
        return UniformSample(value)

    def package_params(self, low: jnp.ndarray, high: jnp.ndarray):
        self._check_param_value(low)
        self._check_param_value(high)
        return UniformParams(low, high)

    def _check_params(self, params: Any):
        assert isinstance(params, UniformParams)
        self._check_param_value(params.low)
        self._check_param_value(params.high)
        # assert jnp.all(params.high > params.low), "Uniform: high must be greater than low."

    def _check_sample(self, sample: Any):
        assert isinstance(sample, UniformSample)
        self._check_value(sample.value)
