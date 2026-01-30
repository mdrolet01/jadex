from typing import Any

import jax
import jax.numpy as jnp
import optax
from chex import PRNGKey, Shape
from flax import struct
from jax.typing import ArrayLike

from jadex.distributions.base_distribution import BaseDistribution, DistParams, Sample
from jadex.distributions.dist_utils import logits_to_probs, probs_to_logits


@struct.dataclass
class BernoulliSample(Sample):
    """A sample from a Bernoulli distribution."""

    @classmethod
    def create(cls, leading_shape: Shape, dim_shape: Shape, init_val: int):
        """Create a new sample with given shape."""
        value = jnp.full((*leading_shape, *dim_shape), init_val, dtype=int)
        return cls(value=value)


@struct.dataclass
class BernoulliParams(DistParams):
    """Parameters for a Bernoulli distribution."""

    _unnormalized_logits: jnp.ndarray

    @property
    def logits(self) -> jnp.ndarray:
        """Get logits from raw logits."""
        return probs_to_logits(self.probs, is_binary=True)

    @property
    def probs(self) -> jnp.ndarray:
        """Get probabilities from logits."""
        return logits_to_probs(self._unnormalized_logits, is_binary=True)


@struct.dataclass
class Bernoulli(BaseDistribution):
    """A Bernoulli distribution.

    This class implements a standard Bernoulli distribution that produces binary samples.
    Each sample is 1 with probability p and 0 with probability 1-p.
    """

    @property
    def dtype(self):
        return jnp.int32

    @property
    def param_dtype(self):
        return jnp.float32

    # Core distribution operations
    def sample(self, params: BernoulliParams, num_samples: int, rng_key: PRNGKey):
        """Draw binary samples from the Bernoulli distribution.

        Args:
            params: Bernoulli parameters defining success probabilities
            num_samples: Number of samples
            rng_key: Random number generator key

        Returns:
            samples: Binary samples with values in {0, 1}
        """
        self._check_params(params)

        # Always include sample dimension
        shape = (num_samples,) + params.logits.shape
        samples = jax.random.bernoulli(rng_key, params.probs, shape).astype(self.dtype)
        return self.package_sample(samples)

    def log_prob(self, params: BernoulliParams, sample: BernoulliSample) -> ArrayLike:
        """Compute log probability of binary samples.

        Args:
            params: Bernoulli parameters defining success probabilities
            sample: Binary samples to evaluate

        Returns:
            log_probs: Log probability of each sample under the Bernoulli distribution
        """
        self._check_params(params)
        self._check_sample(sample)

        log_probs = -optax.losses.sigmoid_binary_cross_entropy(params.logits, sample.value)
        return jnp.sum(log_probs, axis=self.param_dims)

    def kl(self, p_params: BernoulliParams, q_params: BernoulliParams):
        """Compute KL divergence between two Bernoulli distributions.

        Args:
            p_params: Parameters for distribution P
            q_params: Parameters for distribution Q

        Returns:
            kl: KL(P||Q) for each batch element
        """
        self._check_params(p_params)
        self._check_params(q_params)

        kls = p_params.probs * (jax.nn.softplus(-q_params.logits) - jax.nn.softplus(-p_params.logits))
        kls += (1 - p_params.probs) * (jax.nn.softplus(q_params.logits) - jax.nn.softplus(p_params.logits))
        return jnp.sum(kls, axis=self.param_dims).clip(0.0)

    def entropy(self, params: BernoulliParams):
        """Compute entropy of the Bernoulli distribution.

        Args:
            params: Bernoulli parameters defining success probabilities

        Returns:
            entropy: Entropy of each Bernoulli distribution
        """
        self._check_params(params)

        entropy = optax.losses.sigmoid_binary_cross_entropy(params.logits, params.probs)
        return jnp.sum(entropy, axis=self.param_dims)

    # Expected values and regularization
    def get_expected_value(self, params: BernoulliParams):
        """Get expected values (probabilities) of the Bernoulli distribution."""
        self._check_params(params)
        return params.probs

    def get_prior_params(self):
        """Get default parameters with zero logits."""
        logits = jnp.log(jnp.ones(self.shape) / 2)
        return BernoulliParams(logits)

    # Sample creation and packaging
    def create_sample(self, leading_shape: Shape, init_val: int = 0):
        """Create an initialized sample."""
        return BernoulliSample.create(leading_shape, self.shape, init_val)

    def package_sample(self, value: jnp.ndarray):
        """Package value into Bernoulli sample."""
        self._check_value(value)
        return BernoulliSample(value)

    def package_params(self, unnormalized_logits: jnp.ndarray):
        """Package raw logits into Bernoulli parameters."""
        self._check_param_value(unnormalized_logits)
        return BernoulliParams(unnormalized_logits)

    def _check_params(self, params: Any):
        assert isinstance(params, BernoulliParams)
        self._check_param_value(params.logits)

    def _check_sample(self, sample: Any):
        assert isinstance(sample, BernoulliSample)
        self._check_value(sample.value)
