from typing import Any, Optional

import jax
import jax.numpy as jnp
from chex import PRNGKey, Shape
from flax import struct
from jax.lax import stop_gradient as sg

from jadex.distributions.base_distribution import BaseDistribution, DistParams, Sample
from jadex.distributions.dist_utils import logits_to_probs, probs_to_logits


@struct.dataclass
class CategoricalSample(Sample):
    """A sample from the Categorical distribution.

    Attributes:
        value: Index of the sampled category
        onehot: One-hot encoding of the sampled category
    """

    onehot: jnp.ndarray

    @classmethod
    def create(cls, leading_shape: Shape, dim_shape: Shape, num_classes: int, init_val: int):
        """Create a new sample with given shape."""
        assert isinstance(init_val, int), "init_val must be an integer"
        index = jnp.full((*leading_shape, *dim_shape), init_val, dtype=int)
        onehot = jnp.zeros((*leading_shape, *dim_shape, num_classes))
        return cls(value=index, onehot=onehot)


@struct.dataclass
class CategoricalParams(DistParams):
    """Parameters for a categorical distribution.

    NOTE: This assumes the last axis contains logits:
    logits should not span across multiple axes, but
    you can have multiple leading dimensions!

    Attributes:
        _unnormalized_logits: Unnormalized logits for the categorical distribution
        logits: Normalized logits for the categorical distribution
        probs: Probabilities from logits
    """

    _unnormalized_logits: jnp.ndarray

    @property
    def logits(self) -> jnp.ndarray:
        """Get normalized log probabilities."""
        return probs_to_logits(self.probs)

    @property
    def probs(self) -> jnp.ndarray:
        """Get probabilities from logits."""
        return logits_to_probs(self._unnormalized_logits)


@struct.dataclass
class Categorical(BaseDistribution):
    """A categorical distribution.

    This class implements a distribution over K discrete outcomes.
    Each sample is an integer in [0, K-1] with specified probabilities.
    """

    @property
    def dtype(self):
        return jnp.int32

    @property
    def param_dtype(self):
        return jnp.float32

    @property
    def num_classes(self) -> tuple:
        return self.param_shape[-1]

    @property
    def maximum_entropy(self):
        return self.entropy(self.get_prior_params()).mean()

    # Core distribution operations
    @staticmethod
    def _sample(logits: jnp.ndarray, rng_key: PRNGKey):
        return jnp.argmax(logits + jax.random.gumbel(rng_key, logits.shape), axis=-1)

    def sample(self, params: CategoricalParams, leading_shape: Shape, temperature: float, rng_key: PRNGKey):
        """Draw samples from the categorical distribution.

        Args:
            params: Categorical parameters defining class probabilities
            num_samples: Number of samples
            rng_key: Random number generator key

        Returns:
            samples: Discrete samples with values in [0, K-1]
        """
        self._check_params(params)

        logits, _ = jnp.broadcast_arrays(params.logits, jnp.zeros(leading_shape + params.logits.shape))
        value = self._sample(logits, rng_key)
        return self.package_sample(value)

    def sample_single_categorical(
        self, unnormalized_logits: jnp.ndarray, temperature: float, rng_key: PRNGKey
    ):
        """
        Draw a single RV sample from the categorical distribution.
        NOTE: Checking here is less strict; the unnormalized_logits shape only has to match
        on the last axis, since we want to sample a single RV.
        """
        assert unnormalized_logits.shape[-1] == self.num_classes
        logits = probs_to_logits(logits_to_probs(unnormalized_logits))
        value = self._sample(logits, rng_key)
        onehot = jax.nn.one_hot(value, self.num_classes, dtype=self.param_dtype)
        return CategoricalSample(value=value, onehot=onehot)

    def log_prob(self, params: CategoricalParams, sample: CategoricalSample):
        """Compute log probability of categorical samples.

        Args:
            params: Categorical parameters defining class probabilities
            sample: Integer samples to evaluate

        Returns:
            log_probs: Log probability of each sample under the categorical distribution
        """
        self._check_params(params)
        self._check_sample(sample)

        indices = jnp.expand_dims(sample.value, axis=-1)
        indices, logits = jnp.broadcast_arrays(indices, params.logits)
        log_probs = jnp.take_along_axis(logits, indices[..., :1], axis=-1)

        return jnp.sum(log_probs, axis=self.param_dims)

    def kl(self, p_params: CategoricalParams, q_params: CategoricalParams):
        """Compute KL divergence between two categorical distributions.

        Args:
            p_params: Parameters for distribution P
            q_params: Parameters for distribution Q

        Returns:
            kl: KL(P||Q) for each batch element
        """
        self._check_params(p_params)
        self._check_params(q_params)

        kls = p_params.probs * (p_params.logits - q_params.logits)
        return jnp.sum(kls, axis=self.param_dims).clip(0.0)

    def entropy(self, params: CategoricalParams):
        """Compute entropy of the categorical distribution.

        Args:
            params: Categorical parameters defining class probabilities

        Returns:
            entropy: Entropy of each categorical distribution
        """
        self._check_params(params)

        return -jnp.sum(params.probs * params.logits, axis=self.param_dims)

    # Expected values and regularization
    def get_expected_value(self, params: CategoricalParams):
        """Get expected values of the categorical distribution."""
        self._check_params(params)
        return jnp.argmax(params.logits, axis=-1)

    def get_prior_params(self):
        """Get default parameters with uniform probabilities."""
        unnormalized_logits = jnp.log(jnp.ones(self.param_shape) / self.num_classes)
        return self.package_params(unnormalized_logits)

    def sample_from_prior(self, rng_key: PRNGKey, leading_shape: Shape = ()):
        """Sample from a uniform categorical distribution."""
        uniform_logits = jnp.log(jnp.ones(self.param_shape) / self.num_classes)
        uniform_params = self.package_params(uniform_logits)
        # if self is GumbelSoftmaxCategorical, below will also package the onehots:
        return self.sample(
            params=uniform_params, leading_shape=leading_shape, temperature=1.0, rng_key=rng_key
        )

    # Sample creation and packaging
    def create_sample(self, leading_shape: Shape, init_val: int = 0):
        """Create an initialized categorical sample."""
        return CategoricalSample.create(leading_shape, self.shape, self.num_classes, init_val)

    def package_sample(self, value: jnp.ndarray):
        """Package value into categorical sample."""
        self._check_value(value)
        return CategoricalSample(
            value=value, onehot=jax.nn.one_hot(value, self.num_classes, dtype=self.param_dtype)
        )

    def package_params(self, unnormalized_logits: jnp.ndarray):
        """Package unnormalized logits into categorical parameters."""
        self._check_param_value(unnormalized_logits)
        return CategoricalParams(_unnormalized_logits=unnormalized_logits)

    def _check_params(self, params: Any):
        assert isinstance(params, CategoricalParams)
        self._check_param_value(params.logits)

    def _check_sample(self, sample: Any):
        assert isinstance(sample, CategoricalSample)
        self._check_value(sample.value)
        self._check_param_value(sample.onehot)  # should be same dtype/size as logits


@struct.dataclass
class GumbelSoftmaxSample(CategoricalSample):
    """A sample from the Gumbel Softmax distribution.

    Attributes:
        value: Index of the sampled category
        onehot: One-hot encoding of the sampled category
    """

    pass


@struct.dataclass
class GumbelSoftmaxCategorical(Categorical):
    """A categorical distribution with Gumbel-Softmax relaxation.

    This class implements a continuous approximation to the categorical distribution
    using the Gumbel-Softmax trick for backpropagation through discrete samples.
    """

    # Core distribution operations
    def _sample(self, logits: jnp.ndarray, temperature: Optional[float], rng_key: PRNGKey):
        if temperature is None:
            # NOTE: This is non-differentiable! Should only be used for model evaluation.
            value = Categorical._sample(logits, rng_key)
            onehot = jax.nn.one_hot(value, self.num_classes, dtype=self.param_dtype)
        else:
            noisy_logits = logits + jax.random.gumbel(rng_key, logits.shape)
            probs = jax.nn.softmax(noisy_logits / temperature, axis=-1)
            value = jnp.argmax(probs, axis=-1)
            hard_onehot = jax.nn.one_hot(value, self.num_classes, dtype=self.param_dtype)
            onehot = sg(hard_onehot - probs) + probs  # straight through
        return value, onehot

    def sample(
        self, params: CategoricalParams, leading_shape: Shape, temperature: Optional[float], rng_key: PRNGKey
    ):
        """Draw samples using Gumbel-Softmax relaxation.

        Args:
            params: BaseDistribution parameters
            num_samples: Number of samples
            temperature: Temperature parameter for relaxation. If None, sample from categorical (non-differntiable)
            rng_key: Random number generator key

        Returns:
            samples: Relaxed samples from the distribution
        """
        self._check_params(params)

        logits, _ = jnp.broadcast_arrays(params.logits, jnp.zeros(leading_shape + params.logits.shape))
        value, onehot = self._sample(logits=logits, temperature=temperature, rng_key=rng_key)
        samples = self.package_sample(value, onehot)
        return samples

    def sample_single_categorical(
        self, unnormalized_logits: jnp.ndarray, temperature: float, rng_key: PRNGKey
    ):
        """
        Draw a single RV sample from the relaxed categorical distribution.
        NOTE: Checking here is less strict; the unnormalized_logits shape only has to match
        on the last axis, since we want to sample a single RV.
        """
        assert unnormalized_logits.shape[-1] == self.num_classes
        logits = probs_to_logits(logits_to_probs(unnormalized_logits))
        value, onehot = self._sample(logits=logits, temperature=temperature, rng_key=rng_key)
        samples = GumbelSoftmaxSample(value, onehot)
        return samples

    # Sample creation and packaging
    def create_sample(self, leading_shape: Shape, init_val: int = 0):
        """Create an initialized sample."""
        return GumbelSoftmaxSample.create(leading_shape, self.shape, self.num_classes, init_val)

    def package_sample(self, value: jnp.ndarray, onehot: jnp.ndarray = None):
        """Package value and onehot encoding into Gumbel sample."""
        self._check_value(value)

        if onehot is None:
            onehot = jax.nn.one_hot(value, self.num_classes, dtype=self.param_dtype)

        return GumbelSoftmaxSample(value, onehot)

    def _check_sample(self, sample: Any):
        assert isinstance(sample, GumbelSoftmaxSample)
        self._check_value(sample.value)
        self._check_param_value(sample.onehot)  # should be same dtype/size as logits


@struct.dataclass
class GRMCKSample(CategoricalSample):
    """A sample from GRMCK.

    Attributes:
        value: Index of the sampled category
        onehot: One-hot encoding of the sampled category
    """

    pass


class GRMCKCategorical(GumbelSoftmaxCategorical):
    """
    JAX Implementation of the straight-through Gumbel-Rao estimator.
    Adapted from: https://github.com/nshepperd/gumbel-rao-pytorch

    Reference:
    "Rao-Blackwellizing the Straight-Through Gumbel-Softmax Gradient Estimator"
    https://arxiv.org/abs/2010.04838
    """

    # Core distribution operations
    @staticmethod
    def conditional_gumbel(rng_key: PRNGKey, logits: jnp.ndarray, D: jnp.ndarray, k=1):
        """Outputs k samples of Q = StandardGumbel(), such that argmax(logits + Q) = D."""
        # logits: (..., num_classes)
        # D: one-hot (..., num_classes)

        # iid Exponential samples (rate = 1)
        E = jax.random.exponential(rng_key, shape=(k,) + logits.shape)

        # E of chosen class
        Ei = (D * E).sum(axis=-1, keepdims=True)  # (..., 1)

        (D * E).sum(axis=-1, keepdims=True)

        # partition function (Z)
        Z = jnp.exp(logits).sum(axis=-1, keepdims=True)

        # adjusted Gumbel samples
        eps = jnp.finfo(jnp.float32).eps
        adjusted = D * (-jnp.log(Ei + eps) + jnp.log(Z + eps))
        adjusted += (1 - D) * -jnp.log(E / jnp.exp(logits) + Ei / Z + eps)

        return adjusted - logits  # (..., num_classes)

    @staticmethod
    def gumbel_rao(rng_key, logits, k=1, temperature=1.0, I=None):
        """
        Returns a categorical sample from logits as a one-hot vector, with Gumbel-Rao gradient.
        k: number of Rao-Blackwellization samples.
        """
        num_classes = logits.shape[-1]

        key1, key2 = jax.random.split(rng_key, 2)

        # Sample categorical if I is None
        if I is None:
            I = Categorical._sample(logits, key1)

        D = jax.nn.one_hot(I, num_classes)

        # Conditional Gumbel adjustment
        adjusted = logits + GRMCKCategorical.conditional_gumbel(key2, logits, D, k=k)
        avg_probs = jax.nn.softmax(adjusted / temperature, axis=-1).mean(axis=0)
        st_onehot = sg(D - avg_probs) + avg_probs  # straight through
        return st_onehot

    def _sample(self, logits: jnp.ndarray, temperature: Optional[float], rng_key: PRNGKey):
        if temperature is None:
            # NOTE: This is non-differentiable! Should only be used for model evaluation.
            value = Categorical._sample(logits, rng_key)
            onehot = jax.nn.one_hot(value, self.num_classes, dtype=self.param_dtype)
        else:
            onehot = self.gumbel_rao(rng_key, logits, k=self.cfg.k_value, temperature=temperature)
            value = onehot.argmax(axis=-1)
        return value, onehot

    def sample_single_categorical(
        self, unnormalized_logits: jnp.ndarray, temperature: float, rng_key: PRNGKey
    ):
        """
        Draw a single RV sample from the relaxed categorical distribution using the Gumbel-Rao gradient.
        NOTE: Checking here is less strict; the unnormalized_logits shape only has to match
        on the last axis, since we want to sample a single RV.
        """
        assert unnormalized_logits.shape[-1] == self.num_classes
        logits = probs_to_logits(logits_to_probs(unnormalized_logits))
        value, onehot = self._sample(logits=logits, temperature=temperature, rng_key=rng_key)
        samples = GRMCKSample(value=value, onehot=onehot)
        return samples
