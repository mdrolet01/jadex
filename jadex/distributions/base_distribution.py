from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Type

import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, Shape
from flax import struct
from jax.typing import ArrayLike
from omegaconf import DictConfig

from jadex.base.registrable import Registrable
from jadex.utils import non_pytree


@struct.dataclass
class Sample(ABC):
    """Base class for all distribution samples."""

    value: jnp.ndarray

    def __len__(self) -> int:
        return len(self.value)

    @classmethod
    def create(cls, leading_shape: Shape, dim_shape: Shape, init_val: Any) -> "Sample":
        """Create a new sample with given shapes initialized to init_val."""
        value = jnp.ones((*leading_shape, *dim_shape)) * init_val
        return cls(value)

    def matches(self, target_type: List[Type]):
        if not isinstance(target_type, list):
            target_type = [target_type]

        return type(self) in target_type


@struct.dataclass
class DistParams(ABC):
    """Base class for distribution parameters."""


@struct.dataclass
class BaseDistribution(ABC, Registrable):
    registered: ClassVar[Dict[str, "BaseDistribution"]] = dict()
    cfg: DictConfig = non_pytree()

    @classmethod
    def create(cls, cfg):
        return cls(cfg)

    @property
    @abstractmethod
    def dtype(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def param_dtype(self):
        raise NotImplementedError

    @property
    def shape(self) -> tuple:
        return tuple(self.cfg.shape)

    @property
    def ndim(self):
        """Get number of dimensions."""
        return len(self.shape)

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def param_shape(self):
        return tuple(self.cfg.param_shape)

    @property
    def param_ndim(self):
        """Get number of dimensions for params"""
        return len(self.param_shape)

    @property
    def param_dims(self) -> tuple:
        """Get tuple of last dimension indices based on param ndim"""
        return tuple(range(-self.param_ndim, 0))

    # Core distribution operations
    @abstractmethod
    def sample(self, params: DistParams, num_samples: int, rng_key: PRNGKey) -> Sample:
        """Draw samples from the distribution."""
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, params: DistParams, sample: Sample) -> ArrayLike:
        """Compute log probability density of samples."""
        raise NotImplementedError

    @abstractmethod
    def kl(self, p_params: DistParams, q_params: DistParams) -> ArrayLike:
        """Compute KL divergence between two distributions."""
        raise NotImplementedError

    @abstractmethod
    def entropy(self, params: DistParams) -> ArrayLike:
        """Compute entropy of the distribution."""
        raise NotImplementedError

    # Reparameterization
    def reparameterize(self, params: DistParams, rng_key: PRNGKey) -> Sample:
        """Reparameterize samples from the distribution."""
        raise NotImplementedError

    # Expected values and regularization
    @abstractmethod
    def get_expected_value(self, params: DistParams) -> ArrayLike:
        """Get expected values of the distribution."""
        raise NotImplementedError

    def sample_from_prior(self, rng_key: PRNGKey, leading_shape: Shape = ()) -> Sample:
        """Sample from prior distribution."""
        raise NotImplementedError

    def get_prior_params(self) -> DistParams:
        """Get default parameters for regularization."""
        raise NotImplementedError

    # Sample creation and packaging
    @abstractmethod
    def create_sample(self, leading_shape: Shape, init_val: Any) -> Sample:
        """Create an initialized sample."""
        raise NotImplementedError

    @abstractmethod
    def package_sample(self, *args, **kwargs) -> Sample:
        """Package values into a sample object."""
        raise NotImplementedError

    @abstractmethod
    def package_params(self, *args, **kwargs) -> DistParams:
        """Package values into distribution parameters."""
        raise NotImplementedError

    def matches(self, target_type: List[Type]):
        if not isinstance(target_type, list):
            target_type = [target_type]

        return type(self) in target_type

    def _check_param_value(self, param_value: Any):
        assert isinstance(param_value, jnp.ndarray)
        assert param_value.shape[-self.param_ndim :] == self.param_shape
        assert param_value.dtype == self.param_dtype

    def _check_value(self, value: Any):
        assert isinstance(value, jnp.ndarray)
        assert value.shape[-self.ndim :] == self.shape
        assert value.dtype == self.dtype
