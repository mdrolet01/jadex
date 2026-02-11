from abc import ABC, abstractmethod
from typing import Dict, Optional

import flax.linen as nn
import jax.numpy as jnp
from chex import PRNGKey
from flax import struct
from omegaconf import DictConfig

from jadex.base.base_state import BaseState
from jadex.networks.running_mean_std import RunningMeanStd
from jadex.utils import non_pytree


@struct.dataclass
class BaseModelLossArgs:
    """Arguments for the loss function."""


class BaseModel(nn.Module, ABC):
    cfg: DictConfig = non_pytree()
    scalers: Optional[Dict[str, RunningMeanStd]] = non_pytree()

    def loss_fn(self, loss_args: BaseModelLossArgs, rng_key: PRNGKey):
        """Computes the model's loss

        Args:
            loss_args: BaseModelLossArgs
            rng_key: PRNGKey

        Returns:
            metrics: Dictionary of relevant loss metrics
            (should include `state_updates`, and `losses` if using multiple txs)
        """
        raise NotImplementedError

    @abstractmethod
    def get_loss_args(self, state: BaseState, batch: dict, rng_key: PRNGKey) -> BaseModelLossArgs:
        """Get the model's loss arguments"""
        raise NotImplementedError

    def get_predictions(self, state: BaseState, batch: dict, rng_key: PRNGKey):
        """Generate predictions for input data.

        Args:
            state: BaseState
            batch: Dict of data to predict
            rng_key: PRNGKey

        Returns:
            metrics: Dictionary of relevant metrics
        """
        raise NotImplementedError

    def apply_scaler(self, data: jnp.ndarray, scaler_vars: dict, modality: str):
        if scaler_vars:
            return self.scalers[modality].scale(
                data,
                mean=scaler_vars[modality]["run_stats"]["mean"],
                var=scaler_vars[modality]["run_stats"]["var"],
            )
        else:
            return data

    def apply_inverse_scaler(self, data: jnp.ndarray, scaler_vars: dict, modality: str):
        if scaler_vars:
            return self.scalers[modality].inverse_scale(
                data,
                mean=scaler_vars[modality]["run_stats"]["mean"],
                var=scaler_vars[modality]["run_stats"]["var"],
            )
        else:
            return data
