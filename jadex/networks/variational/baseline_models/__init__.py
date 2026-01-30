import jax.numpy as jnp
from flax.typing import VariableDict

from jadex.distributions.base_distribution import Sample
from jadex.networks.variational.variational_network import VariationalNetwork


class BaselineModel(VariationalNetwork):
    """
    Control Variate Model (used for value baseline in advantage estimation)
    """

    @property
    def scale(self):
        return self.cfg.scale


def register_baseline_models():
    from .vision_baseline import VisionFeedForwardBaselineModel, VisionResNetBaselineModel

    VisionResNetBaselineModel.register()
    VisionFeedForwardBaselineModel.register()
