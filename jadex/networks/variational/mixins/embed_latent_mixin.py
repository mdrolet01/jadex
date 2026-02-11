from typing import Dict, Optional, Protocol, Type

import flax.linen as nn
import jax.numpy as jnp
from chex import PRNGKey
from flax.typing import VariableDict
from omegaconf import DictConfig

from jadex.base.base_state import get_model_variables_and_mutable
from jadex.distributions.base_distribution import BaseDistribution, Sample
from jadex.distributions.categorical import (
    Categorical,
    CategoricalSample,
    GRMCKCategorical,
    GumbelSoftmaxCategorical,
    GumbelSoftmaxSample,
)
from jadex.distributions.diagonal_gaussian import (
    DiagonalGaussian,
    DiagonalGaussianConstantVariance,
    DiagonalGaussianSample,
)
from jadex.networks.variational.constants import LATENT
from jadex.utils import non_pytree


class LatentVariationalNetworkProtocol(Protocol):
    cfg: DictConfig
    input_dists: Dict[str, BaseDistribution]

    @classmethod
    def raise_not_supported(cls, io_type: str, obj_cls: Type, modality, print_info):
        raise NotImplementedError


class EmbedLatentMixin(LatentVariationalNetworkProtocol, nn.Module):
    embed_latent: Optional[nn.Embed | nn.Dense] = non_pytree()

    @classmethod
    def setup_latent_input(cls, latent_dist: BaseDistribution, cfg, print_info) -> dict:
        if latent_dist.matches([Categorical, GumbelSoftmaxCategorical, GRMCKCategorical]):
            embed_latent = nn.Embed(latent_dist.num_classes, cfg.embed_dim)
        elif latent_dist.matches([DiagonalGaussianConstantVariance, DiagonalGaussian]):
            embed_latent = None
            if cfg.embed_dim != latent_dist.shape[-1]:
                embed_latent = nn.Dense(cfg.embed_dim)
        else:
            cls.raise_not_supported("input", latent_dist, LATENT, print_info)

        return dict(embed_latent=embed_latent)

    def process_latent_samples(
        self,
        latent: Sample,
        variables: VariableDict = None,
        train: bool = True,
        rng_key: PRNGKey = None,
    ) -> jnp.ndarray:

        latent_dist = self.input_dists[LATENT]
        latent_shape = latent.value.shape[-latent_dist.ndim :]
        assert latent_shape == latent_dist.shape, f"{latent_shape} != {latent_dist.shape}"

        if variables is not None:
            model_variables, mutable = get_model_variables_and_mutable(variables, "embed_latent")
            if latent.matches([GumbelSoftmaxSample, CategoricalSample, GRMCKCategorical]):
                all_embs, mutable_updates = self.embed_latent.apply(
                    model_variables,
                    jnp.arange(latent_dist.num_classes),
                    # train=train,
                    # rngs={"dropout": rng_key},
                    mutable=mutable,
                )
                latent_emb = jnp.einsum("... t v, v d -> ... t d", latent.onehot, all_embs)
                assert latent_emb.shape[-2:] == (self.cfg.block_size, self.cfg.embed_dim)
            elif latent.matches(DiagonalGaussianSample):
                latent_emb = latent.value
                if self.embed_latent is not None:
                    latent_emb, mutable_updates = self.embed_latent.apply(
                        model_variables,
                        latent_emb,
                        # train=train,
                        # rngs={"dropout": rng_key},
                        mutable=mutable,
                    )
            else:
                self.raise_not_supported("sample", latent, LATENT, self.print_info)

            mutable_update_dict = {"embed_latent": mutable_updates}

            return latent_emb, mutable_update_dict
        else:
            if latent.matches([GumbelSoftmaxSample, CategoricalSample, GRMCKCategorical]):
                all_embs = self.embed_latent(jnp.arange(latent_dist.num_classes))
                latent_emb = jnp.einsum("... t v, v d -> ... t d", latent.onehot, all_embs)
                assert latent_emb.shape[-2:] == (self.cfg.block_size, self.cfg.embed_dim)
            elif latent.matches(DiagonalGaussianSample):
                latent_emb = latent.value
                if self.embed_latent is not None:
                    latent_emb = self.embed_latent(latent_emb)
            else:
                self.raise_not_supported("sample", latent, LATENT, self.print_info)

            return latent_emb
