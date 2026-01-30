from abc import abstractmethod

import flax.linen as nn
import jax.numpy as jnp
from chex import PRNGKey
from flax.typing import VariableDict
from jadex.data.utils.image_patcher import Patcher
from jadex.distributions.base_distribution import BaseDistribution
from jadex.distributions.categorical import Categorical, GRMCKCategorical, GumbelSoftmaxCategorical
from jadex.distributions.diagonal_gaussian import DiagonalGaussian, DiagonalGaussianConstantVariance
from jadex.networks.resnet import ResNetConfig, ResNetDecoder
from jadex.networks.variational.constants import LATENT, X
from jadex.networks.variational.mixins.embed_latent_mixin import EmbedLatentMixin
from jadex.networks.variational.variational_network import VariationalNetwork
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit


class BaseResNetGenerativeModel(VariationalNetwork, EmbedLatentMixin):
    """Base ResNet Generative Model"""

    @classmethod
    @abstractmethod
    def setup_x_output(cls, x_dist: BaseDistribution, cfg, print_info) -> dict:
        raise NotImplementedError

    @abstractmethod
    def upsample_latent_embedding(
        self,
        latent_emb: jnp.ndarray,
        variables: VariableDict = None,
        train: bool = True,
        rng_key: PRNGKey = None,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP LATENT INPUT #####
        latent_kwargs = cls.setup_latent_input(input_dists[LATENT], cfg, print_info)

        ##### SETUP X OUTPUT #####
        x_network_kwargs = cls.setup_x_output(output_dists[X], cfg, print_info)

        return dict(**latent_kwargs, **x_network_kwargs)

    def output(self, samples, train):
        latent = samples[LATENT]
        self._print_input(LATENT, latent)
        latent_emb = self.process_latent_samples(latent=latent, train=train)
        print_jit(f"{LATENT} emb", latent_emb.shape, self.print_info)

        x = self.upsample_latent_embedding(
            latent_dist=self.input_dists[LATENT], latent_emb=latent_emb, train=train
        )

        x_dist = self.output_dists[X]
        if x_dist.matches(DiagonalGaussianConstantVariance):
            params = x_dist.package_params(x)
            self._print_output(X, params, constant_variance=True)
        elif x_dist.matches(DiagonalGaussian):
            x_means, x_logvars = jnp.split(x, 2, axis=-1)
            params = x_dist.package_params(x_means, x_logvars)
            self._print_output(X, params)
        else:
            self.raise_not_supported("output", x_dist, X, self.print_info)

        return params


class VisionResNetGenerativeModel(BaseResNetGenerativeModel):
    """Vision  ResNet Generative Model"""

    patcher: Patcher = non_pytree()
    resnet_decoder: ResNetDecoder = non_pytree()

    @classmethod
    def setup_x_output(cls, output_dist: BaseDistribution, cfg, print_info):
        dim_multiplier = 2 if output_dist.matches(DiagonalGaussian) else 1

        resnet_kwargs = dict(cfg.resnet)
        resnet_kwargs["final_dim"] = output_dist.shape[-1] * dim_multiplier
        output_shape = list(output_dist.shape)
        output_shape[-1] *= dim_multiplier

        patcher: Patcher = Patcher.create(*output_shape, cfg.block_size)
        resnet_kwargs["target_size"] = patcher.get_resnet_decoder_size()

        resnet_config = ResNetConfig.create(resnet_kwargs)
        reset_decoder = ResNetDecoder(resnet_config, print_info)

        return dict(patcher=patcher, resnet_decoder=reset_decoder)

    def upsample_latent_embedding(self, latent_dist: BaseDistribution, latent_emb: jnp.ndarray, train: bool):
        # Reshape token embeddings to a "mini latent image"
        if latent_dist.matches([Categorical, GumbelSoftmaxCategorical, GRMCKCategorical]):
            assert latent_emb.shape[-2:] == (self.cfg.block_size, self.cfg.embed_dim)
        elif latent_dist.matches([DiagonalGaussianConstantVariance, DiagonalGaussian]):
            # embeddings are already passed in (e.g. VQVAE, FSQ)
            pass
        else:
            self.raise_not_supported("upsample latent", latent_dist, LATENT, self.print_info)

        if self.parent_model_name == "FSQModel":
            # To make FSQ more comparable in terms of num parameters
            latent_emb = nn.Dense(int(self.cfg.embed_dim * self.cfg.fsq_ratio))(latent_emb)
            latent_emb = nn.Dense(self.cfg.embed_dim)(latent_emb)
            print_jit("FSQ dense layer", latent_emb.shape, self.print_info)

        latent = self.patcher.get_resnet_decoder_input(latent_emb)
        print_jit(f"{LATENT} emb reshaped", latent.shape, self.print_info)

        x_pad = self.resnet_decoder(latent, train=train)
        x = self.patcher.unpad_resnet_decoder_output(x_pad)
        print_jit(f"unpadded {X}", x.shape, self.print_info)
        return x
