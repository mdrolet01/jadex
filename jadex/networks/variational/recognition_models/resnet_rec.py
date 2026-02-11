from abc import abstractmethod

import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from einops import rearrange
from flax.typing import VariableDict
from omegaconf import open_dict

from jadex.base.base_state import get_model_variables_and_mutable
from jadex.distributions.base_distribution import BaseDistribution, Sample
from jadex.distributions.bernoulli import Bernoulli, BernoulliSample
from jadex.distributions.diagonal_gaussian import (
    DiagonalGaussian,
    DiagonalGaussianConstantVariance,
    DiagonalGaussianSample,
)
from jadex.networks.resnet import ResNetConfig, ResNetEncoder
from jadex.networks.variational.constants import LATENT, X
from jadex.networks.variational.variational_network import VariationalNetwork
from jadex.utils import non_pytree


class BaseResNetRecognitionModel(VariationalNetwork):

    @classmethod
    @abstractmethod
    def setup_x_input(cls, x_dist: BaseDistribution, cfg, print_info) -> dict:
        raise NotImplementedError

    @abstractmethod
    def process_x_samples(
        self,
        x: Sample,
        variables: VariableDict = None,
        train: bool = True,
        rng_key: PRNGKey = None,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP X INPUT #####
        x_network_kwargs = cls.setup_x_input(input_dists[X], cfg, print_info)

        ##### SETUP LATENT OUTPUT #####
        latent_dist = output_dists[LATENT]
        if latent_dist.matches(DiagonalGaussianConstantVariance):
            # setup any relevant models here
            pass
        else:
            cls.raise_not_supported("output", latent_dist, LATENT, print_info)

        return dict(**x_network_kwargs)

    def output(self, samples, train):
        x = samples[X]
        self._print_input(X, x)
        x = self.process_x_samples(x, train)

        output_dist = self.output_dists[LATENT]
        if output_dist.matches(DiagonalGaussianConstantVariance):
            params = output_dist.package_params(x)
            self._print_output(LATENT, params, constant_variance=True)
        else:
            self.raise_not_supported("output", output_dist, LATENT, self.print_info)

        return params


class VisionResNetRecognitionModel(BaseResNetRecognitionModel):
    """
    Network which takes in X and outputs a latent square of embeddings (VQ-VAE).
    """

    resnet_encoder: ResNetEncoder = non_pytree()

    @classmethod
    def setup_x_input(cls, x_dist, cfg, print_info):
        if x_dist.matches([Bernoulli, DiagonalGaussian, DiagonalGaussianConstantVariance]):
            with open_dict(cfg):
                cfg.resnet.final_dim = cfg.embed_dim
                cfg.resnet.target_size = int(np.sqrt(cfg.block_size))
            resnet_config = ResNetConfig.create(cfg)
            resnet_encoder = ResNetEncoder(resnet_config, print_info)
        else:
            cls.raise_not_supported("input", x_dist, X, print_info)

        return dict(resnet_encoder=resnet_encoder)

    def process_x_samples(self, x, variables=None, train=True, rng_key=None):
        if x.matches([BernoulliSample, DiagonalGaussianSample]):
            H, W, C = x.value.shape
            assert self.input_dists[X].shape == (H, W, C)
            image = x.value
        else:
            self.raise_not_supported("sample", x, LATENT, self.print_info)

        if variables is not None:
            apply_variables, mutable = get_model_variables_and_mutable(variables, "resnet_encoder")
            latent_image, resnet_mutable_updates = self.resnet_encoder.apply(
                apply_variables,
                image,
                train=train,
                rngs={"dropout": rng_key},
                mutable=mutable,
            )
            process_x_mutable_update_dict = {"resnet_encoder": resnet_mutable_updates}
        else:
            latent_image = self.resnet_encoder(image, train)

        x = rearrange(latent_image, "b h w c -> b (h w) c")

        if variables is not None:
            return x, process_x_mutable_update_dict

        return x
