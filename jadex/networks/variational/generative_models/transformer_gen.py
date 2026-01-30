from abc import abstractmethod
from typing import Optional

import flax.linen as nn

from jadex.distributions.base_distribution import BaseDistribution
from jadex.distributions.diagonal_gaussian import DiagonalGaussian, DiagonalGaussianConstantVariance
from jadex.networks.conv1d import ResampleConv1D, ResampleConv1DConfig
from jadex.networks.transformer import TransformerDecoder, TransformerDecoderConfig
from jadex.networks.variational.constants import LATENT, X
from jadex.networks.variational.mixins.embed_latent_mixin import EmbedLatentMixin
from jadex.networks.variational.variational_network import VariationalNetwork
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit


class BaseTransformerGenerativeModel(VariationalNetwork, EmbedLatentMixin):
    transformer: TransformerDecoder = non_pytree()

    @classmethod
    @abstractmethod
    def setup_x_output(cls, x_dist: BaseDistribution, latent_dist: BaseDistribution, cfg, print_info):
        raise NotImplementedError

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP LATENT INPUT #####
        latent_dist = input_dists[LATENT]
        latent_kwargs = cls.setup_latent_input(latent_dist, cfg, print_info)
        transformer_config = TransformerDecoderConfig(**cfg.transformer)
        transformer = TransformerDecoder(transformer_config)

        ##### SETUP X OUTPUT #####
        x_network_kwargs = cls.setup_x_output(output_dists[X], latent_dist, cfg, print_info)

        return dict(transformer=transformer, **latent_kwargs, **x_network_kwargs)


class TrajTransformerGenerativeModel(BaseTransformerGenerativeModel):
    output_head: nn.Dense = non_pytree()
    logvar_head: Optional[nn.Dense] = non_pytree()
    resample_conv: Optional[ResampleConv1D] = non_pytree()

    @staticmethod
    def get_supported_x_dists():
        return [DiagonalGaussianConstantVariance, DiagonalGaussian]

    @classmethod
    def setup_x_output(cls, x_dist, latent_dist, cfg, print_info):
        resample_conv = logvar_head = None
        if x_dist.matches([DiagonalGaussianConstantVariance, DiagonalGaussian]):
            if latent_dist.shape[-1] != x_dist.shape[-2]:
                resample_conv_config = ResampleConv1DConfig(**cfg.resample_conv1d)
                resample_conv = ResampleConv1D(resample_conv_config, print_info)
            output_head = nn.Dense(x_dist.shape[-1])
            if x_dist.matches(DiagonalGaussian):
                logvar_head = nn.Dense(x_dist.shape[-1])
        else:
            cls.raise_not_supported("output", x_dist, X, print_info)

        return dict(output_head=output_head, logvar_head=logvar_head, resample_conv=resample_conv)

    def output(self, samples, train):
        ##### PROCESS LATENT INPUT #####
        latent = samples[LATENT]
        self._print_input(LATENT, latent)
        latent_emb = self.process_latent_samples(latent=latent, train=train)
        print_jit(f"{LATENT} emb", latent_emb.shape, self.print_info)

        if self.parent_model_name == "FSQModel":
            # To make FSQ more comparable in terms of num parameters
            latent_emb = nn.Dense(int(self.cfg.embed_dim * self.cfg.fsq_ratio))(latent_emb)
            latent_emb = nn.Dense(self.cfg.embed_dim)(latent_emb)
            print_jit("FSQ dense layer", latent_emb.shape, self.print_info)

        # Pass latent through transformer
        x = self.transformer(latent_emb, train)
        print_jit(f"{X} transformer output", x.shape, self.print_info)

        if self.resample_conv is not None:
            x = self.resample_conv(x, train)

        ##### OUTPUT X #####
        x_dist = self.output_dists[X]
        if x_dist.matches(DiagonalGaussianConstantVariance):
            x_mean = self.output_head(x)
            params = x_dist.package_params(x_mean)
            self._print_output(X, params)
        elif x_dist.matches(DiagonalGaussian):
            x_mean = self.output_head(x)
            x_logvar = self.logvar_head(x)
            params = x_dist.package_params(x_mean, x_logvar)
            self._print_output(X, params)
        else:
            self.raise_not_supported("output", x_dist, X, self.print_info)

        return params
