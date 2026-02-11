from typing import Optional

import flax.linen as nn
from einops import rearrange

from jadex.distributions.bernoulli import Bernoulli
from jadex.distributions.diagonal_gaussian import DiagonalGaussian, DiagonalGaussianConstantVariance
from jadex.networks.nn_utils import FeedForwardNetwork
from jadex.networks.variational.constants import LATENT, X
from jadex.networks.variational.mixins.embed_latent_mixin import EmbedLatentMixin
from jadex.networks.variational.variational_network import VariationalNetwork
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit


class VisionFeedForwardGenerativeModel(VariationalNetwork, EmbedLatentMixin):
    """
    Applies a feedforward network to the concatenation of embeddings.
    """

    ffwd: FeedForwardNetwork = non_pytree()
    output_head: nn.Dense = non_pytree()
    logvar_head: Optional[nn.Dense] = non_pytree()

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP LATENT INPUT #####
        latent_kwargs = cls.setup_latent_input(input_dists[LATENT], cfg, print_info)

        ##### SETUP X OUTPUT #####
        x_dist = output_dists[X]
        ffwd = FeedForwardNetwork(cfg.ffwd_layers, print_info)
        output_head = nn.Dense(x_dist.size)

        if x_dist.matches(DiagonalGaussian):
            logvar_head = nn.Dense(x_dist.size)
        elif x_dist.matches([DiagonalGaussianConstantVariance, Bernoulli]):
            logvar_head = None
        else:
            cls.raise_not_supported("output", x_dist, X, print_info)

        return dict(ffwd=ffwd, output_head=output_head, logvar_head=logvar_head, **latent_kwargs)

    def output(self, samples, train):
        ##### PROCESS LATENT #####
        latent = samples[LATENT]
        self._print_input(LATENT, latent)
        latent_emb = self.process_latent_samples(latent, train=train)
        print_jit(f"{LATENT} emb", latent_emb.shape, self.print_info)

        if self.parent_model_name == "FSQModel":
            # To make FSQ more comparable in terms of num parameters
            latent_emb = nn.Dense(self.cfg.fsq_dim)(latent_emb)
            print_jit("FSQ dense layer", latent_emb.shape, self.print_info)

        latent_emb = rearrange(latent_emb, "... n c -> ... (n c)")
        print_jit(f"{LATENT} emb flat", latent_emb.shape, self.print_info)
        x = self.ffwd(latent_emb, train)

        ##### OUTPUT X #####
        x_dist = self.output_dists[X]
        if x_dist.matches(DiagonalGaussianConstantVariance):
            x_mean = self.output_head(x)
            print_jit(f"{X} mean", x_mean.shape, self.print_info)
            x_mean = x_mean.reshape(*x.shape[:-1], *x_dist.shape)
            params = x_dist.package_params(x_mean)
            self._print_output(X, params, constant_variance=True)
        elif x_dist.matches(DiagonalGaussian):
            x_mean = self.output_head(x)
            print_jit(f"{X} mean", x_mean.shape, self.print_info)
            x_mean = x_mean.reshape(*x.shape[:-1], *x_dist.shape)
            x_logvar = self.logvar_head(x)
            print_jit(f"{X} logvar", x_logvar.shape, self.print_info)
            x_logvar = x_logvar.reshape(*x.shape[:-1], *x_dist.shape)
            params = x_dist.package_params(x_mean, x_logvar)
            self._print_output(X, params)
        elif x_dist.matches(Bernoulli):
            x_logits = self.output_head(x)
            print_jit(f"{X} logits", x_logits.shape, self.print_info)
            x_logits = x_logits.reshape(*x.shape[:-1], *x_dist.shape)
            params = x_dist.package_params(x_logits)
            self._print_output(X, params)
        else:
            self.raise_not_supported("output", x_dist, X, self.print_info)

        return params
