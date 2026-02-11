from typing import Optional

import flax.linen as nn
from einops import rearrange

from jadex.distributions.bernoulli import BernoulliSample
from jadex.distributions.diagonal_gaussian import DiagonalGaussian, DiagonalGaussianSample
from jadex.networks.nn_utils import FeedForwardNetwork
from jadex.networks.variational.constants import LATENT, X
from jadex.networks.variational.variational_network import VariationalNetwork
from jadex.utils import non_pytree


class VisionFeedForwardRecognitionModel(VariationalNetwork):
    """Simple feed forward network for VAE"""

    ffwd: FeedForwardNetwork = non_pytree()
    output_head: nn.Dense = non_pytree()
    logvar_head: Optional[nn.Dense] = non_pytree()

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP X INPUT #####
        ffwd = FeedForwardNetwork(cfg.ffwd_layers, print_info)

        ##### SETUP LATENT OUTPUT #####
        latent_dist = output_dists[LATENT]
        if latent_dist.matches(DiagonalGaussian):
            output_head = nn.Dense(latent_dist.size)
            logvar_head = nn.Dense(latent_dist.size)
        else:
            cls.raise_not_supported("output", latent_dist, LATENT, print_info)

        return dict(ffwd=ffwd, output_head=output_head, logvar_head=logvar_head)

    ##### Output BaseDistribution #####
    def output(self, samples, train):
        x = samples[X]
        self._print_input(X, x)

        if x.matches([BernoulliSample, DiagonalGaussianSample]):
            flat_image = rearrange(x.value, "... h w c -> ... 1 (h w c)")
        else:
            self.raise_not_supported("sample", x, X, self.print_info)

        x = self.ffwd(flat_image, train)

        output_dist = self.output_dists[LATENT]
        if output_dist.matches(DiagonalGaussian):
            latent_mean = self.output_head(x)
            latent_logvar = self.logvar_head(x)
            params = output_dist.package_params(latent_mean, latent_logvar)
            self._print_output(LATENT, params)
        else:
            self.raise_not_supported("output", output_dist, LATENT, self.print_info)

        return params
