import flax.linen as nn
import numpy as np
from einops import rearrange
from omegaconf import open_dict

from jadex.data.utils.image_patcher import Patcher
from jadex.distributions.base_distribution import BaseDistribution, Sample
from jadex.distributions.bernoulli import Bernoulli, BernoulliSample
from jadex.distributions.diagonal_gaussian import (
    DiagonalGaussian,
    DiagonalGaussianConstantVariance,
    DiagonalGaussianSample,
)
from jadex.networks.resnet import ResNetConfig, ResNetEncoder
from jadex.networks.variational.baseline_models import BaselineModel
from jadex.networks.variational.constants import LATENT, X
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit


class VisionResNetBaselineModel(BaselineModel):
    resnet_encoder: ResNetEncoder = non_pytree()
    output_ffwd: nn.Dense = non_pytree()

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP X INPUT #####
        x_network_kwargs = cls.setup_x_input(input_dists[X], cfg, print_info)
        return dict(**x_network_kwargs)

    @classmethod
    def setup_x_input(cls, x_dist: BaseDistribution, cfg, print_info):
        if x_dist.matches([Bernoulli, DiagonalGaussian, DiagonalGaussianConstantVariance]):
            with open_dict(cfg):
                cfg.scale = int(x_dist.size)
                cfg.resnet.target_size = int(np.sqrt(cfg.num_patches))
            resnet_config = ResNetConfig.create(cfg.resnet)
            resnet_encoder = ResNetEncoder(resnet_config, print_info)
            output_ffwd = nn.Dense(1)
        else:
            cls.raise_not_supported("input", x_dist, X, print_info)

        return dict(resnet_encoder=resnet_encoder, output_ffwd=output_ffwd)

    def process_x_samples(self, x: Sample, train: bool):
        if x.matches([BernoulliSample, DiagonalGaussianSample]):
            H, W, C = x.value.shape[-3:]
            assert self.input_dists[X].shape == (H, W, C)
            image = x.value
        else:
            self.raise_not_supported("sample", x, LATENT, self.print_info)

        latent_image = self.resnet_encoder(image, train)
        x = rearrange(latent_image, "b h w c -> b (h w c)")
        return x

    def output(self, samples, train):
        x = samples[X]
        self._print_input(X, x)
        x = self.process_x_samples(x, train)
        print_jit(f"{X} emb", x.shape, self.print_info)
        x = self.output_ffwd(x).squeeze(-1)
        baseline = x * self.scale
        print_jit("baseline", baseline.shape, self.print_info, output=True, footer=True)
        return baseline


class VisionFeedForwardBaselineModel(BaselineModel):
    output_ffwd: nn.Dense = non_pytree()
    patcher: Patcher = non_pytree()
    patch_ffwd: nn.Dense = non_pytree()

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP X INPUT #####
        x_network_kwargs = cls.setup_x_input(input_dists[X], cfg, print_info)
        return dict(**x_network_kwargs)

    @classmethod
    def setup_x_input(cls, x_dist: BaseDistribution, cfg, print_info):
        if x_dist.matches([Bernoulli, DiagonalGaussian, DiagonalGaussianConstantVariance]):
            with open_dict(cfg):
                cfg.scale = int(x_dist.size)
            output_ffwd = nn.Dense(1)
            patcher = Patcher.create(*x_dist.shape, desired_num_patches=cfg.num_patches)
            patch_ffwd = nn.Dense(cfg.embed_dim)
        else:
            cls.raise_not_supported("input", x_dist, X, print_info)

        return dict(
            output_ffwd=output_ffwd,
            patcher=patcher,
            patch_ffwd=patch_ffwd,
        )

    def process_x_samples(self, x: Sample, train: bool):
        if x.matches([BernoulliSample, DiagonalGaussianSample]):
            H, W, C = x.value.shape[-3:]
            assert self.input_dists[X].shape == (H, W, C)
            image = x.value
        else:
            self.raise_not_supported("sample", x, LATENT, self.print_info)

        flat_pad_patches = self.patcher.patchify_pad_flat(image)
        x = self.patch_ffwd(flat_pad_patches)
        print_jit(f"{X} patches", x.shape, self.print_info)
        x = rearrange(x, "b t c -> b (t c)")
        return x

    def output(self, samples, train):
        x = samples[X]
        self._print_input(X, x)
        x = self.process_x_samples(x, train)
        print_jit(f"{X} emb", x.shape, self.print_info)
        x = self.output_ffwd(x).squeeze(-1)
        baseline = x * self.scale
        print_jit("baseline", baseline.shape, self.print_info, output=True, footer=True)
        return baseline
