from typing import Optional

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax import struct
from omegaconf import DictConfig

from jadex.networks.nn_utils import get_activation_fn, get_norm_layer_fn
from jadex.utils import is_power_of_two
from jadex.utils.printing import print_jit


@struct.dataclass
class ResidualBlocksConfig:
    num_blocks: int
    hidden_dim: int
    activation: str
    norm: str


@struct.dataclass
class ResNetConfig:
    target_size: int
    hidden_dim: int
    final_dim: int
    residual: Optional[ResidualBlocksConfig]
    activation: str
    norm: str

    @classmethod
    def create(cls, cfg: DictConfig | dict):
        cfg_dict = dict(cfg)
        residual_cfg = cfg_dict.pop("residual", None)
        residual = ResidualBlocksConfig(**residual_cfg) if residual_cfg else None
        return cls(**cfg_dict, residual=residual)


class ResidualBlocks(nn.Module):
    config: ResidualBlocksConfig
    print_info: dict = DictConfig({"name": "ResidualBlocks", "uuid": "RESIDUAL_BLOCKS"})

    def activation_fn(self, x):
        return get_activation_fn(self.config.activation)(x)

    def norm_layer_fn(self, x, train):
        return get_norm_layer_fn(self.config.norm)(x, train)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        cfg = self.config

        for i in range(cfg.num_blocks):
            x_i = self.activation_fn(x)

            x_i = nn.Conv(
                features=cfg.hidden_dim,
                kernel_size=(3, 3),
                strides=(1, 1),
            )(x_i)

            x_i = self.norm_layer_fn(x_i, train)
            x_i = self.activation_fn(x_i)

            x_i = nn.Conv(
                features=cfg.hidden_dim,
                kernel_size=(1, 1),
                strides=(1, 1),
            )(x_i)
            x_i = self.norm_layer_fn(x_i, train)

            x += x_i
            print_jit(f"residual block {i+1} x", x.shape, self.print_info)

        return self.activation_fn(x)


class ResNetEncoder(nn.Module):
    config: ResNetConfig
    print_info: dict = DictConfig({"name": "ResNetEncoder", "uuid": "RESNET_ENC"})
    first_input_layer: bool = False
    final_output_layer: bool = False

    def activation_fn(self, x):
        return get_activation_fn(self.config.activation)(x)

    def norm_layer_fn(self, x, train):
        return get_norm_layer_fn(self.config.norm)(x, train)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        cfg = self.config
        H, W, C = x.shape[-3:]
        assert H == W, "Only square images are supported"
        assert cfg.target_size <= H // 2 and cfg.target_size >= 2

        print_jit(
            "received x",
            x.shape,
            self.print_info,
            header=self.first_input_layer,
            input=self.first_input_layer,
        )

        num_downsamples = int(np.log2(H / cfg.target_size))
        for i in range(num_downsamples):
            features = max(cfg.hidden_dim // (2 ** (num_downsamples - i)), C)
            x = nn.Conv(
                features=features,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="SAME",
            )(x)

            x = self.norm_layer_fn(x, train)
            x = self.activation_fn(x)
            print_jit(f"downsample {i}", x.shape, self.print_info)

        x = nn.Conv(
            features=cfg.hidden_dim,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
        )(x)

        print_jit("pre-residual", x.shape, self.print_info)

        if cfg.residual:
            x = ResidualBlocks(cfg.residual, self.print_info)(x, train)

        x = nn.Conv(features=cfg.final_dim, kernel_size=(1, 1), strides=(1, 1))(x)

        print_jit(
            "final x",
            x.shape,
            self.print_info,
            footer=self.final_output_layer,
            output=self.final_output_layer,
        )

        return x


class ResNetDecoder(nn.Module):
    config: ResNetConfig
    print_info: dict = DictConfig({"name": "ResNetDecoder", "uuid": "RESNET_DEC"})
    first_input_layer: bool = False
    final_output_layer: bool = False

    def activation_fn(self, x):
        return get_activation_fn(self.config.activation)(x)

    def norm_layer_fn(self, x, train):
        return get_norm_layer_fn(self.config.norm)(x, train)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        cfg = self.config
        H, W, C = x.shape[-3:]
        assert H == W, "Only square images are supported"
        assert is_power_of_two(cfg.target_size) and cfg.target_size >= H * 2

        print_jit(
            "received x",
            x.shape,
            self.print_info,
            header=self.first_input_layer,
            input=self.first_input_layer,
        )

        x = nn.Conv(features=cfg.hidden_dim, kernel_size=(3, 3), strides=(1, 1))(x)
        print_jit(f"first conv x", x.shape, self.print_info)

        if cfg.residual:
            x = ResidualBlocks(cfg.residual, print_info=self.print_info)(x, train)

        num_upsamples = int(np.log2(cfg.target_size / H))
        for i in range(num_upsamples):
            num_features = max(x.shape[-1] // 2, cfg.final_dim)
            if i == num_upsamples - 1:
                num_features = cfg.final_dim

            x = nn.ConvTranspose(
                features=num_features,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding="SAME",
            )(x)

            if i < num_upsamples - 1:
                x = self.norm_layer_fn(x, train)
                x = self.activation_fn(x)

            print_jit(f"upsample {i+1} x", x.shape, self.print_info)

        print_jit(
            f"final x",
            x.shape,
            self.print_info,
            footer=self.final_output_layer,
            output=self.final_output_layer,
        )

        return x
