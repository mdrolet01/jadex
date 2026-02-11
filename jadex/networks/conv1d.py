import functools
from typing import List

import flax.linen as nn
import jax.numpy as jnp
from flax import struct
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig

from jadex.networks.nn_utils import get_activation_fn, get_bias_init, get_kernel_init, get_norm_layer_fn
from jadex.utils.printing import print_jit


@struct.dataclass
class ResampleConv1DConfig:
    output_length: int
    output_channels: int
    norm: str
    activation: str
    stride: int
    hidden_dim: int
    kernel_size: int
    weight_norm: bool = False
    kernel_dilation: int | str | List | ListConfig = 1
    use_bias: bool = True
    kernel_init_name: str = "glorot_normal"
    bias_init_name: str = "zeros"

    @property
    def kernel_init(self):
        return get_kernel_init(self.kernel_init_name)

    @property
    def bias_init(self):
        return get_bias_init(self.bias_init_name)


class ResampleConv1D(nn.Module):
    config: ResampleConv1DConfig
    print_info: dict = DictConfig({"name": "ResampleConv1D", "uuid": "RESAMPLE_CONV"})

    def activation_fn(self, x):
        return get_activation_fn(self.config.activation)(x)

    def norm_layer_fn(self, x, train):
        return get_norm_layer_fn(self.config.norm)(x, train)

    def kernel_dilation(self, i, mode=None):
        dilation = self.config.kernel_dilation
        if isinstance(dilation, str):
            if dilation.startswith("auto-pow-"):
                # automatically determine to increment or decrement dilations (e.g. "auto-pow-5")
                assert mode is not None
                dilation = dilation.replace("auto", "inc" if mode == "up" else "dec")

            if dilation.startswith("inc-pow-"):
                # incremental dilations (e.g. "inc-pow-5")
                max_pow = int(dilation.split("inc-pow-")[1])
                dilation = 2 ** min(i, max_pow)
            elif dilation.startswith("dec-pow-"):
                # decremental dilations (e.g. "dec-pow-5")
                max_pow = int(dilation.split("dec-pow-")[1])
                dilation = 2 ** max(0, max_pow - i)
            else:
                raise NotImplementedError(f"{self.kernel_dilation} not supported")
        elif isinstance(dilation, List) or isinstance(dilation, ListConfig):
            # A list of kernel dilations (e.g. [1, 2, 4, 8, 16])
            assert len(dilation) > 0
            idx = min(i, len(dilation) - 1)
            dilation = dilation[idx]
            assert isinstance(dilation, int), "must be a list of integers!"

        return dilation

    def apply_conv(self, x, train, stride, dilation, hidden_layer, mode):
        cfg = self.config

        if mode == "up":
            conv = nn.ConvTranspose(
                features=cfg.hidden_dim if hidden_layer else cfg.output_channels,
                kernel_size=cfg.kernel_size,
                strides=stride,
                kernel_dilation=dilation,
                kernel_init=cfg.kernel_init,
                use_bias=cfg.use_bias,
                bias_init=cfg.bias_init,
            )
        else:
            conv = nn.Conv(
                features=cfg.hidden_dim if hidden_layer else cfg.output_channels,
                kernel_size=cfg.kernel_size,
                strides=stride,
                kernel_dilation=dilation,
                use_bias=cfg.use_bias,
                bias_init=cfg.bias_init,
            )

        if cfg.weight_norm and hidden_layer:
            conv = nn.WeightNorm(conv)

        x = conv(x)

        if hidden_layer:
            x = self.activation_fn(x)
            x = self.norm_layer_fn(x, train)

        return x

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        cfg = self.config
        T, C = x.shape[-2:]

        if T < cfg.output_length:
            assert T * cfg.stride <= cfg.output_length, "Stride is too big!"
            mode = "up"
            check_cond_fn = lambda x: x.shape[-2] < (cfg.output_length // cfg.stride)
            final_stride_fn = lambda x: cfg.output_length // x.shape[-2]
        elif T > cfg.output_length:
            assert T // cfg.stride >= cfg.output_length, "Stride is too big!"
            mode = "down"
            check_cond_fn = lambda x: x.shape[-2] > (cfg.output_length * cfg.stride)
            final_stride_fn = lambda x: x.shape[-2] // cfg.output_length
        else:
            print_jit("applying dense layer... resample not needed for", x.shape, self.print_info)
            return nn.Dense(cfg.output_channels)(x)

        apply_conv_fn = functools.partial(self.apply_conv, train=train, mode=mode)
        print_jit(f"{mode}sample received", x.shape, self.print_info)

        i = 0
        while check_cond_fn(x):
            dilation = self.kernel_dilation(i, mode)
            x = apply_conv_fn(x, stride=cfg.stride, dilation=dilation, hidden_layer=True)
            print_jit(f"{mode}sample {i + 1} (dilation {dilation})", x.shape, self.print_info)
            i += 1

        dilation = self.kernel_dilation(i, mode)
        x = apply_conv_fn(x, stride=final_stride_fn(x), dilation=dilation, hidden_layer=False)
        print_jit(f"{mode}sample {i + 1} (dilation {dilation})", x.shape, self.print_info)
        return x
