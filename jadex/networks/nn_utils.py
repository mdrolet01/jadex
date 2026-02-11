import re
from typing import Callable, List, Optional

import flax.linen as nn
import jax.numpy as jnp
from omegaconf import DictConfig

from jadex.utils.printing import print_jit

__activation_fns = {
    "relu": nn.relu,
    "tanh": nn.tanh,
    "sigmoid": nn.sigmoid,
    "swish": nn.swish,
    "elu": nn.elu,
}

__norm_layer_fns = {
    "batch_norm": lambda x, train: nn.BatchNorm()(x, use_running_average=not train),
    "layer_norm": lambda x, train: nn.LayerNorm()(x),
    "group_norm": lambda x, train: nn.GroupNorm()(x),
}


__kernel_inits = {
    "default": nn.linear.default_kernel_init,
    "xavier": nn.initializers.xavier_uniform(),
    "kaiming": nn.initializers.kaiming_normal(),
    "glorot_normal": nn.initializers.glorot_normal(),
}

__bias_inits = {
    "normal": nn.initializers.normal(stddev=1e-6),
    "zeros": nn.initializers.zeros_init(),
}

__posemb_inits = {
    "default": nn.linear.default_embed_init,
}

__dtypes = {
    "float32": jnp.float32,
}

NULL_NAMES = ["nada", "none", "null"]


def get_activation_fn(activation: Optional[str]) -> Callable:
    if activation is None or (isinstance(activation, str) and activation.lower() in NULL_NAMES):
        return lambda x: x
    assert activation in __activation_fns.keys()
    return __activation_fns[activation]


def get_norm_layer_fn(norm: Optional[str]) -> Callable:
    if norm is None or (isinstance(norm, str) and norm.lower() in NULL_NAMES):
        return lambda x, train: x
    assert norm in __norm_layer_fns.keys()
    norm_layer_fn = __norm_layer_fns[norm]
    return norm_layer_fn


def get_kernel_init(kernel_init_name: Optional[str]):
    if kernel_init_name is None or kernel_init_name.lower() in NULL_NAMES:
        return None
    assert kernel_init_name in __kernel_inits.keys()
    return __kernel_inits[kernel_init_name]


def get_bias_init(bias_init_name: Optional[str]):
    if bias_init_name is None or bias_init_name.lower() in NULL_NAMES:
        return None
    assert bias_init_name in __bias_inits.keys()
    return __bias_inits[bias_init_name]


def get_embed_init(embed_init_name: Optional[str]):
    if embed_init_name is None or embed_init_name.lower() in NULL_NAMES:
        return None
    assert embed_init_name in __posemb_inits.keys()
    return __posemb_inits[embed_init_name]


def get_dtype(dtype_name: str):
    assert dtype_name in __dtypes.keys()
    return __dtypes[dtype_name]


class FeedForwardNetwork(nn.Module):
    layer_names: List[str]
    print_info: dict = DictConfig({"name": "FeedForward", "uuid": "FFWD"})

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        layer_idx = 0
        arg_list = []

        for i, spec in enumerate(self.layer_names):
            match = re.match(r"(\w+):([\w\.]+)", spec)
            assert match is not None, f"Layer spec '{spec}' is invalid. Expected format 'type:param'."
            layer_type, param = match.groups()

            if layer_type == "dense":
                # Print previous dense group if any
                if arg_list:
                    print_jit(
                        f"ffwd layer {layer_idx}: Dense({', '.join(arg_list)}), x",
                        x.shape,
                        self.print_info,
                    )
                layer_idx += 1
                arg_list = [param]
                x = nn.Dense(int(param))(x)

            elif layer_type == "activation":
                fn = get_activation_fn(param)
                x = fn(x)
                arg_list.append(param)

            elif layer_type == "norm":
                fn = get_norm_layer_fn(param)
                x = fn(x, train)
                arg_list.append(param)

            elif layer_type == "dropout":
                rate = float(param)
                x = nn.Dropout(rate=rate)(x, deterministic=not train)
                arg_list.append(f"dropout:{rate}")

            else:
                raise ValueError(f"Unsupported layer type: '{layer_type}'")

        # Print the final layer group
        if arg_list:
            print_jit(
                f"ffwd layer {layer_idx}: Dense({', '.join(arg_list)}), x",
                x.shape,
                self.print_info,
            )

        return x
