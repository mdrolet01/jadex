import re
import zlib
from dataclasses import fields
from typing import Callable, Dict, List, Type

import chex
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from chex import PRNGKey
from flax import struct
from flax.serialization import from_bytes, to_bytes
from flax.typing import VariableDict
from omegaconf import DictConfig, OmegaConf, open_dict

from jadex.data.dataloader.jax_sampler import SampleBuffer, SampleBufferData
from jadex.utils import non_pytree
from jadex.utils.hdf5io import load as hdf5io_load
from jadex.utils.hdf5io import save_h5_checkpoint
from jadex.utils.printing import print_green, print_jit_str, print_yellow
from jadex.utils.scheduler import create_scheduler

EXPECTED_MUTABLE_KEYS = ["batch_stats", "run_stats"]


class BaseState(struct.PyTreeNode):
    """
    Handles checkpoints and state updates for model training.
    """

    cfg: DictConfig = non_pytree()
    apply_fn: Callable = non_pytree()
    variables: VariableDict
    rng_key: PRNGKey
    txs: Dict[str, optax.GradientTransformation] = non_pytree()
    tx_groups: Dict[str, List] = non_pytree()
    tx_lr_schedulers: Dict[str, optax.Schedule] = non_pytree()  # for logging
    max_step_freq: int = non_pytree()
    total_time_min: float
    internal_step: int
    sample_buffer_data: SampleBufferData
    opt_states: Dict[str, optax.OptState]
    step: int | jax.Array

    @property
    def scaler_vars(self):
        return self.variables.get("scalers", {})

    @classmethod
    def create(
        cls,
        cfg: DictConfig,
        apply_fn: Callable,
        variables: VariableDict,
        rng_key: PRNGKey,
        **kwargs,
    ):
        txs = {}
        tx_groups = {}
        tx_lr_schedulers = {}
        opt_states = {}
        max_step_freq = 1

        for opt_name, opt_cfg in cfg.optimizers.items():
            with open_dict(opt_cfg.copy()) as opt_cfg_dict:
                opt_param_names = opt_cfg_dict.pop("params")
                opt, lr_scheduler, grad_accum_steps = make_optimizer(opt_cfg_dict)

            if grad_accum_steps and grad_accum_steps > max_step_freq:
                max_step_freq = grad_accum_steps

            opt_params = {}
            param_counts = {}
            tx_num_params = 0
            for name in opt_param_names:
                model_params = variables[name]["params"]
                opt_params[name] = model_params
                # get param counts
                num_model_params = sum(x.size for x in jax.tree_util.tree_leaves(model_params))
                param_counts[name] = num_model_params
                tx_num_params += num_model_params

            with open_dict(opt_cfg):
                opt_cfg.param_counts = param_counts

            opt_state = opt.init(opt_params)
            txs[opt_name] = opt
            opt_states[opt_name] = opt_state
            tx_groups[opt_name] = opt_param_names
            tx_lr_schedulers[opt_name] = lr_scheduler

        if max_step_freq > 1:
            print_yellow(f"NOTE: state.step will increment once every {max_step_freq} forward passes.")

        sample_buffer_data = SampleBuffer.create_data(
            dataset_len=cfg.dataset.num_train,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            rng_key=jax.random.PRNGKey(cfg.train.seed),
        )

        return cls(
            cfg=cfg,
            apply_fn=apply_fn,
            variables=variables,
            rng_key=rng_key,
            txs=txs,
            tx_groups=tx_groups,
            tx_lr_schedulers=tx_lr_schedulers,
            max_step_freq=max_step_freq,
            total_time_min=0.0,
            internal_step=0,
            sample_buffer_data=sample_buffer_data,
            opt_states=opt_states,
            step=0,
            **kwargs,
        )

    def print_opt_params(self):
        total_param_count = 0
        for tx_name, opt_param_names in self.tx_groups.items():
            for name in opt_param_names:
                sub_params = self.variables[name]["params"]
                tree_str = params_tree_string(sub_params)
                print_jit_str(name.upper() + "\n\n" + tree_str, with_header_footer=True)
                total_param_count += sum(x.size for x in jax.tree_util.tree_leaves(sub_params))

        total_mem = float(jax.local_devices()[0].memory_stats()["bytes_limit"]) / 1024**3

        print_jit_str(
            f"Total model parameters: {total_param_count:,}. Total memory on device: {round(total_mem, 2)} GB\n"
            + f"JAX devices: {jax.devices()}",
            with_header_footer=True,
        )

    def _multi_tx_update(self, loss_args, rng_key):
        rng_key, new_rng_key = jax.random.split(rng_key)
        tx_names = sorted(self.txs.keys())

        # Collect params for each tx
        apply_variables = self.variables
        tx_params_list = []
        for tx_name in tx_names:
            tx_params = {}
            for model_name in self.tx_groups[tx_name]:
                tx_params[model_name] = apply_variables[model_name].pop("params")
            tx_params_list.append(tx_params)

        def loss_fn(*tx_params_list):
            # Update variables with tx params
            for tx_params_dict in tx_params_list:
                for model_name, model_params in tx_params_dict.items():
                    apply_variables[model_name]["params"] = model_params

            # Forward pass
            metrics = self.apply_fn(apply_variables, loss_args=loss_args, rng_key=rng_key, train=True)

            # Collect tx losses
            losses = []
            for tx_name in tx_names:
                tx_group = self.tx_groups[tx_name]
                tx_losses = [metrics["losses"][model_name] for model_name in tx_group]
                losses.append(sum(tx_losses))

            return tuple(losses), metrics

        # Multivariate grad function
        _, grad_fn, metrics = jax.vjp(loss_fn, *tx_params_list, has_aux=True)

        new_opt_states = {}
        new_internal_step = self.internal_step + 1
        new_step = jax.lax.cond(
            new_internal_step % self.max_step_freq == 0, lambda: self.step + 1, lambda: self.step
        )

        variables = self.update_variables_with_mutable(self.variables, metrics.pop("mutable_updates"))
        for i, tx_name in enumerate(tx_names):
            # Get grad wrt params
            tx_vec = tuple(np.eye(len(tx_names))[i])
            p_grads_list = grad_fn(tx_vec)

            grads = p_grads_list[i]
            if not self.cfg.dataset.get("in_memory", False):
                grads = jax.lax.pmean(grads, axis_name="batch")

            # Apply grad updates
            grad_updates, new_opt_states[tx_name] = self.txs[tx_name].update(
                grads, self.opt_states[tx_name], tx_params_list[i]
            )
            param_updates = optax.apply_updates(tx_params_list[i], grad_updates)

            # Update variables
            for model_name, model_params_update in param_updates.items():
                chex.assert_trees_all_equal_structs(variables[model_name]["params"], model_params_update)
                variables[model_name]["params"] = model_params_update

            metrics[f"{tx_name}_grad_norm"] = optax.global_norm(grads)
            metrics[f"{tx_name}_lr"] = self.tx_lr_schedulers[tx_name](self.step)

        new_state = self.replace(
            internal_step=new_internal_step,
            step=new_step,
            opt_states=new_opt_states,
            rng_key=new_rng_key,
            variables=variables,
            **metrics.pop("state_updates", {}),
        )

        return new_state, metrics

    def _single_tx_update(self, loss_args, rng_key):
        assert len(self.txs) == 1
        tx_name = next(iter(self.txs.keys()))
        tx_group = self.tx_groups[tx_name]
        rng_key, new_rng_key = jax.random.split(rng_key)

        # Collect params for tx
        apply_variables = self.variables
        tx_params = {}
        for param_name in tx_group:
            tx_params[param_name] = apply_variables[param_name].pop("params")

        def loss_fn(tx_params):
            # Update variables with tx params
            for model_name, model_params in tx_params.items():
                apply_variables[model_name]["params"] = model_params

            # Forward pass
            metrics = self.apply_fn(apply_variables, loss_args=loss_args, rng_key=rng_key, train=True)

            return metrics["loss"], metrics

        new_opt_states = {}
        new_internal_step = self.internal_step + 1
        new_step = jax.lax.cond(
            new_internal_step % self.max_step_freq == 0, lambda: self.step + 1, lambda: self.step
        )

        # Compute grad wrt params
        grads, metrics = jax.grad(loss_fn, has_aux=True)(tx_params)
        if not self.cfg.dataset.get("in_memory", False):
            grads = jax.lax.pmean(grads, axis_name="batch")

        # Apply grad updates
        grad_updates, new_opt_states[tx_name] = self.txs[tx_name].update(
            grads, self.opt_states[tx_name], tx_params
        )
        param_updates = optax.apply_updates(tx_params, grad_updates)

        # Update variables
        variables = self.update_variables_with_mutable(self.variables, metrics.pop("mutable_updates"))
        for model_name, model_params_update in param_updates.items():
            chex.assert_trees_all_equal_structs(variables[model_name]["params"], model_params_update)
            variables[model_name]["params"] = model_params_update

        metrics[f"{tx_name}_grad_norm"] = optax.global_norm(grads)
        metrics[f"{tx_name}_lr"] = self.tx_lr_schedulers[tx_name](self.step)

        new_state = self.replace(
            internal_step=new_internal_step,
            step=new_step,
            opt_states=new_opt_states,
            rng_key=new_rng_key,
            variables=variables,
            **metrics.pop("state_updates", {}),
        )

        return new_state, metrics

    def update_variables_with_mutable(self, variables, mutable_updates):
        assert isinstance(mutable_updates, dict)
        if mutable_updates:
            # update scalers
            scaler_updates = mutable_updates.pop("scalers", {})
            if scaler_updates:
                for modality in scaler_updates.keys():
                    modality_scaler_update = scaler_updates[modality]
                    assert isinstance(modality_scaler_update, dict)
                    assert len(modality_scaler_update.keys()) == 1
                    assert "run_stats" in modality_scaler_update.keys()
                    chex.assert_trees_all_equal_structs(
                        variables["scalers"][modality], modality_scaler_update
                    )
                    variables["scalers"][modality] = modality_scaler_update

            # update remaining mutables
            for variable_name, variable_update in mutable_updates.items():
                # NOTE: we only expect batch_stats or run_stats to be updated
                assert isinstance(variable_update, dict)
                if variable_update:
                    for update_name in variable_update.keys():
                        assert update_name in EXPECTED_MUTABLE_KEYS
                        chex.assert_trees_all_equal_structs(
                            variables[variable_name][update_name], variable_update[update_name]
                        )
                        variables[variable_name][update_name] = variable_update[update_name]

        return variables

    def perform_gradient_update(self, loss_args, rng_key):
        if len(self.txs) > 1:
            state, metrics = self._multi_tx_update(loss_args, rng_key)
        else:
            state, metrics = self._single_tx_update(loss_args, rng_key)

        return state, metrics

    @staticmethod
    def load_cfg(h5_file: str, merge_dict: dict = {}) -> DictConfig:
        """Load the hydra configuration from a hdf5 file."""
        with h5py.File(h5_file, "r") as h5:
            hydra_cfg_dict = yaml.safe_load(h5.attrs["hydra_cfg"])
            for key, val in merge_dict.items():
                hydra_cfg_dict[key] = val
            hydra_cfg = OmegaConf.create(hydra_cfg_dict)
        return hydra_cfg

    def load_checkpoint(self, path: str, checkpoint_idx: int) -> "BaseState":
        """Load a checkpoint from a file.

        Args:
            path: Path to the checkpoint file
            checkpoint_idx: Index of the checkpoint to load

        Returns:
            BaseState: Loaded model state
        """

        with h5py.File(path, "r") as h5:
            ckpt_names = list(h5["checkpoint"].keys())
            sorted_ckpts = np.argsort([int(x.split("id_")[1]) for x in ckpt_names])
            ckpt_idx = sorted_ckpts[checkpoint_idx]
            group = f"/checkpoint/{ckpt_names[ckpt_idx]}/state"

        dd_data = hdf5io_load(path, group=group)
        return self._load_from_dict(dd_data)

    def save_checkpoint(self, path: str, checkpoint_idx: int, compression="default") -> None:
        """Save the current model state to a hdf5 checkpoint file.

        Args:
            path: Path to the checkpoint file
            hydra_cfg: MainConfig object used to run the main script
            checkpoint_idx: Index of the checkpoint to save
            compression: Compression method to use
        """
        state_data = self._get_state_dict()
        save_h5_checkpoint(path, state_data, self.cfg, checkpoint_idx, compression)

    def _load_from_dict(self, data: Dict[str, bytes]) -> "BaseState":
        """Load the model state from a dictionary."""
        kwargs = {}
        for name, value_bytes in data.items():
            try:
                obj = getattr(self, name)
                value = deserialize_flax(obj, value_bytes)
                kwargs[name] = value
            except:
                print_yellow(f"Warning: could not load attribute {name}!")

        loaded_state = self.replace(**kwargs)
        print_green(f"Loaded state.{list(kwargs.keys())} from checkpoint...")
        return loaded_state

    def _get_state_dict(self) -> Dict[str, bytes]:
        """Save the model state to a dictionary."""
        state_dict = {}
        for field in fields(self):
            if field.metadata.get("pytree_node", True):
                name = field.name
                value = getattr(self, name)
                value_bytes = serialize_flax(value)
                state_dict[name] = value_bytes
        return state_dict


def deserialize_flax(flax_obj, msgpack_arr: jnp.ndarray):
    return from_bytes(flax_obj, zlib.decompress(msgpack_arr.tobytes()))


def serialize_flax(flax_obj):
    return jnp.frombuffer(zlib.compress(to_bytes(flax_obj)), dtype=jnp.uint8)


def get_mutable(variables: VariableDict):
    assert isinstance(variables, dict)
    mutables = []
    for key in variables.keys():
        if key != "params":
            assert key in EXPECTED_MUTABLE_KEYS
            mutables.append(key)
    return mutables


def get_model_variables_and_mutable(variables: VariableDict, model_name: str):
    assert "params" in variables.keys()
    assert model_name in variables["params"].keys()

    model_variables = {"params": variables["params"][model_name]}
    mutable = []
    for key in variables.keys():
        if key != "params":
            assert key in EXPECTED_MUTABLE_KEYS
            if model_name in variables[key]:
                model_variables[key] = variables[key][model_name]
                mutable.append(key)

    return model_variables, mutable


def combine_mutable_updates(model_mutable_update_dict: dict, cur_mutable_updates: dict, prefix=""):
    for model_name, mutable_dicts in model_mutable_update_dict.items():
        for mutable_key in mutable_dicts:
            assert mutable_key in EXPECTED_MUTABLE_KEYS
            if mutable_key not in cur_mutable_updates.keys():
                cur_mutable_updates[mutable_key] = {}

            if prefix:
                if prefix not in cur_mutable_updates[mutable_key].keys():
                    cur_mutable_updates[mutable_key][prefix] = {}
                cur_mutable_updates[mutable_key][prefix][model_name] = mutable_dicts[mutable_key]
            else:
                cur_mutable_updates[mutable_key][model_name] = mutable_dicts[mutable_key]

    return cur_mutable_updates


def make_optimizer(cfg: dict):
    opt_type = cfg.pop("type")

    # Remove other (non-optimizer-based) params
    other = cfg.pop("other", {})
    grad_clip = other.pop("grad_clip", None)
    global_grad_clip = other.pop("global_grad_clip", None)
    grad_accum_steps = other.pop("grad_accum_steps", None)
    apply_if_finite = other.pop("apply_if_finite", None)
    cfg.pop("param_counts", None)  # if loading from checkpoint

    lr_scheduler = create_scheduler(cfg.pop("lr_scheduler"))
    opt_cls: Type[optax.GradientTransformation] = eval(f"optax." + opt_type)
    opt = opt_cls(learning_rate=lr_scheduler, **cfg)

    if grad_clip is not None and grad_clip > 0:
        assert global_grad_clip is None, "Both can't be used"
        opt = optax.chain(optax.adaptive_grad_clip(grad_clip), opt)

    if global_grad_clip is not None and global_grad_clip > 0:
        opt = optax.chain(optax.clip_by_global_norm(global_grad_clip), opt)

    if grad_accum_steps is not None and grad_accum_steps > 1:
        opt = optax.MultiSteps(opt, every_k_schedule=grad_accum_steps)

    if apply_if_finite is not None and apply_if_finite > 0:
        opt = optax.apply_if_finite(opt, apply_if_finite)

    return opt, lr_scheduler, grad_accum_steps


def build_params_tree_vebose(params, prefix="", indent=""):
    """
    This is the simplest way to print out the param tree (very verbose though!)
    """
    lines = []

    if isinstance(params, dict):
        items = list(params.items())
        for i, (k, v) in enumerate(items):
            is_last = i == len(items) - 1
            branch = "└── " if is_last else "├── "
            lines.append(f"{indent}{branch}{k}")
            new_indent = indent + ("    " if is_last else "│   ")
            lines.extend(build_params_tree_vebose(v, prefix + k + "/", new_indent))

    elif isinstance(params, (list, tuple)):
        for i, v in enumerate(params):
            is_last = i == len(params) - 1
            branch = "└── " if is_last else "├── "
            lines.append(f"{indent}{branch}[{i}]")
            new_indent = indent + ("    " if is_last else "│   ")
            lines.extend(build_params_tree_vebose(v, prefix + f"{i}/", new_indent))

    else:
        # Leaf node
        lines.append(f"{indent}└── shape={params.shape}, size={params.size}")

    return lines


def build_params_tree(params, prefix="", indent="", depth=0, max_depth=None, max_series=3):
    """
    Pretty-prints a parameter tree with shapes/sizes inline.
    Groups kernel/bias/scale on the same line if they are the only children.
    """

    def format_shape_info(v: jnp.ndarray):
        """Return a formatted string for objects with shape and size attributes."""
        if hasattr(v, "shape") and hasattr(v, "size"):
            return str(v.shape).replace("(", "").replace(")", "")
        return str(v)  # fallback if no shape/size

    lines = []

    if max_depth is not None and depth > max_depth:
        lines.append(f"{indent}└── ... (depth limit reached)")
        return lines

    if isinstance(params, dict):
        items = list(params.items())
        grouped_items = []
        i = 0

        # Collapse numbered sequences
        while i < len(items):
            key, val = items[i]
            m = re.match(r"^(.*?)(\d+)$", key)
            if m:
                base = m.group(1)
                series = [(key, val)]
                j = i + 1
                while j < len(items):
                    km, vm = items[j]
                    mm = re.match(rf"^{re.escape(base)}(\d+)$", km)
                    if mm:
                        series.append((km, vm))
                        j += 1
                    else:
                        break
                if len(series) > max_series:
                    grouped_items.append(series[0])
                    grouped_items.append((f"... {series[1][0]} ... {series[-2][0]}", None))
                    grouped_items.append(series[-1])
                else:
                    grouped_items.extend(series)
                i = j
            else:
                grouped_items.append((key, val))
                i += 1

        for idx, (k, v) in enumerate(grouped_items):
            is_last = idx == len(grouped_items) - 1
            branch = "└── " if is_last else "├── "

            # Group kernel/bias/scale children into one line
            if isinstance(v, dict) and set(v.keys()).issubset({"kernel", "bias", "scale"}):
                parts = []
                for subk, subv in v.items():
                    parts.append(f"{subk}({format_shape_info(subv)})")
                lines.append(f"{indent}{branch}{k}: {' + '.join(parts)}")
            elif hasattr(v, "shape") and hasattr(v, "size"):
                lines.append(f"{indent}{branch}{k}: {format_shape_info(v)}")
            elif v is None:
                lines.append(f"{indent}{branch}{k}")
            else:
                lines.append(f"{indent}{branch}{k}")
                new_indent = indent + ("    " if is_last else "│   ")
                lines.extend(
                    build_params_tree(v, prefix + k + "/", new_indent, depth + 1, max_depth, max_series)
                )

    elif isinstance(params, (list, tuple)):
        for i, v in enumerate(params):
            is_last = i == len(params) - 1
            branch = "└── " if is_last else "├── "
            if hasattr(v, "shape") and hasattr(v, "size"):
                lines.append(f"{indent}{branch}[{i}]: {format_shape_info(v)}")
            else:
                lines.append(f"{indent}{branch}[{i}]")
                new_indent = indent + ("    " if is_last else "│   ")
                lines.extend(
                    build_params_tree(v, prefix + f"{i}/", new_indent, depth + 1, max_depth, max_series)
                )

    else:
        lines.append(f"{indent}└── {params}")

    return lines


def params_tree_string(params):
    tree_lines = build_params_tree(params)
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    tree_lines.append(f"\nTotal parameters: {total_params:,}")
    return "\n".join(tree_lines)
