import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from flax.typing import VariableDict
from jax.lax import stop_gradient as sg
from omegaconf import open_dict

from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.base.base_model import BaseModelLossArgs, BaseState
from jadex.base.base_state import get_mutable
from jadex.distributions.base_distribution import Sample
from jadex.networks.variational.constants import LATENT, X
from jadex.utils.printing import print_jit_str


@struct.dataclass
class FSQLossArgs(BaseModelLossArgs):
    x_data: jnp.ndarray


class FSQState(BaseState):
    pass


class FSQModel(BaseVAEModel):

    def init(self, rng_key, **kwargs):
        init_key, state_key = jax.random.split(rng_key)
        x = self.x_dist.create_sample((self.cfg.train.batch_size,))
        z_given_x = self.latent_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            recognition_model=self.recognition_model.init(init_key, {X: x}, train=True),
            generative_model=self.generative_model.init(init_key, {LATENT: z_given_x}, train=True),
            scalers={},
        )

        if self.cfg.dataset.scaler_mode == "online":
            variables["scalers"][X] = self.scalers[X].init(init_key, x.value)

        state = FSQState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    @staticmethod
    def round_ste(z: jnp.ndarray):
        """Round with straight through gradients."""
        zhat = jnp.round(z)
        return z + sg(zhat - z)

    @property
    def _levels_np(self):
        return np.array(self.cfg.model.fsq_levels)

    def bound(self, z):
        """Bound z, an array of shape (..., d)."""
        eps = 1e-3
        half_l = (self._levels_np - 1) * (1 - eps) / 2
        offset = jnp.where(self._levels_np % 2 == 1, 0.0, 0.5)
        shift = jnp.tan(offset / half_l)
        return jnp.tanh(z + shift) * half_l - offset

    def quantize_ste(self, z_e: jnp.ndarray):
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z_e))
        # Renormalize to [-1, 1]
        half_width = self._levels_np // 2
        quantized = quantized / half_width
        return quantized

    def loss_fn(self, variables: VariableDict, loss_args: FSQLossArgs, rng_key, train=True):
        ##### FSQ Loss Function #####
        x, mutable_updates = self.package_x(loss_args.x_data, variables, mutable=True)

        # Forward recognition model
        q_key, p_key = jax.random.split(rng_key)
        q_z_given_x_params, mutable_updates["recognition_model"] = self.recognition_model.apply(
            variables["recognition_model"],
            {X: x},
            rngs={"dropout": q_key},
            train=train,
            mutable=get_mutable(variables["recognition_model"]),
        )

        # Quantize latents and apply straight through estimator
        z_e = self.latent_dist.get_expected_value(q_z_given_x_params)
        z_value = self.quantize_ste(z_e)
        z = self.latent_dist.package_sample(z_value)

        # Forward generative model
        p_x_given_z_params, mutable_updates["generative_model"] = self.generative_model.apply(
            variables["generative_model"],
            {LATENT: z},
            rngs={"dropout": p_key},
            train=train,
            mutable=get_mutable(variables["generative_model"]),
        )

        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        reconstruction_loss = -jnp.mean(log_p_x_given_z / self.x_dist.size)

        metrics = dict(
            loss=reconstruction_loss,
            train_x_hats=self.x_dist.get_expected_value(p_x_given_z_params),
            mutable_updates=mutable_updates,
            reconstruction_loss=reconstruction_loss,
            log_p_x_given_z=log_p_x_given_z,
            state_updates={},
        )

        return metrics

    def get_loss_args(self, state: FSQState, batch, rng_key):
        loss_args = FSQLossArgs(x_data=batch[X])
        return loss_args

    def encode(self, state, x, rng_key=None):
        # Forward recognition model
        q_z_given_x_params = self.recognition_model.apply(
            state.variables["recognition_model"], {X: x}, train=False
        )
        z_e = self.latent_dist.get_expected_value(q_z_given_x_params)
        z_value = self.quantize_ste(z_e)
        z = self.latent_dist.package_sample(z_value)
        return z

    def encode_index(self, state: BaseState, x: Sample, rng_key=None) -> Sample:
        q_z_given_x_params = self.recognition_model.apply(
            state.variables["recognition_model"], {X: x}, train=False
        )
        z_e = self.latent_dist.get_expected_value(q_z_given_x_params)

        max_levels = self._levels_np
        half_width = max_levels // 2
        centered_codes = jnp.round(self.bound(z_e)).astype(jnp.int32)
        codes = centered_codes + half_width

        weights = np.concatenate([np.cumprod(max_levels[::-1])[::-1][1:], np.array([1])])
        vocab_idxs = (codes * weights).sum(axis=-1)
        return vocab_idxs

    def get_predictions(self, state: FSQState, batch, rng_key=None):
        x = self.package_x(batch[X], state.variables)

        z = self.encode(state, x)

        # Forward generative model
        p_x_given_z_params = self.generative_model.apply(
            state.variables["generative_model"], {LATENT: z}, train=False
        )

        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        x_hats = self.x_dist.get_expected_value(p_x_given_z_params)

        metrics = dict(x_hats=x_hats, log_p_x_given_z=log_p_x_given_z)

        return metrics

    @classmethod
    def set_model_bits(cls, cfg):
        fsq_levels = cfg.model.fsq_levels
        num_bits = cfg.model.get("num_bits")
        block_size = cfg.model.block_size

        true_num_bits = int(block_size * np.log2(np.prod(fsq_levels)))

        # Check consistency
        if num_bits is not None and true_num_bits != num_bits:
            print_jit_str(
                f"WARNING! FSQ has {true_num_bits} bits. This does not match the requested number of bits: {num_bits}."
            )

        with open_dict(cfg):
            cfg.model.fsq_embed_dim = len(fsq_levels)
            cfg.model.num_bits = true_num_bits

        # Final setup message
        print_jit_str(
            f"Setting up: {cfg.model.block_size} blocks, {cfg.model.fsq_levels} fsq levels, "
            + f"and {cfg.model.num_bits} bits for {cfg.model.name}\n"
        )
