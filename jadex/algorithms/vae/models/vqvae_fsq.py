from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey
from flax import struct
from flax.typing import VariableDict
from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.base.base_model import BaseModelLossArgs, BaseState
from jadex.base.base_state import get_mutable
from jadex.distributions.base_distribution import DistParams, Sample
from jadex.networks.variational.constants import LATENT, X
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit_str
from jax.lax import stop_gradient as sg
from omegaconf import open_dict


@struct.dataclass
class VQVAELossArgs(BaseModelLossArgs):
    x_data: jnp.ndarray


class VQVAEState(BaseState):
    pass


class VQVAEModel(BaseVAEModel):
    """Vector Quantized VAE (VQ-VAE) model implementation."""

    codebook: Optional[nn.Embed] = non_pytree()

    @staticmethod
    def get_quantizer_type():
        return "vector"

    @classmethod
    def create_model_kwargs(cls, cfg, networks, dists):
        base_kwargs = BaseVAEModel.create_model_kwargs(cfg, networks, dists)

        quantizer_type = cls.get_quantizer_type()
        if quantizer_type == "fsq":
            base_kwargs["codebook"] = None
        elif quantizer_type == "vector":
            base_kwargs["codebook"] = nn.Embed(cfg.model.vocab_size, cfg.model.embed_dim)
        else:
            raise ValueError

        return base_kwargs

    def init(self, rng_key, **kwargs):
        init_key, state_key = jax.random.split(rng_key)
        x = self.x_dist.create_sample((self.cfg.train.batch_size,))
        z_given_x = self.latent_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            recognition_model=self.recognition_model.init(init_key, {X: x}, train=True),
            generative_model=self.generative_model.init(init_key, {LATENT: z_given_x}, train=True),
            scalers={},
        )

        if self.get_quantizer_type() == "vector":
            variables["codebook"] = self.codebook.init(
                init_key, jnp.zeros((self.cfg.train.batch_size,), dtype=int)
            )

        if self.cfg.dataset.scaler_mode == "online":
            variables["scalers"][X] = self.scalers[X].init(init_key, x.value)

        state = VQVAEState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    def compute_loss(self, z_q: jnp.ndarray, z_e: jnp.ndarray, p_x_given_z_params: DistParams, x: Sample):
        # NOTE: The joint pdf is the sum over all pixels/units, so divide by the size to make comparable in loss
        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        loss_terms = dict(
            codebook_loss=jnp.mean((z_q - sg(z_e)) ** 2),
            commitment_loss=self.cfg.model.commitment_coeff * jnp.mean((sg(z_q) - z_e) ** 2),
            reconstruction_loss=-jnp.mean(log_p_x_given_z / self.x_dist.size),
        )
        loss = sum(loss_terms.values())
        loss_metrics = dict(**loss_terms, log_p_x_given_z=log_p_x_given_z)
        return loss, loss_metrics

    def forward_models(
        self,
        variables: VariableDict,
        x: Sample,
        rng_key: PRNGKey,
        train: bool,
        mutable_updates: dict,
        generative_condition: dict[str, Sample] = {},
    ):
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
        z_q, _ = self.quantize(variables.get("codebook", None), z_e)
        z = self.latent_dist.package_sample(z_e + sg(z_q - z_e))

        # Forward generative model
        p_x_given_z_params, mutable_updates["generative_model"] = self.generative_model.apply(
            variables["generative_model"],
            {LATENT: z, **generative_condition},
            rngs={"dropout": p_key},
            train=train,
            mutable=get_mutable(variables["generative_model"]),
        )

        return z_q, z_e, p_x_given_z_params, mutable_updates

    def loss_fn(self, variables: VariableDict, loss_args: VQVAELossArgs, rng_key, train=True):
        ##### VQVAE Loss Function #####
        x, mutable_updates = self.package_x(loss_args.x_data, variables, mutable=True)

        z_q, z_e, p_x_given_z_params, mutable_updates = self.forward_models(
            variables=variables, x=x, rng_key=rng_key, train=train, mutable_updates=mutable_updates
        )

        loss, loss_metrics = self.compute_loss(z_q=z_q, z_e=z_e, p_x_given_z_params=p_x_given_z_params, x=x)

        metrics = dict(
            loss=loss,
            train_x_hats=self.x_dist.get_expected_value(p_x_given_z_params),
            mutable_updates=mutable_updates,
            state_updates={},
            **loss_metrics,
        )

        return metrics

    def get_loss_args(self, state: VQVAEState, batch, rng_key):
        loss_args = VQVAELossArgs(x_data=batch[X])
        return loss_args

    def get_predictions(self, state: VQVAEState, batch, rng_key):
        x = self.x_dist.package_sample(self.apply_scaler(batch[X], state.scaler_vars, X))

        # Forward recognition model
        q_z_given_x_params, _ = self.recognition_model.apply(
            state.variables["recognition_model"],
            {X: x},
            train=False,
            mutable=get_mutable(state.variables["recognition_model"]),
        )
        z_e = self.latent_dist.get_expected_value(q_z_given_x_params)

        # Quantize latents
        z_q, _ = self.quantize(state.variables.get("codebook", None), z_e)
        z_q = self.latent_dist.package_sample(z_q)

        # Forward generative model
        p_x_given_z_params, _ = self.generative_model.apply(
            state.variables["generative_model"],
            {LATENT: z_q},
            train=False,
            mutable=get_mutable(state.variables["generative_model"]),
        )

        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        x_hats = self.x_dist.get_expected_value(p_x_given_z_params)

        metrics = dict(x_hats=x_hats, log_p_x_given_z=log_p_x_given_z)

        return metrics

    def encode(self, state, x, rng_key):
        # Forward recognition model
        q_z_given_x_params, _ = self.recognition_model.apply(
            state.variables["recognition_model"],
            {X: x},
            train=False,
            mutable=get_mutable(state.variables["recognition_model"]),
        )
        z_e = self.latent_dist.get_expected_value(q_z_given_x_params)

        # Quantize latents
        z_q, _ = self.quantize(state.variables.get("codebook", None), z_e)
        z_q = self.latent_dist.package_sample(z_q)
        return z_q

    def encode_index(self, state, x, rng_key):
        # Forward recognition model
        q_z_given_x_params, _ = self.recognition_model.apply(
            state.variables["recognition_model"],
            {X: x},
            train=False,
            mutable=get_mutable(state.variables["recognition_model"]),
        )
        z_e = self.latent_dist.get_expected_value(q_z_given_x_params)

        # Quantize latents
        z_q, closest_idxs = self.quantize(state.variables.get("codebook", None), z_e)
        return closest_idxs

    def quantize(self, params: VariableDict, z_e: jnp.ndarray):
        """Quantize continuous latents using the codebook.

        Args:
            z_e: Continuous latent encodings to quantize
            params: Params of embed table

        Returns:
            tuple: A tuple containing:
                - quantized_latents: Quantized latent vectors
                - quantized_idxs: Indices of closest codebook entries
        """

        # Get embeddings from codebook
        all_codebook_vecs = sg(self.codebook.apply(params, jnp.arange(self.cfg.model.vocab_size)))

        # Compute pairwise distances using the formula:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        distances = (
            jnp.sum(z_e**2, axis=-1, keepdims=True)
            + jnp.sum(all_codebook_vecs**2, axis=-1)
            - 2 * jnp.einsum("...td,...kd->...tk", z_e, all_codebook_vecs)
        )

        closest_idxs = jnp.argmin(distances, axis=-1)
        z_q = self.codebook.apply(params, closest_idxs)

        return z_q, closest_idxs


class FSQModel(VQVAEModel):
    """Finte Scalar Quantization (FSQ) model implementation."""

    @staticmethod
    def get_quantizer_type():
        return "fsq"

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

    def quantize(self, _, z_e: jnp.ndarray):
        """Quanitzes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z_e))
        # Renormalize to [-1, 1]
        half_width = self._levels_np // 2
        quantized = quantized / half_width
        return quantized, None

    def encode_index(self, state: BaseState, x: Sample, rng_key: PRNGKey) -> Sample:
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
