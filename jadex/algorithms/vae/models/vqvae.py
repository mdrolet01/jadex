import flax.linen as nn
import jax
import jax.numpy as jnp
from flax import struct
from flax.typing import VariableDict
from jax.lax import stop_gradient as sg

from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.base.base_model import BaseModelLossArgs, BaseState
from jadex.base.base_state import get_mutable
from jadex.networks.variational.constants import LATENT, X
from jadex.utils import non_pytree


@struct.dataclass
class VQVAELossArgs(BaseModelLossArgs):
    x_data: jnp.ndarray


class VQVAEState(BaseState):
    pass


class VQVAEModel(BaseVAEModel):
    """Vector Quantized VAE (VQ-VAE) model implementation."""

    codebook: nn.Embed = non_pytree()

    @classmethod
    def create_model_kwargs(cls, cfg, networks, dists):
        base_kwargs = BaseVAEModel.create_model_kwargs(cfg, networks, dists)
        base_kwargs["codebook"] = nn.Embed(cfg.model.vocab_size, cfg.model.embed_dim)
        return base_kwargs

    def init(self, rng_key, **kwargs):
        init_key, state_key = jax.random.split(rng_key)
        x = self.x_dist.create_sample((self.cfg.train.batch_size,))
        z_given_x = self.latent_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            recognition_model=self.recognition_model.init(init_key, {X: x}, train=True),
            generative_model=self.generative_model.init(init_key, {LATENT: z_given_x}, train=True),
            codebook=self.codebook.init(init_key, jnp.zeros((self.cfg.train.batch_size,), dtype=int)),
            scalers={},
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

    def quantize(self, params: VariableDict, z_e: jnp.ndarray):
        """Quantize continuous latents using the codebook.

        Args:
            z_e: Continuous latent encodings to quantize
            params: Params of codebook

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

    def loss_fn(self, variables: VariableDict, loss_args: VQVAELossArgs, rng_key, train=True):
        ##### VQVAE Loss Function #####
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
        z_q, _ = self.quantize(variables["codebook"], z_e)
        z_value = z_e + sg(z_q - z_e)
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

        loss_terms = dict(
            codebook_loss=jnp.mean((z_q - sg(z_e)) ** 2),
            commitment_loss=self.cfg.model.commitment_coeff * jnp.mean((sg(z_q) - z_e) ** 2),
            reconstruction_loss=-jnp.mean(log_p_x_given_z / self.x_dist.size),
        )
        loss = sum(loss_terms.values())

        metrics = dict(
            loss=loss,
            train_x_hats=self.x_dist.get_expected_value(p_x_given_z_params),
            mutable_updates=mutable_updates,
            state_updates={},
            **loss_terms,
            log_p_x_given_z=log_p_x_given_z,
        )

        return metrics

    def get_loss_args(self, state: VQVAEState, batch, rng_key):
        loss_args = VQVAELossArgs(x_data=batch[X])
        return loss_args

    def encode(self, state, x, rng_key=None):
        # Forward recognition model
        q_z_given_x_params = self.recognition_model.apply(
            state.variables["recognition_model"], {X: x}, train=False
        )
        z_e = self.latent_dist.get_expected_value(q_z_given_x_params)
        # Quantize latents
        z_q, _ = self.quantize(state.variables["codebook"], z_e)
        z_q = self.latent_dist.package_sample(z_q)
        return z_q

    def encode_index(self, state, x, rng_key=None):
        # Forward recognition model
        q_z_given_x_params = self.recognition_model.apply(
            state.variables["recognition_model"], {X: x}, train=False
        )
        z_e = self.latent_dist.get_expected_value(q_z_given_x_params)
        # Quantize latents
        z_q, closest_idxs = self.quantize(state.variables["codebook"], z_e)
        return closest_idxs

    def get_predictions(self, state: VQVAEState, batch, rng_key=None):
        x = self.x_dist.package_sample(self.apply_scaler(batch[X], state.scaler_vars, X))

        z_q = self.encode(state, x)

        # Forward generative model
        p_x_given_z_params = self.generative_model.apply(
            state.variables["generative_model"], {LATENT: z_q}, train=False
        )

        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        x_hats = self.x_dist.get_expected_value(p_x_given_z_params)

        metrics = dict(x_hats=x_hats, log_p_x_given_z=log_p_x_given_z)

        return metrics
