import jax
import jax.numpy as jnp
from flax import struct
from flax.typing import VariableDict
from omegaconf import open_dict

from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.base.base_model import BaseModelLossArgs, BaseState
from jadex.base.base_state import get_mutable
from jadex.networks.variational.constants import LATENT, X
from jadex.utils.printing import print_jit_str


@struct.dataclass
class VAELossArgs(BaseModelLossArgs):
    x_data: jnp.ndarray
    step: int


class VAEState(BaseState):
    pass


class VAEModel(BaseVAEModel):
    """Standard VAE Implementation"""

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

        state = VAEState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    def loss_fn(self, variables: VariableDict, loss_args: VAELossArgs, rng_key, train=True):
        ##### VAE Loss Function #####
        x, mutable_updates = self.package_x(loss_args.x_data, variables, mutable=True)
        beta = self.schedulers["beta"](loss_args.step)

        ##### Forward Models #####
        q_key, p_key = jax.random.split(rng_key)
        q_z_given_x_params, mutable_updates["recognition_model"] = self.recognition_model.apply(
            variables["recognition_model"],
            {X: x},
            rngs={"dropout": q_key},
            train=train,
            mutable=get_mutable(variables["recognition_model"]),
        )

        z_given_x = self.latent_dist.reparameterize(q_z_given_x_params, rng_key)

        p_x_given_z_params, mutable_updates["generative_model"] = self.generative_model.apply(
            variables["generative_model"],
            {LATENT: z_given_x},
            rngs={"dropout": p_key},
            train=train,
            mutable=get_mutable(variables["generative_model"]),
        )

        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        elbo, latent_prior_kl = self.compute_elbo(q_z_given_x_params, log_p_x_given_z, return_kl=True)
        loss = -jnp.mean(log_p_x_given_z - beta * latent_prior_kl)

        metrics = dict(
            loss=loss,
            elbo=elbo,
            latent_prior_kl=latent_prior_kl,
            train_x_hats=self.x_dist.get_expected_value(p_x_given_z_params),
            mutable_updates=mutable_updates,
            state_updates={},
        )

        return metrics

    def get_loss_args(self, state: VAEState, batch, rng_key):
        loss_args = VAELossArgs(x_data=batch[X], step=state.step)
        return loss_args

    def get_predictions(self, state: VAEState, batch, rng_key):
        x = self.package_x(batch[X], state.variables)

        q_z_given_x_params = self.recognition_model.apply(
            state.variables["recognition_model"], {X: x}, train=False
        )

        # Take the mean value for predictions
        z_given_x = self.latent_dist.package_sample(self.latent_dist.get_expected_value(q_z_given_x_params))

        p_x_given_z_params = self.generative_model.apply(
            state.variables["generative_model"], {LATENT: z_given_x}, train=False
        )

        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        elbo, latent_prior_kl = self.compute_elbo(q_z_given_x_params, log_p_x_given_z, return_kl=True)
        x_hats = self.x_dist.get_expected_value(p_x_given_z_params)

        metrics = dict(
            elbo=elbo,
            latent_prior_kl=latent_prior_kl,
            log_p_x_given_z=log_p_x_given_z,
            x_hats=x_hats,
        )

        return metrics

    @classmethod
    def set_model_bits(cls, cfg):
        num_bits = cfg.model.get("num_bits")
        embed_dim = cfg.model.get("embed_dim")

        # Case 1: num_bits is provided, but embed_dim is not
        if num_bits is not None and embed_dim is None:
            # Ensure that num_bits is a multiple of 32
            assert num_bits % 32 == 0, "num_bits must be a multiple of 32 for VAE"
            with open_dict(cfg):
                cfg.model.embed_dim = int(num_bits / 32)
            print_jit_str(f"Calculated embed_dim: {cfg.model.embed_dim}")

        # Case 2: embed_dim is provided, but num_bits is not
        elif embed_dim is not None and num_bits is None:
            with open_dict(cfg):
                cfg.model.num_bits = embed_dim * 32
            print_jit_str(f"Calculated num_bits: {cfg.model.num_bits}")

        # Case 3: Both num_bits and embed_dim are provided, check for consistency
        elif num_bits is not None and embed_dim is not None:
            calculated_bits = embed_dim * 32
            if num_bits != calculated_bits:
                print(
                    f"WARNING: {num_bits} bits does not match {embed_dim} latent dimensions. Changing num_bits to {calculated_bits}..."
                )
                with open_dict(cfg):
                    cfg.model.num_bits = calculated_bits

        # Case 4: Neither num_bits nor embed_dim is provided
        else:
            raise ValueError("You must provide either num_bits or embed_dim for VAE.")

        # Final message showing the setup
        print_jit_str(
            f"Setting up: {cfg.model.embed_dim} latent dimensions and {cfg.model.num_bits} bits for VAE\n"
        )
