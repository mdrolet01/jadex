import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import struct
from flax.typing import VariableDict

from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.base.base_model import BaseModelLossArgs, BaseState
from jadex.base.base_state import get_mutable
from jadex.distributions.base_distribution import DistParams, Sample
from jadex.networks.variational.constants import LATENT, X


@struct.dataclass
class GumbelLossArgs(BaseModelLossArgs):
    x_data: jnp.ndarray
    step: int


class GumbelState(BaseState):
    pass


class GumbelModel(BaseVAEModel):
    """Gumbel Softmax VAE model implementation."""

    def init(self, rng_key, **kwargs):
        init_key, state_key = jax.random.split(rng_key)
        x = self.x_dist.create_sample((self.cfg.train.batch_size,))
        z_given_x = self.latent_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            recognition_model=self.recognition_model.init(init_key, {LATENT: z_given_x, X: x}, train=True),
            generative_model=self.generative_model.init(init_key, {LATENT: z_given_x}, train=True),
            scalers={},
        )

        if self.cfg.dataset.scaler_mode == "online":
            variables["scalers"][X] = self.scalers[X].init(init_key, x.value)

        state = GumbelState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    def compute_loss(
        self,
        q_z_given_x_params: DistParams,
        z_given_x: Sample,
        p_x_given_z_params: DistParams,
        x: Sample,
        beta: float,
    ):
        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        log_q_z_given_x = self.latent_dist.log_prob(q_z_given_x_params, z_given_x)
        elbo, latent_prior_kl = self.compute_elbo(q_z_given_x_params, log_p_x_given_z, return_kl=True)
        beta_elbo = self.compute_beta_elbo(log_p_x_given_z, latent_prior_kl)
        loss = -jnp.mean(log_p_x_given_z - beta * latent_prior_kl)

        loss_metrics = dict(
            elbo=elbo,
            beta_elbo=beta_elbo,
            beta=beta,
            latent_prior_kl=latent_prior_kl,
            log_p_x_given_z=log_p_x_given_z,
            log_q_z_given_x=log_q_z_given_x,
        )

        return loss, loss_metrics

    def forward_models(
        self,
        variables: VariableDict,
        x: Sample,
        temperature: float,
        rng_key: PRNGKey,
        train: bool,
        mutable_updates: dict,
        generative_condition: dict[str, Sample] = {},
    ):
        q_key, p_key = jax.random.split(rng_key)
        z_given_x, q_z_given_x_params, mutable_updates["recognition_model"] = (
            self.recognition_model.sample_autoregressive(
                variables["recognition_model"],
                inputs={X: x},
                temperature=temperature,
                train=train,
                rng_key=q_key,
            )
        )

        p_x_given_z_params, mutable_updates["generative_model"] = self.generative_model.apply(
            variables["generative_model"],
            {LATENT: z_given_x, **generative_condition},
            rngs={"dropout": p_key},
            train=train,
            mutable=get_mutable(variables["generative_model"]),
        )

        return q_z_given_x_params, z_given_x, p_x_given_z_params, mutable_updates

    def loss_fn(self, variables: VariableDict, loss_args: GumbelLossArgs, rng_key, train=True):
        ##### Gumbel Loss Function #####
        temperature = self.schedulers["temperature"](loss_args.step)
        x, mutable_updates = self.package_x(loss_args.x_data, variables, mutable=True)

        q_z_given_x_params, z_given_x, p_x_given_z_params, mutable_updates = self.forward_models(
            variables=variables,
            x=x,
            temperature=temperature,
            rng_key=rng_key,
            train=train,
            mutable_updates=mutable_updates,
        )

        loss, loss_metrics = self.compute_loss(
            q_z_given_x_params=q_z_given_x_params,
            z_given_x=z_given_x,
            p_x_given_z_params=p_x_given_z_params,
            x=x,
            beta=self.schedulers["beta"](loss_args.step),
        )

        metrics = dict(
            loss=loss,
            temperature=temperature,
            train_x_hats=self.x_dist.get_expected_value(p_x_given_z_params),
            mutable_updates=mutable_updates,
            state_updates={},
            **loss_metrics,
        )

        return metrics

    def get_loss_args(self, state: GumbelState, batch, rng_key):
        loss_args = GumbelLossArgs(x_data=batch[X], step=state.step)
        return loss_args

    def encode(self, state: GumbelState, x: Sample, rng_key, return_params=False):
        z_given_x, q_z_given_x_params, _ = self.recognition_model.sample_autoregressive(
            variables=state.variables["recognition_model"],
            inputs={X: x},
            temperature=None,
            train=False,
            rng_key=rng_key,
        )

        if return_params:
            return z_given_x, q_z_given_x_params

        return z_given_x

    def get_predictions(self, state: GumbelState, batch, rng_key):
        x = self.x_dist.package_sample(self.apply_scaler(batch[X], state.scaler_vars, X))

        z_given_x, q_z_given_x_params = self.encode(state, x, rng_key, return_params=True)

        p_x_given_z_params = self.generative_model.apply(
            state.variables["generative_model"], {LATENT: z_given_x}, train=False
        )

        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        log_q_z_given_x = self.latent_dist.log_prob(q_z_given_x_params, z_given_x)
        elbo, latent_prior_kl = self.compute_elbo(q_z_given_x_params, log_p_x_given_z, return_kl=True)
        beta_elbo = self.compute_beta_elbo(log_p_x_given_z, latent_prior_kl)
        x_hats = self.x_dist.get_expected_value(p_x_given_z_params)

        metrics = dict(
            elbo=elbo,
            beta_elbo=beta_elbo,
            latent_prior_kl=latent_prior_kl,
            log_p_x_given_z=log_p_x_given_z,
            log_q_z_given_x=log_q_z_given_x,
            x_hats=x_hats,
        )

        return metrics


class GRMCKModel(GumbelModel):
    """
    Any GRMCK-specific overrides go here
    """

    pass


##### Non-Autoregressive Models #####


class GumbelNonAutoregressiveModel(GumbelModel):
    """Non-Autoregressive version of Gumbel Softmax (Gumbel-NA)"""

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

        state = GumbelState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    def forward_models(
        self,
        variables: VariableDict,
        x: Sample,
        temperature: float,
        rng_key: PRNGKey,
        train: bool,
        mutable_updates: dict,
        generative_condition: dict[str, Sample] = {},
    ):
        q_key, z_key, p_key = jax.random.split(rng_key, 3)
        q_z_given_x_params, mutable_updates["recognition_model"] = self.recognition_model.apply(
            variables["recognition_model"],
            {X: x},
            train=train,
            rngs={"dropout": q_key},
            mutable=get_mutable(variables["recognition_model"]),
        )

        z_given_x = self.latent_dist.sample(
            q_z_given_x_params, leading_shape=(), temperature=temperature, rng_key=z_key
        )

        p_x_given_z_params, mutable_updates["generative_model"] = self.generative_model.apply(
            variables["generative_model"],
            {LATENT: z_given_x, **generative_condition},
            rngs={"dropout": p_key},
            train=train,
            mutable=get_mutable(variables["generative_model"]),
        )

        return q_z_given_x_params, z_given_x, p_x_given_z_params, mutable_updates

    def encode(self, state: GumbelState, x: Sample, rng_key, return_params=False):
        q_z_given_x_params = self.recognition_model.apply(
            state.variables["recognition_model"], {X: x}, train=False
        )

        z_given_x = self.latent_dist.sample(
            q_z_given_x_params, leading_shape=(), temperature=None, rng_key=rng_key
        )

        if return_params:
            return z_given_x, q_z_given_x_params

        return z_given_x


class GRMCKNonAutoregressiveModel(GumbelNonAutoregressiveModel):
    """
    Non-Autoregressive version of GRMCK (GRMCK-NA).
    Any GRMCK-specific overrides go here
    """

    pass
