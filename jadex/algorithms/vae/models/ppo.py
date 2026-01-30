from copy import deepcopy

import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import struct
from flax.typing import VariableDict
from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.base.base_model import BaseModelLossArgs, BaseState
from jadex.base.base_state import get_mutable
from jadex.distributions.base_distribution import DistParams, Sample
from jadex.distributions.diagonal_gaussian import DiagonalGaussianParams
from jadex.networks.variational.baseline_models import BaselineModel
from jadex.networks.variational.constants import LATENT, X
from jadex.utils import non_pytree
from jax.lax import stop_gradient as sg


@struct.dataclass
class PPOLossArgs(BaseModelLossArgs):
    x_data: jnp.ndarray
    z_given_x: Sample
    old_recognition_variables: VariableDict
    step: int
    kl_coeff: float


class PPOState(BaseState):
    old_recognition_variables: VariableDict
    kl_coeff: float


class PPOModel(BaseVAEModel):
    """Proximal Policy Optimization (for VAE) model implementation."""

    baseline_model: BaselineModel = non_pytree()

    def init(self, rng_key):
        init_key, state_key = jax.random.split(rng_key, 2)
        x = self.x_dist.create_sample((self.cfg.train.batch_size,))
        z_given_x = self.latent_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            recognition_model=self.recognition_model.init(init_key, {LATENT: z_given_x, X: x}, train=True),
            generative_model=self.generative_model.init(init_key, {LATENT: z_given_x}, train=True),
            baseline_model=self.baseline_model.init(init_key, {X: x}, train=True),
            scalers={},
        )

        if self.cfg.dataset.scaler_mode == "online":
            variables["scalers"][X] = self.scalers[X].init(init_key, x.value)

        state = PPOState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
            old_recognition_variables=deepcopy(variables["recognition_model"]),
            kl_coeff=jnp.float32(10.0),
        )

        return state

    def compute_ppo_losses(
        self,
        q_z_given_x_params: DistParams,
        old_q_z_given_x_params: DistParams,
        z_given_x: Sample,
        log_p_x_given_z: jnp.ndarray,
        beta: float,
        kl_coeff: float,
        baseline: jnp.ndarray,
    ):
        ##### Compute objective quantities #####
        log_q_z_given_x = self.latent_dist.log_prob(q_z_given_x_params, z_given_x)
        old_log_q_z_given_x = self.latent_dist.log_prob(old_q_z_given_x_params, z_given_x)
        elbo, latent_prior_kl = self.compute_elbo(q_z_given_x_params, log_p_x_given_z, return_kl=True)
        beta_elbo = self.compute_beta_elbo(log_p_x_given_z, latent_prior_kl)

        ##### Fit baseline/critic #####
        rewards = sg(log_p_x_given_z)
        baseline_loss = jnp.square((rewards - baseline) / self.baseline_model.scale).mean()
        advantages = rewards - sg(baseline)

        if self.cfg.model.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ##### PPO surrogate loss #####
        iws = jnp.exp(log_q_z_given_x - old_log_q_z_given_x)
        clipped_iws = jnp.clip(iws, 1 - self.cfg.model.clip_eps, 1 + self.cfg.model.clip_eps)
        advantage_loss = -jnp.minimum(iws * advantages, clipped_iws * advantages)
        old_kl = self.latent_dist.kl(old_q_z_given_x_params, q_z_given_x_params)
        policy_loss = jnp.mean(advantage_loss + kl_coeff * old_kl + beta * latent_prior_kl)

        ppo_losses = dict(recognition_model=policy_loss, baseline_model=baseline_loss)

        metrics = dict(
            log_q_z_given_x=log_q_z_given_x,
            elbo=elbo,
            beta_elbo=beta_elbo,
            latent_prior_kl=latent_prior_kl,
            advantages=advantages,
            old_kl=old_kl,
            beta=beta,
            iws=iws,
            kl_coeff=kl_coeff,
            clip_frac=jnp.mean(jnp.abs(iws - 1) > self.cfg.model.clip_eps),
        )

        return ppo_losses, metrics

    def compute_generative_losses(self, p_x_given_z_params: DistParams, x: Sample):
        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        reconstruction_loss = -jnp.mean(log_p_x_given_z)
        generative_losses = dict(generative_model=reconstruction_loss)
        generative_metrics = dict(log_p_x_given_z=log_p_x_given_z)
        return generative_losses, generative_metrics

    def forward_models(
        self,
        variables: VariableDict,
        old_recognition_variables: VariableDict,
        z_given_x: Sample,
        x: Sample,
        rng_key: PRNGKey,
        train: bool,
        mutable_updates: dict,
        generative_condition: dict[str, Sample] = {},
    ):
        q_key, p_key, b_key = jax.random.split(rng_key, 3)

        q_z_given_x_params, mutable_updates["recognition_model"] = self.recognition_model.apply(
            variables["recognition_model"],
            {LATENT: z_given_x, X: x},
            rngs={"dropout": q_key},
            train=train,
            mutable=get_mutable(variables["recognition_model"]),
        )

        old_q_z_given_x_params = self.recognition_model.apply(
            old_recognition_variables, {LATENT: z_given_x, X: x}, train=False
        )

        p_x_given_z_params, mutable_updates["generative_model"] = self.generative_model.apply(
            variables["generative_model"],
            {LATENT: z_given_x, **generative_condition},
            train=train,
            rngs={"dropout": p_key},
            mutable=get_mutable(variables["generative_model"]),
        )

        baseline, mutable_updates["baseline_model"] = self.baseline_model.apply(
            variables["baseline_model"],
            {X: x},
            train=train,
            rngs={"dropout": b_key},
            mutable=get_mutable(variables["baseline_model"]),
        )

        return q_z_given_x_params, old_q_z_given_x_params, p_x_given_z_params, baseline, mutable_updates

    def update_kl_coeff(self, old_kl: float, cur_kl_coeff: float):
        new_kl_coeff = jax.lax.cond(
            old_kl < (self.cfg.model.old_kl.target / self.cfg.model.old_kl.scale),
            lambda: cur_kl_coeff / self.cfg.model.old_kl.gain,
            lambda: jax.lax.cond(
                old_kl > (self.cfg.model.old_kl.target * self.cfg.model.old_kl.scale),
                lambda: cur_kl_coeff * self.cfg.model.old_kl.gain,
                lambda: cur_kl_coeff,
            ),
        )
        new_kl_coeff = jnp.clip(
            new_kl_coeff, self.cfg.model.old_kl.min_coeff, self.cfg.model.old_kl.max_coeff
        )
        return new_kl_coeff

    def loss_fn(self, variables: VariableDict, loss_args: PPOLossArgs, rng_key, train=True):
        ##### PPO Loss Function #####
        x, mutable_updates = self.package_x(loss_args.x_data, variables, mutable=True)

        q_z_given_x_params, old_q_z_given_x_params, p_x_given_z_params, baseline, mutable_updates = (
            self.forward_models(
                variables=variables,
                old_recognition_variables=loss_args.old_recognition_variables,
                z_given_x=loss_args.z_given_x,
                x=x,
                rng_key=rng_key,
                train=train,
                mutable_updates=mutable_updates,
            )
        )

        generative_losses, generative_metrics = self.compute_generative_losses(p_x_given_z_params, x)

        ppo_losses, ppo_metrics = self.compute_ppo_losses(
            q_z_given_x_params=q_z_given_x_params,
            old_q_z_given_x_params=old_q_z_given_x_params,
            z_given_x=loss_args.z_given_x,
            log_p_x_given_z=generative_metrics["log_p_x_given_z"],
            beta=self.schedulers["beta"](loss_args.step),
            kl_coeff=loss_args.kl_coeff,
            baseline=baseline,
        )

        losses = dict(**ppo_losses, **generative_losses)

        new_kl_coeff = self.update_kl_coeff(
            old_kl=ppo_metrics["old_kl"].mean(), cur_kl_coeff=loss_args.kl_coeff
        )

        metrics = dict(
            losses=losses,
            loss=sum(losses.values()),
            train_x_hats=self.x_dist.get_expected_value(p_x_given_z_params),
            mutable_updates=mutable_updates,
            state_updates=dict(
                old_recognition_variables=deepcopy(variables["recognition_model"]), kl_coeff=new_kl_coeff
            ),
            **ppo_metrics,
            **generative_metrics,
        )

        if isinstance(p_x_given_z_params, DiagonalGaussianParams):
            metrics.update(
                dict(
                    mean_p_x_given_z_var=jnp.mean(p_x_given_z_params.variance),
                    median_p_x_given_z_var=jnp.median(p_x_given_z_params.variance),
                    min_p_x_given_z_var=jnp.min(p_x_given_z_params.variance),
                    max_p_x_given_z_var=jnp.max(p_x_given_z_params.variance),
                )
            )

        return metrics

    def get_loss_args(self, state: PPOState, batch: dict, rng_key):
        x = self.package_x(batch[X], state.variables)
        z_given_x = self.encode(state, x, rng_key, use_old=True)
        loss_args = PPOLossArgs(
            x_data=batch[X],
            z_given_x=z_given_x,
            step=state.step,
            old_recognition_variables=state.old_recognition_variables,
            kl_coeff=state.kl_coeff,
        )
        return loss_args

    def encode(self, state: PPOState, x: Sample, rng_key, return_params=False, use_old=False):
        if use_old:
            variables = state.old_recognition_variables
        else:
            variables = state.variables["recognition_model"]

        z_given_x, q_z_given_x_params, _ = self.recognition_model.sample_autoregressive(
            variables=variables,
            inputs={X: x},
            temperature=1.0,
            train=False,
            rng_key=rng_key,
        )

        if return_params:
            return z_given_x, q_z_given_x_params

        return z_given_x

    def get_predictions(self, state: PPOState, batch: dict, rng_key):
        x = self.package_x(batch[X], state.variables)
        z_given_x, q_z_given_x_params = self.encode(state, x, rng_key, return_params=True)

        p_x_given_z_params = self.generative_model.apply(
            state.variables["generative_model"], {LATENT: z_given_x}, train=False
        )

        log_p_x_given_z = self.x_dist.log_prob(p_x_given_z_params, x)
        elbo, latent_prior_kl = self.compute_elbo(q_z_given_x_params, log_p_x_given_z, return_kl=True)
        beta_elbo = self.compute_beta_elbo(log_p_x_given_z, latent_prior_kl)
        x_hats = self.x_dist.get_expected_value(p_x_given_z_params)

        metrics = dict(
            elbo=elbo,
            beta_elbo=beta_elbo,
            latent_prior_kl=latent_prior_kl,
            log_p_x_given_z=log_p_x_given_z,
            x_hats=x_hats,
        )

        return metrics
