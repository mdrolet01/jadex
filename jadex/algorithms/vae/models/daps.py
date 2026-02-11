import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax import struct
from flax.typing import VariableDict
from jax import vmap
from jax.lax import stop_gradient as sg
from jax.scipy.special import logsumexp

from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.base.base_model import BaseModelLossArgs, BaseState
from jadex.base.base_state import BaseState, get_mutable
from jadex.distributions.base_distribution import DistParams, Sample
from jadex.distributions.diagonal_gaussian import DiagonalGaussianParams
from jadex.networks.variational.constants import LATENT, X


@struct.dataclass
class DAPSLossArgs(BaseModelLossArgs):
    x_data: jnp.ndarray
    z_given_x: Sample
    step: int


class DAPSState(BaseState):
    pass


class DAPSModel(BaseVAEModel):
    """Discrete Autoencoding via Policy Search (DAPS) model implementation."""

    def init(self, rng_key):
        init_key, state_key = jax.random.split(rng_key, 2)
        x = self.x_dist.create_sample((self.cfg.train.batch_size,))
        z_given_x = self.latent_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            recognition_model=self.recognition_model.init(init_key, {LATENT: z_given_x, X: x}, train=True),
            generative_model=self.generative_model.init(init_key, {LATENT: z_given_x}, train=True),
            log_eta=dict(params=jnp.log(self.cfg.model.init_eta)),
            scalers={},
        )

        if self.cfg.dataset.scaler_mode == "online":
            variables["scalers"][X] = self.scalers[X].init(init_key, x.value)

        state = DAPSState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    def compute_policy_losses(
        self,
        q_z_given_x_params: DistParams,
        z_given_x: Sample,
        p_x_given_z_params: DistParams,
        x: Sample,
        log_eta: float,
        beta: float,
    ):
        ##### Compute objective quantities #####
        log_q_z_given_x = self.latent_dist.log_prob(q_z_given_x_params, z_given_x)
        log_p_x_given_z = vmap(self.x_dist.log_prob, (0, None))(p_x_given_z_params, x)
        elbo, latent_prior_kl = self.compute_elbo(q_z_given_x_params, log_p_x_given_z, return_kl=True)
        beta_elbo = self.compute_beta_elbo(log_p_x_given_z, latent_prior_kl)
        rewards = sg(log_p_x_given_z)

        if self.cfg.model.baseline == "mean":
            baseline = jnp.mean(rewards, axis=0)
        elif self.cfg.model.baseline == "logsumexp":
            baseline = logsumexp(rewards, axis=0)
        else:
            raise ValueError

        advantages = rewards - baseline

        ##### Get eta based on ESS #####
        eta = jnp.exp(log_eta)
        ess_target = self.cfg.model.num_value_samples * self.cfg.model.ess_target
        cur_ess = self.compute_ess(eta, sg(advantages), sg(beta), sg(log_q_z_given_x))
        eta_loss = jnp.square(cur_ess - ess_target).mean()

        ##### Weighted MLE #####
        unnormalized_log_qstar = (advantages + eta * sg(log_q_z_given_x)) / (eta + beta)
        unnormalized_iws = unnormalized_log_qstar - sg(log_q_z_given_x)
        iws = jnp.exp(unnormalized_iws - logsumexp(unnormalized_iws, axis=0, keepdims=True))
        weighted_likelihood = sg(iws) * log_q_z_given_x

        policy_loss = -jnp.sum(weighted_likelihood, axis=0).mean()
        policy_losses = dict(recognition_model=policy_loss, log_eta=eta_loss)

        metrics = dict(
            log_q_z_given_x=log_q_z_given_x,
            log_p_x_given_z=log_p_x_given_z,
            elbo=elbo,
            beta_elbo=beta_elbo,
            latent_prior_kl=latent_prior_kl,
            advantages=advantages,
            ess=cur_ess / self.cfg.model.num_value_samples,
            eta=eta,
            beta=beta,
        )

        return policy_losses, metrics

    def compute_generative_losses(self, p_x_given_z_params: DistParams, x: Sample):
        log_p_x_given_z = vmap(self.x_dist.log_prob, (0, None))(p_x_given_z_params, x)
        reconstruction_loss = -jnp.mean(log_p_x_given_z)
        generative_losses = dict(generative_model=reconstruction_loss)
        generative_metrics = {}
        return generative_losses, generative_metrics

    def forward_models(
        self,
        variables: VariableDict,
        z_given_x: Sample,
        x: Sample,
        rng_key: PRNGKey,
        train: bool,
        mutable_updates: dict,
        generative_condition: dict[str, Sample] = {},
    ):
        q_key, p_key = jax.random.split(rng_key)

        q_z_given_x_params, q_mutable = vmap(
            lambda z_: self.recognition_model.apply(
                variables["recognition_model"],
                {LATENT: z_, X: x},
                rngs={"dropout": q_key},
                train=train,
                mutable=get_mutable(variables["recognition_model"]),
            )
        )(z_given_x)

        p_x_given_z_params, p_mutable = vmap(
            lambda z_: self.generative_model.apply(
                variables["generative_model"],
                {LATENT: z_, **generative_condition},
                train=train,
                rngs={"dropout": p_key},
                mutable=get_mutable(variables["generative_model"]),
            )
        )(z_given_x)

        # Average mutable stats across monte carlo axis
        mutable_updates["generative_model"] = jax.tree.map(lambda x: jnp.mean(x, axis=0), p_mutable)
        mutable_updates["recognition_model"] = jax.tree.map(lambda x: jnp.mean(x, axis=0), q_mutable)

        return q_z_given_x_params, p_x_given_z_params, mutable_updates

    def loss_fn(self, variables: VariableDict, loss_args: DAPSLossArgs, rng_key, train=True):
        ##### DAPS Loss Function #####
        x, mutable_updates = self.package_x(loss_args.x_data, variables, mutable=True)

        q_z_given_x_params, p_x_given_z_params, mutable_updates = self.forward_models(
            variables=variables,
            z_given_x=loss_args.z_given_x,
            x=x,
            rng_key=rng_key,
            train=train,
            mutable_updates=mutable_updates,
        )

        policy_losses, policy_metrics = self.compute_policy_losses(
            q_z_given_x_params=q_z_given_x_params,
            z_given_x=loss_args.z_given_x,
            p_x_given_z_params=p_x_given_z_params,
            x=x,
            log_eta=variables["log_eta"]["params"],
            beta=self.schedulers["beta"](loss_args.step),
        )

        generative_losses, generative_metrics = self.compute_generative_losses(p_x_given_z_params, x)

        losses = dict(**policy_losses, **generative_losses)

        metrics = dict(
            losses=losses,
            loss=sum(losses.values()),
            train_x_hats=self.x_dist.get_expected_value(p_x_given_z_params)[0],
            mutable_updates=mutable_updates,
            state_updates={},
            **policy_metrics,
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

    def get_loss_args(self, state: DAPSState, batch: dict, rng_key):
        sample_keys = jax.random.split(rng_key, self.cfg.model.num_value_samples)
        x = self.package_x(batch[X], state.variables)
        z_given_x = vmap(lambda key: self.encode(state, x, key))(sample_keys)
        loss_args = DAPSLossArgs(x_data=batch[X], z_given_x=z_given_x, step=state.step)
        return loss_args

    def encode(self, state: DAPSState, x: Sample, rng_key, return_params=False):
        z_given_x, q_z_given_x_params, _ = self.recognition_model.sample_autoregressive(
            variables=state.variables["recognition_model"],
            inputs={X: x},
            temperature=1.0,
            train=False,
            rng_key=rng_key,
        )

        if return_params:
            return z_given_x, q_z_given_x_params

        return z_given_x

    def get_predictions(self, state: DAPSState, batch: dict, rng_key):
        x = self.package_x(batch[X], state.variables)
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

    @staticmethod
    def compute_ess(eta: float, advantages: jnp.ndarray, beta: float, log_q_z_given_x: jnp.ndarray):
        """Compute effective sample size (ESS)."""
        unnormalized_iws = (advantages - beta * log_q_z_given_x) / (eta + beta)
        iws = jnp.exp(unnormalized_iws - logsumexp(unnormalized_iws, axis=0, keepdims=True))
        return jnp.mean(1.0 / jnp.sum(iws**2, axis=0))


##### DAPS Non-Autoregressive Model #####


class DAPSNonAutoregressiveModel(DAPSModel):
    """Non-Autoregressive version of DAPS (DAPS-NA)"""

    def init(self, rng_key):
        init_key, state_key = jax.random.split(rng_key, 2)
        x = self.x_dist.create_sample((self.cfg.train.batch_size,))
        z_given_x = self.latent_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            recognition_model=self.recognition_model.init(init_key, {X: x}, train=True),
            generative_model=self.generative_model.init(init_key, {LATENT: z_given_x}, train=True),
            log_eta=dict(params=jnp.log(self.cfg.model.init_eta)),
            scalers={},
        )

        if self.cfg.dataset.scaler_mode == "online":
            variables["scalers"][X] = self.scalers[X].init(init_key, x.value)

        state = DAPSState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    def forward_models(
        self,
        variables: VariableDict,
        z_given_x: Sample,
        x: Sample,
        rng_key: PRNGKey,
        train: bool,
        mutable_updates: dict,
        generative_condition: dict[str, Sample] = {},
    ):
        q_key, p_key = jax.random.split(rng_key)

        q_z_given_x_params, q_mutable = vmap(
            lambda z_: self.recognition_model.apply(
                variables["recognition_model"],
                {X: x},
                rngs={"dropout": q_key},
                train=train,
                mutable=get_mutable(variables["recognition_model"]),
            )
        )(z_given_x)

        p_x_given_z_params, p_mutable = vmap(
            lambda z_: self.generative_model.apply(
                variables["generative_model"],
                {LATENT: z_, **generative_condition},
                train=train,
                rngs={"dropout": p_key},
                mutable=get_mutable(variables["generative_model"]),
            )
        )(z_given_x)

        # Average mutable stats across monte carlo axis
        mutable_updates["generative_model"] = jax.tree.map(lambda x: jnp.mean(x, axis=0), p_mutable)
        mutable_updates["recognition_model"] = jax.tree.map(lambda x: jnp.mean(x, axis=0), q_mutable)

        return q_z_given_x_params, p_x_given_z_params, mutable_updates

    def encode(self, state: DAPSState, x: Sample, rng_key, return_params=False):
        q_z_given_x_params = self.recognition_model.apply(
            state.variables["recognition_model"], {X: x}, train=False
        )

        z_given_x = self.latent_dist.sample(q_z_given_x_params, (), temperature=None, rng_key=rng_key)

        if return_params:
            return z_given_x, q_z_given_x_params

        return z_given_x
