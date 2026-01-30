from typing import ClassVar, Dict, Optional

import jax
import jax.numpy as jnp
import optax
from chex import PRNGKey
from flax import struct
from flax.typing import VariableDict
from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.algorithms.vae.models.daps import DAPSModel
from jadex.algorithms.vae.models.vqvae_fsq import VQVAEModel
from jadex.base.base_model import BaseModel, BaseModelLossArgs, BaseState
from jadex.base.base_state import BaseState, get_mutable
from jadex.base.registrable import Registrable
from jadex.distributions.base_distribution import BaseDistribution, Sample
from jadex.distributions.categorical import Categorical
from jadex.networks.running_mean_std import RunningMeanStd
from jadex.networks.variational.constants import CUR_LATENT, GOAL, LATENT, LATENT_HIST, TEXT, X
from jadex.networks.variational.other_models.traj_gpt_policy import TrajGptPolicy
from jadex.networks.variational.variational_network import create_networks_and_dists, merge_nn_cfg
from jadex.utils import non_pytree
from jadex.utils.scheduler import create_schedulers
from omegaconf import DictConfig, OmegaConf, open_dict


@struct.dataclass
class TrajGptLossArgs(BaseModelLossArgs):
    z_cur: Sample
    z_hist: Sample
    goal_data: jnp.ndarray
    input_ids: Sample


class TrajGptState(BaseState):
    pass


class BaseTrajGptModel(BaseModel, Registrable):
    registered: ClassVar[Dict[str, "BaseTrajGptModel"]] = dict()
    vae_model: BaseVAEModel = non_pytree()
    vae_state: BaseState = non_pytree()
    text_dist: Categorical = non_pytree()
    goal_dist: BaseDistribution = non_pytree()
    gpt_latent_dist: Categorical = non_pytree()
    traj_gpt_policy: TrajGptPolicy = non_pytree()
    schedulers: Optional[Dict[str, optax.Schedule]] = non_pytree()

    @property
    def x_dist(self):
        return self.vae_model.x_dist

    @classmethod
    def add_gpt_latent_dist(cls, cfg, vae_cfg):
        raise NotImplementedError

    @classmethod
    def create(cls, cfg, vae_cfg, vae_model, vae_state):
        # Combine with VAE dists to create networks
        cfg = cls.add_gpt_latent_dist(cfg, vae_cfg)
        dists_cfg = DictConfig(
            {
                **OmegaConf.to_container(vae_cfg.dists),
                **OmegaConf.to_container(cfg.dists, resolve=True),
            }
        )

        cfg = merge_nn_cfg(cfg)
        networks, dists = create_networks_and_dists(cfg.networks, dists_cfg, cfg.model.name)
        model_kwargs = cls.create_model_kwargs(cfg, networks, dists)

        return cls(
            cfg=cfg,
            vae_model=vae_model,
            vae_state=vae_state,
            **model_kwargs,
        )

    @staticmethod
    def create_model_kwargs(cfg, networks, dists) -> Dict:
        """Override this method to generate model-specific kwargs."""
        base_kwargs = dict(
            traj_gpt_policy=networks["traj_gpt_policy"],
            text_dist=dists["text_dist"],
            goal_dist=dists["goal_dist"],
            gpt_latent_dist=dists["gpt_latent_dist"],
            schedulers=create_schedulers(cfg),
            scalers={GOAL: RunningMeanStd()},
        )
        return base_kwargs

    def init(self, rng_key):
        init_key, state_key = jax.random.split(rng_key, 2)
        text = self.text_dist.create_sample((self.cfg.train.batch_size,))
        goal = self.goal_dist.create_sample((self.cfg.train.batch_size,))
        latent = self.gpt_latent_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            traj_gpt_policy=self.traj_gpt_policy.init(
                init_key, {CUR_LATENT: latent, LATENT_HIST: latent, TEXT: text, GOAL: goal}, train=True
            ),
            scalers={GOAL: self.scalers[GOAL].init(init_key, goal.value)},
        )

        state = TrajGptState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    def loss_fn(self, variables: VariableDict, loss_args: TrajGptLossArgs, rng_key: PRNGKey, train=True):
        goal, mutable_updates = self.package_goal(loss_args.goal_data, variables, mutable=True)

        q_z_given_c_params, mutable_updates["traj_gpt_policy"] = self.traj_gpt_policy.apply(
            variables["traj_gpt_policy"],
            {
                CUR_LATENT: loss_args.z_cur,
                LATENT_HIST: loss_args.z_hist,
                TEXT: loss_args.input_ids,
                GOAL: goal,
            },
            rngs={"dropout": rng_key},
            train=train,
            mutable=get_mutable(variables["traj_gpt_policy"]),
        )

        log_q_z_given_c = self.gpt_latent_dist.log_prob(q_z_given_c_params, loss_args.z_cur)
        label_recognition_loss = -jnp.mean(log_q_z_given_c)

        losses = dict(label_recognition_model=label_recognition_loss)

        metrics = dict(
            losses=losses,
            loss=sum(losses.values()),
            mutable_updates=mutable_updates,
            state_updates={},
        )

        return metrics

    def get_loss_args(self, state: TrajGptState, batch: dict, rng_key):
        hist_key, cur_key = jax.random.split(rng_key, 2)

        x_hist_data, x_cur_data = jnp.array_split(batch[X], 2, axis=-2)

        x_hist = self.vae_model.package_x(x_hist_data, self.vae_state.variables)
        z_hist = self.vae_encode_discrete_sequence(x_hist, hist_key)

        x_cur = self.vae_model.package_x(x_cur_data, self.vae_state.variables)
        z_cur = self.vae_encode_discrete_sequence(x_cur, cur_key)

        loss_args = TrajGptLossArgs(
            z_hist=z_hist,
            z_cur=z_cur,
            goal_data=batch[GOAL],
            input_ids=self.text_dist.package_sample(batch[TEXT]),
        )

        return loss_args

    def vae_encode_discrete_sequence(self, x: Sample, rng_key: PRNGKey) -> Sample:
        return self.vae_model.encode(self.vae_state, x, rng_key)

    def vae_decode_discrete_sequence(self, discrete_sequence: Sample) -> Sample:
        p_x_given_z_params, _ = self.vae_model.generative_model.apply(
            self.vae_state.variables["generative_model"],
            {LATENT: discrete_sequence},
            train=False,
            mutable=get_mutable(self.vae_state.variables["generative_model"]),
        )
        x_hat = self.x_dist.get_expected_value(p_x_given_z_params)
        return x_hat

    def encode(self, state: TrajGptState, conditions: Dict[str, Sample], rng_key, return_params=False):
        z_given_c, q_z_given_c_params, _ = self.traj_gpt_policy.sample_autoregressive(
            variables=state.variables["traj_gpt_policy"],
            inputs=conditions,
            temperature=1.0,
            train=False,
            rng_key=rng_key,
        )

        if return_params:
            return z_given_c, q_z_given_c_params

        return z_given_c

    def package_goal(self, goal_data, variables=None, mutable=False):
        """
        Normalizes and packages goal as a Sample
        If mutable is True, then the normalization stats are updated/returned.
        """
        mutable_updates = {}
        if mutable:
            mutable_updates["scalers"] = {}
            goal_data, mutable_updates["scalers"][GOAL] = self.scalers[GOAL].apply(
                variables["scalers"][GOAL], goal_data, mutable=True
            )
        else:
            goal_data = self.apply_scaler(goal_data, variables["scalers"], GOAL)

        goal = self.goal_dist.package_sample(goal_data)

        if mutable:
            return goal, mutable_updates

        return goal

    def predict_traj(self, state: TrajGptState, batch: dict, rng_key):
        hist_key, cur_key = jax.random.split(rng_key, 2)

        x_hist_data, x_cur_data = jnp.array_split(batch[X], 2, axis=-2)
        x_hist = self.vae_model.package_x(x_hist_data, self.vae_state.variables)
        z_hist = self.vae_encode_discrete_sequence(x_hist, hist_key)

        x_cur = self.vae_model.package_x(x_cur_data, self.vae_state.variables)
        z_cur = self.vae_encode_discrete_sequence(x_cur, cur_key)
        x_cur_recon = self.vae_decode_discrete_sequence(z_cur)

        conditions = {
            LATENT_HIST: z_hist,
            TEXT: self.text_dist.package_sample(batch[TEXT]),
            GOAL: self.package_goal(batch[GOAL], state.variables),
        }

        zcur_hat_given_c = self.encode(state, conditions, rng_key)
        x_cur_hat = self.vae_decode_discrete_sequence(zcur_hat_given_c)

        return x_cur_hat, x_cur_recon, x_cur_data


class DAPSTrajGptModel(BaseTrajGptModel):
    vae_model: DAPSModel = non_pytree()

    @classmethod
    def add_gpt_latent_dist(cls, cfg, vae_cfg):
        with open_dict(cfg):
            cfg.dists.gpt_latent_dist = vae_cfg.dists.latent_dist
        return cfg


class VQVAETrajGptModel(BaseTrajGptModel):
    vae_model: VQVAEModel = non_pytree()

    @classmethod
    def add_gpt_latent_dist(cls, cfg, vae_cfg):
        gpt_latent_dist_cfg = DictConfig(
            dict(
                name="Categorical",
                param_shape=[vae_cfg.model.block_size, vae_cfg.model.vocab_size],
                shape=[vae_cfg.model.block_size],
            )
        )

        with open_dict(cfg):
            cfg.dists.gpt_latent_dist = gpt_latent_dist_cfg
        return cfg

    def vae_encode_discrete_sequence(self, x: Sample, rng_key: PRNGKey) -> Sample:
        """Encode input x into discrete sequence tokens for GPT."""
        codebook_indices = self.vae_model.encode_index(self.vae_state, x, rng_key)
        discrete_sequence = self.gpt_latent_dist.package_sample(codebook_indices)
        return discrete_sequence

    def vae_decode_discrete_sequence(self, discrete_sequence: Sample) -> Sample:
        """Decode discrete GPT sequence back into VAE latent space."""
        codebook_indices = discrete_sequence.value
        z_q = self.vae_model.codebook.apply(self.vae_state.variables["codebook"], codebook_indices)
        z_q = self.vae_model.latent_dist.package_sample(z_q)
        return BaseTrajGptModel.vae_decode_discrete_sequence(self, z_q)
