from typing import ClassVar, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import PRNGKey
from einops import rearrange
from flax import struct
from flax.typing import VariableDict
from omegaconf import DictConfig, OmegaConf, open_dict

from jadex.algorithms.vae.models.base_vae import BaseVAEModel
from jadex.algorithms.vae.models.daps import DAPSModel
from jadex.algorithms.vae.models.fsq import FSQModel
from jadex.algorithms.vae.models.gumbel import GRMCKModel, GumbelModel
from jadex.algorithms.vae.models.ppo import PPOModel
from jadex.algorithms.vae.models.vqvae import VQVAEModel
from jadex.base.base_model import BaseModel, BaseModelLossArgs, BaseState
from jadex.base.base_state import BaseState, get_mutable
from jadex.base.registrable import Registrable
from jadex.distributions.base_distribution import Sample
from jadex.distributions.categorical import Categorical, CategoricalParams
from jadex.distributions.uniform import Uniform
from jadex.global_configs.constants import JADEX_CHECKPOINT_DIR
from jadex.networks.variational.constants import LABEL, LATENT, TIME, X
from jadex.networks.variational.variational_network import (
    VariationalNetwork,
    create_networks_and_dists,
    merge_nn_cfg,
)
from jadex.utils import non_pytree
from jadex.utils.scheduler import create_schedulers


@struct.dataclass
class ImageLabelDiscreteFlowLossArgs(BaseModelLossArgs):
    label: Sample
    z_t: Sample
    z_1: Sample
    time: Sample


class ImageLabelDiscreteFlowState(BaseState):
    pass


class BaseImageLabelDiscreteFlowModel(BaseModel, Registrable):
    registered: ClassVar[Dict[str, "BaseImageLabelDiscreteFlowModel"]] = dict()
    vae_model: BaseVAEModel = non_pytree()
    vae_state: BaseState = non_pytree()
    label_dist: Categorical = non_pytree()
    time_dist: Uniform = non_pytree()
    gpt_latent_dist: Categorical = non_pytree()
    label_recognition_model: VariationalNetwork = non_pytree()
    schedulers: Optional[Dict[str, optax.Schedule]] = non_pytree()

    @property
    def x_dist(self):
        return self.vae_model.x_dist

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

    @classmethod
    def merge_cfg(cls, cfg):
        vae_cfg = BaseState.load_cfg(JADEX_CHECKPOINT_DIR / cfg.model.vae_checkpoint_name)

        with open_dict(vae_cfg):
            vae_cfg.train = cfg.train

        with open_dict(cfg):
            cfg.dataset = vae_cfg.dataset
            cfg.dataset.include_labels = True

        # Add resolved VAE cfg to main cfg
        OmegaConf.resolve(vae_cfg)
        with open_dict(cfg):
            cfg.vae_cfg = vae_cfg

        # Combine with VAE dists to create networks
        cfg = cls.add_gpt_latent_dist(cfg, vae_cfg)

        merge_nn_cfg(cfg)

    @classmethod
    def create(cls, cfg, vae_model, vae_state):
        dists_cfg = DictConfig(
            {
                **OmegaConf.to_container(cfg.vae_cfg.dists),
                **OmegaConf.to_container(cfg.dists, resolve=True),
            }
        )

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
            label_recognition_model=networks["label_recognition_model"],
            label_dist=dists["label_dist"],
            time_dist=dists["time_dist"],
            gpt_latent_dist=dists["gpt_latent_dist"],
            schedulers=create_schedulers(cfg),
            scalers={},
        )
        return base_kwargs

    def init(self, rng_key):
        init_key, state_key = jax.random.split(rng_key, 2)
        label = self.label_dist.create_sample((self.cfg.train.batch_size,))
        latent = self.gpt_latent_dist.create_sample((self.cfg.train.batch_size,))
        time = self.time_dist.create_sample((self.cfg.train.batch_size,))

        variables = dict(
            label_recognition_model=self.label_recognition_model.init(
                init_key, {LATENT: latent, LABEL: label, TIME: time}, train=True
            )
        )

        state = ImageLabelDiscreteFlowState.create(
            cfg=self.cfg,
            apply_fn=self.loss_fn,
            variables=variables,
            rng_key=state_key,
        )

        return state

    def loss_fn(
        self,
        variables: VariableDict,
        loss_args: ImageLabelDiscreteFlowLossArgs,
        rng_key: PRNGKey,
        train=True,
    ):
        drop_key, label_key = jax.random.split(rng_key, 2)
        q_z_given_label_params, q_mutable = self.label_recognition_model.apply(
            variables["label_recognition_model"],
            {LATENT: loss_args.z_t, TIME: loss_args.time, LABEL: loss_args.label},
            rngs={"dropout": drop_key, "label_dropout": label_key},
            train=train,
            mutable=get_mutable(variables["label_recognition_model"]),
        )

        q_z_given_label_params = self.modify_label_recognition_params(q_z_given_label_params)
        log_q_z_given_label = self.gpt_latent_dist.log_prob(q_z_given_label_params, loss_args.z_1)
        label_recognition_loss = -jnp.mean(log_q_z_given_label)
        losses = dict(label_recognition_model=label_recognition_loss)

        metrics = dict(
            losses=losses,
            loss=sum(losses.values()),
            mutable_updates=q_mutable,
            state_updates={},
        )

        return metrics

    @property
    def gpt_latent_prior_params(self):
        # FSQ overrides this method (by adding a mask)!
        return self.gpt_latent_dist.get_prior_params()

    def modify_label_recognition_params(self, params: CategoricalParams):
        # FSQ overrides this method (by adding a mask)!
        return params

    def get_loss_args(self, state: ImageLabelDiscreteFlowState, batch: dict, rng_key):
        batch_size = batch[X].shape[0]
        x = self.vae_model.package_x(batch[X], self.vae_state.variables)
        z0_key, z1_key, t_key, rand_t_key = jax.random.split(rng_key, 4)
        label = self.label_dist.package_sample(jnp.expand_dims(batch[LABEL], axis=-1))

        z_0 = self.gpt_latent_dist.sample(
            self.gpt_latent_prior_params, leading_shape=(batch_size,), temperature=1.0, rng_key=z0_key
        )

        z_1 = self.vae_encode_discrete_sequence(x, z1_key)

        time = self.time_dist.sample_from_prior(t_key, leading_shape=(batch_size,))

        rand_time = self.time_dist.sample_from_prior(
            rand_t_key, leading_shape=(batch_size, self.gpt_latent_dist.shape[-1])
        )

        z_t = self.gpt_latent_dist.package_sample(
            jnp.where(rand_time.value.squeeze(-1) < time.value, z_1.value, z_0.value)
        )

        loss_args = ImageLabelDiscreteFlowLossArgs(label=label, z_t=z_t, z_1=z_1, time=time)

        return loss_args

    def vae_encode_discrete_sequence(self, x: Sample, rng_key: PRNGKey) -> Sample:
        return self.vae_model.encode(self.vae_state, x, rng_key)

    def vae_decode_discrete_sequence(self, discrete_sequence: Sample) -> Sample:
        p_x_given_z_params = self.vae_model.generative_model.apply(
            self.vae_state.variables["generative_model"], {LATENT: discrete_sequence}, train=False
        )
        x_hat = self.x_dist.get_expected_value(p_x_given_z_params)
        return x_hat

    def generate_from_labels(self, state: ImageLabelDiscreteFlowState, batch: dict, rng_key):
        batch_size = batch[LABEL].shape[0]

        init_rng_key, z_key = jax.random.split(rng_key, 2)

        label = self.label_dist.package_sample(jnp.expand_dims(batch[LABEL], axis=-1))

        init_z_t = self.gpt_latent_dist.sample_from_prior(z_key, leading_shape=(batch_size,))

        dt = self.cfg.model.dt

        def body_fn(carry, _):
            z_t, t, rng_key = carry

            samp_key, next_rng_key = jax.random.split(rng_key, 2)

            time = self.time_dist.package_sample(jnp.full((batch_size, 1), t))

            q_z_given_label_params = self.label_recognition_model.apply(
                state.variables["label_recognition_model"],
                {LATENT: z_t, TIME: time, LABEL: label},
                train=False,
            )

            q_z_given_label_params = self.modify_label_recognition_params(q_z_given_label_params)

            u = (q_z_given_label_params.probs - z_t.onehot) / (1.0 - t)

            t_params = self.gpt_latent_dist.package_params(jnp.log(z_t.onehot + dt * u))

            z_t = self.gpt_latent_dist.sample(
                params=t_params, leading_shape=(), temperature=1, rng_key=samp_key
            )

            return (z_t, t + dt, next_rng_key), None

        (z_t, _, _), _ = jax.lax.scan(body_fn, (init_z_t, 0.0, init_rng_key), xs=None, length=int(1.0 / dt))

        x_hat = self.vae_decode_discrete_sequence(z_t)

        return x_hat


class DAPSImageLabelDiscreteFlowModel(BaseImageLabelDiscreteFlowModel):
    vae_model: DAPSModel = non_pytree()


class PPOImageLabelDiscreteFlowModel(BaseImageLabelDiscreteFlowModel):
    vae_model: PPOModel = non_pytree()


class GumbelImageLabelDiscreteFlowModel(BaseImageLabelDiscreteFlowModel):
    vae_model: GumbelModel = non_pytree()


class GRMCKImageLabelDiscreteFlowModel(BaseImageLabelDiscreteFlowModel):
    vae_model: GRMCKModel = non_pytree()


class VQVAEImageLabelDiscreteFlowModel(BaseImageLabelDiscreteFlowModel):
    vae_model: VQVAEModel = non_pytree()

    def vae_encode_discrete_sequence(self, x: Sample, rng_key: PRNGKey) -> Sample:
        q_z_given_x_params = self.vae_model.recognition_model.apply(
            self.vae_state.variables["recognition_model"], {X: x}, train=False
        )
        z_e = self.vae_model.latent_dist.get_expected_value(q_z_given_x_params)
        z_q, codebook_indices = self.vae_model.quantize(self.vae_state.variables["codebook"], z_e)
        discrete_sequence = self.gpt_latent_dist.package_sample(codebook_indices)
        return discrete_sequence

    def vae_decode_discrete_sequence(self, discrete_sequence: Sample) -> Sample:
        codebook_indices = discrete_sequence.value
        z_q = self.vae_model.codebook.apply(self.vae_state.variables["codebook"], codebook_indices)
        z_q = self.vae_model.latent_dist.package_sample(z_q)
        return super().vae_decode_discrete_sequence(z_q)


class FSQImageLabelDiscreteFlowModel(BaseImageLabelDiscreteFlowModel):
    """
    NOTE: This discretization logic could probably be implemented differently / simplified
    e.g., by using FSQ's encode_index instead
    """

    vae_model: FSQModel = non_pytree()

    @classmethod
    def add_gpt_latent_dist(cls, cfg, vae_cfg):
        max_num_classes = max(vae_cfg.model.fsq_levels)
        num_cats = int(vae_cfg.model.block_size * len(vae_cfg.model.fsq_levels))
        gpt_latent_dist_cfg = DictConfig(
            dict(
                name="Categorical",
                param_shape=[num_cats, max_num_classes],
                shape=[num_cats],
            )
        )

        with open_dict(cfg):
            cfg.dists.gpt_latent_dist = gpt_latent_dist_cfg
            cfg.networks.label_recognition_model.block_size = num_cats
            cfg.vae_cfg.model.vocab_size = max_num_classes

        return cfg

    @property
    def gpt_latent_prior_params(self):
        masked_logits = jnp.where(self.get_fsq_mask(), super().gpt_latent_prior_params.logits, -jnp.inf)
        return self.gpt_latent_dist.package_params(masked_logits)

    def modify_label_recognition_params(self, params: CategoricalParams):
        masked_logits = jnp.where(self.get_fsq_mask(), params.logits, -jnp.inf)
        return self.gpt_latent_dist.package_params(masked_logits)

    def vae_encode_discrete_sequence(self, x: Sample, rng_key: PRNGKey) -> Sample:
        q_z_given_x_params = self.vae_model.recognition_model.apply(
            self.vae_state.variables["recognition_model"], {X: x}, train=False
        )
        z_e = self.vae_model.latent_dist.get_expected_value(q_z_given_x_params)
        z_q = self.vae_model.quantize_ste(z_e)
        codebook_indices = self.fsq_to_categorical(z_q)
        codebook_indices = rearrange(codebook_indices, "b k d -> b (k d)")
        discrete_sequence = self.gpt_latent_dist.package_sample(codebook_indices)
        return discrete_sequence

    def vae_decode_discrete_sequence(self, discrete_sequence: Sample) -> Sample:
        codebook_indices = discrete_sequence.value
        cats = rearrange(codebook_indices, "b (k d) -> b k d", k=self.cfg.vae_cfg.model.block_size)
        z_q_val = self.categorical_to_fsq(cats)
        z_q = self.vae_model.latent_dist.package_sample(z_q_val)
        return super().vae_decode_discrete_sequence(z_q)

    def get_fsq_mask(self):
        levels = self.vae_model._levels_np
        max_level = int(max(levels))
        vec_mask = np.vstack([np.arange(max_level) < level for level in levels])
        mask = jnp.concatenate([vec_mask for _ in range(self.cfg.vae_cfg.model.block_size)])
        return mask

    def normalize_levels(self, q: jnp.ndarray):
        levels = self.vae_model._levels_np

        q = jnp.asarray(q)
        d = q.shape[-1]

        if isinstance(levels, (int, np.integer)):
            lev = jnp.full((d,), int(levels), dtype=jnp.int32)
        else:
            lev_np = np.asarray(levels).astype(np.int32)
            if lev_np.ndim == 0:
                lev = jnp.full((d,), int(lev_np), dtype=jnp.int32)
            elif lev_np.ndim == 1:
                if lev_np.size == 1:
                    lev = jnp.full((d,), int(lev_np.item()), dtype=jnp.int32)
                elif lev_np.size == d:
                    lev = jnp.asarray(lev_np, dtype=jnp.int32)
                else:
                    raise ValueError(
                        f"'levels' length ({lev_np.size}) must be 1 or match q.shape[-1] ({d})."
                    )
            else:
                raise ValueError(f"'levels' must be scalar or 1D, got shape {lev_np.shape}.")

        bw_shape = (1,) * (q.ndim - 1) + (d,)
        levels_bw = jnp.reshape(lev, bw_shape)
        half_bw = jnp.reshape(lev // 2, bw_shape).astype(jnp.int32)
        return levels_bw, half_bw

    def fsq_to_categorical(self, q: jnp.ndarray) -> jnp.ndarray:
        q = jnp.asarray(q)
        levels_bw, half_bw = self.normalize_levels(q)
        y_int = jnp.rint(q * half_bw).astype(jnp.int32)
        cats = y_int + half_bw
        cats = jnp.clip(cats, 0, (levels_bw - 1)).astype(jnp.int32)
        return cats

    def categorical_to_fsq(self, cats: jnp.ndarray) -> jnp.ndarray:
        cats = jnp.asarray(cats, dtype=jnp.int32)
        dummy_q = jnp.zeros_like(cats, dtype=jnp.float32)
        levels_bw, half_bw = self.normalize_levels(dummy_q)
        cats = jnp.clip(cats, 0, (levels_bw - 1).astype(jnp.int32))
        q = (cats - half_bw) / jnp.asarray(half_bw, dtype=jnp.float32)
        return q.astype(jnp.float32)
