import multiprocessing
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import PRNGKey
from omegaconf import DictConfig

from jadex.algorithms.vae.models import create_vae
from jadex.base.base_state import BaseState
from jadex.base.base_trainer import BaseTrainer, PmapTrainer, SupervisedTrainerMixin
from jadex.data.dataloader import create_dataloader
from jadex.data.dataloader.jax_sampler import SampleBuffer
from jadex.data.datasets import create_dataset
from jadex.global_configs import jadex_hydra_main
from jadex.networks.variational.constants import X
from jadex.utils.plotting import plot_prediction

"""
This is the main training script for running VAE experiments. Example usage:

python train_vae.py model=daps dataset=cifar

"""


class VaeVisionTrainer(SupervisedTrainerMixin, BaseTrainer):

    def __init__(self, cfg, model, train_dataset, test_dataset):
        BaseTrainer.__init__(
            self, cfg=cfg, model=model, train_dataset=train_dataset, test_dataset=test_dataset
        )

        self.fid = None
        self._train_dataloader = None
        self._train_iter = None

        self.train_data = (
            jnp.array(self.train_dataset.data, jnp.float32) if hasattr(self.train_dataset, "data") else None
        )
        self.test_data = (
            jnp.array(self.test_dataset.data, jnp.float32)
            if (self.test_dataset is not None and hasattr(self.test_dataset, "data"))
            else None
        )

    ##### Setup overrides #####

    @staticmethod
    def create_model(cfg: DictConfig):
        return create_vae(cfg)

    @classmethod
    def make_ctx(cls, cfg: DictConfig):
        if cfg.dataset.id in ["imagenet128", "imagenet256"]:
            return multiprocessing.get_context("forkserver")
        return None

    @property
    def recon_metrics_include_ssim(self) -> bool:
        return True

    def setup(self, state: BaseState, ctx):
        ##### Optional FID #####
        if self.test_dataset is not None and self.cfg.dataset.id in ["cifar", "imagenet128", "imagenet256"]:
            from jadex.utils.fidjax import create_fid  # lazy import

            self.fid = create_fid(self.cfg)

        ##### ImageNet train iterator (needs ctx + state.sample_buffer_data) #####
        if ctx is not None and self.cfg.dataset.id in ["imagenet128", "imagenet256"]:
            self._train_dataloader = create_dataloader(
                cfg=self.cfg,
                mode="train",
                dataset=self.train_dataset,
                sample_buffer=SampleBuffer.create_from_state(state.sample_buffer_data, ctx),
                ctx=ctx,
                wrap_jax=False,
            )
            self._train_iter = iter(self._train_dataloader)

    ##### Batching #####

    @staticmethod
    def _maybe_mnist_binarize(cfg: DictConfig, x: jnp.ndarray) -> jnp.ndarray:
        # Only needed for In-Memory training
        if cfg.dataset.id == "mnist":
            x = jnp.expand_dims(x, axis=-1)
            x = (x > 127).astype(jnp.int32)
        return x

    def _fetch_train_x_host(self):
        return next(self._train_iter)[X]

    def get_train_batch(self, rng_key: PRNGKey) -> Dict[str, jnp.ndarray]:
        if self._train_iter is not None:
            # ImageNet: fetch from dataloader on host via io_callback
            x = jax.experimental.io_callback(
                self._fetch_train_x_host,
                jax.ShapeDtypeStruct((self.cfg.train.batch_size, *self.cfg.dists.x_dist.shape), jnp.float32),
            )
        else:
            idxs = jax.random.randint(rng_key, (self.cfg.train.batch_size,), 0, len(self.train_data))
            x = self.train_data[idxs]
            x = self._maybe_mnist_binarize(self.cfg, x)
            if self.cfg.dataset.scaler_mode == "data":
                x = self.train_dataset.apply_scaler(x)

        return {X: x}

    def _fetch_val_x_host(self, idxs):
        return jnp.stack([self.test_dataset.get((i, None))[X] for i in idxs])

    def _get_val_batch(self, idxs: jnp.ndarray) -> jnp.ndarray:
        if self.cfg.dataset.id in ["imagenet128", "imagenet256"] and self.test_dataset is not None:
            # ImageNet: fetch from test_dataset on host via io_callback
            x = jax.experimental.io_callback(
                self._fetch_val_x_host,
                jax.ShapeDtypeStruct((len(idxs), *self.cfg.dists.x_dist.shape), jnp.float32),
                idxs=idxs,
            )
        else:
            x = self.test_data[idxs]
            x = self._maybe_mnist_binarize(self.cfg, x)
            if self.cfg.dataset.scaler_mode == "data":
                x = self.train_dataset.apply_scaler(x)

        return {X: x}

    ##### Validation #####

    def run_validation(self, state: BaseState, get_placeholder: bool = False) -> Dict[str, Any]:
        assert self.test_dataset is not None, "Validation requested but no test dataset provided."

        val_batch_size = int(self.cfg.test.batch_size)
        if get_placeholder:
            num_val = val_batch_size * 2
        else:
            if self.cfg.dataset.id in ["imagenet128", "imagenet256"]:
                num_val = len(self.test_dataset)
            else:
                num_val = len(self.test_data)

        def _get_predictions(val_batch, rng_key):
            model_metrics = self.model.get_predictions(state, val_batch, rng_key)

            # Prefix model metrics as val_*
            model_metrics = {f"val_{k}": v for k, v in model_metrics.items()}

            # Optional FID
            if self.fid is not None:
                val_x_hats = model_metrics["val_x_hats"]
                if self.cfg.dataset.scaler_mode == "online":
                    images = self.model.apply_inverse_scaler(val_x_hats, state.scaler_vars, X)
                elif self.cfg.dataset.scaler_mode == "data":
                    images = self.train_dataset.apply_inverse_scaler(val_x_hats)
                else:
                    raise ValueError("Unsupported scaler_mode")
                images = jnp.clip(images, 0, 255).astype(jnp.uint8)
                model_metrics["val_fid_acts"] = self.fid.compute_acts(images)

            model_metrics.update(self.get_metrics(prefix="val", batch=val_batch, metrics=model_metrics))

            return model_metrics

        def body_fn(carry, batch_idxs):
            rng_key, last_batch, last_x_hats = carry
            predict_key, new_key = jax.random.split(rng_key, 2)
            val_batch = self._get_val_batch(batch_idxs)
            val_metrics = _get_predictions(val_batch, predict_key)
            val_x_hats = val_metrics.pop("val_x_hats")
            return (new_key, val_batch, val_x_hats.astype(jnp.float32)), val_metrics

        perm_key, scan_key = jax.random.split(state.rng_key, 2)
        all_idxs = jax.random.permutation(perm_key, num_val)
        split_idxs = jnp.array(jnp.array_split(all_idxs, num_val // val_batch_size))

        batch_ph = self._get_val_batch(split_idxs[-1])
        x_hats_ph = self.model.x_dist.create_sample((val_batch_size,)).value.astype(jnp.float32)

        (_, ret_batch, ret_x_hats), ret_metrics = jax.lax.scan(
            body_fn, (scan_key, batch_ph, x_hats_ph), split_idxs
        )

        # Finalize FID
        if self.fid is not None:
            all_fid_acts = jnp.concatenate(ret_metrics.pop("val_fid_acts"), 0)
            stats = self.fid.compute_stats(all_fid_acts)
            ret_metrics["val_fid_score"] = jax.pure_callback(
                self.fid.compute_score, jax.ShapeDtypeStruct((), jnp.float32), stats=stats
            )

        ret_metrics = jax.tree.map(jnp.mean, ret_metrics)
        ret_metrics["val_x_hats"] = ret_x_hats

        return ret_batch, ret_metrics


class VaeLafanTrainer(SupervisedTrainerMixin, BaseTrainer):
    ##### Setup overrides #####

    @staticmethod
    def create_model(cfg: DictConfig):
        return create_vae(cfg)

    @classmethod
    def create_datasets(cls, cfg: DictConfig, ctx) -> Tuple[Any, Optional[Any]]:
        traj_dataset = create_dataset(cfg, "train", ctx=None)
        return traj_dataset, None

    def get_train_batch(self, rng_key: PRNGKey) -> Dict[str, jnp.ndarray]:
        return self.train_dataset.get_train_batch(rng_key)

    def log_expensive(
        self, prefix: str, batch: Dict[str, jnp.ndarray], metrics: Dict[str, Any], state: BaseState
    ):
        assert prefix in ["val", "train"]
        xs = batch[X]
        x_hats = self.model.apply_inverse_scaler(metrics[f"{prefix}_x_hats"], state.scaler_vars, X)
        return plot_prediction(x_hats, xs, self.cfg, prefix)

    ##### Validation #####

    def run_validation(self, state: BaseState, get_placeholder: bool = False) -> Dict[str, Any]:
        if get_placeholder:
            num_val = int(self.cfg.test.batch_size) * 2
        else:
            num_val = int(self.cfg.test.num_val_samples)

        batch_size = int(self.cfg.test.batch_size)
        scan_length = int(num_val // batch_size)

        def _get_predictions(val_batch, rng_key):
            model_metrics = self.model.get_predictions(state, val_batch, rng_key)
            model_metrics = {f"val_{k}": v for k, v in model_metrics.items()}
            model_metrics.update(self.get_metrics(prefix="val", batch=val_batch, metrics=model_metrics))
            return model_metrics

        def body_fn(carry, _):
            rng_key, last_batch, last_x_hats = carry
            predict_key, sample_key, new_key = jax.random.split(rng_key, 3)
            val_batch = self.train_dataset.get_val_batch(sample_key)
            val_metrics = _get_predictions(val_batch, predict_key)
            val_x_hats = val_metrics.pop("val_x_hats")

            return (new_key, val_batch, val_x_hats.astype(jnp.float32)), val_metrics

        # Always use the same key for validation
        val_rng_key = jax.random.PRNGKey(0)

        batch_ph = self.train_dataset.get_val_batch(val_rng_key)
        x_hats_ph = self.model.x_dist.create_sample((batch_size,)).value.astype(jnp.float32)

        (_, ret_batch, ret_x_hats), ret_metrics = jax.lax.scan(
            body_fn, (val_rng_key, batch_ph, x_hats_ph), None, scan_length
        )

        ret_metrics = jax.tree.map(jnp.mean, ret_metrics)
        ret_metrics["val_x_hats"] = ret_x_hats

        return ret_batch, ret_metrics


class VaePmapTrainer(SupervisedTrainerMixin, PmapTrainer):

    @staticmethod
    def create_model(cfg: DictConfig):
        return create_vae(cfg)

    @property
    def recon_metrics_include_ssim(self) -> bool:
        return True

    ##### Validation #####

    def run_validation(self, state: BaseState, get_placeholder: bool = False) -> Dict[str, Any]:
        assert not get_placeholder, "Not Implemented!"
        assert self.test_dataloader is not None

        def _get_predictions(val_batch, rng_key):
            model_metrics = self.model.get_predictions(state, val_batch, rng_key)
            model_metrics = {f"val_{k}": v for k, v in model_metrics.items()}

            # Optional FID
            if self.fid is not None:
                val_x_hats = model_metrics["val_x_hats"]
                if self.cfg.dataset.scaler_mode == "online":
                    images = self.model.apply_inverse_scaler(val_x_hats, state.scaler_vars, X)
                elif self.cfg.dataset.scaler_mode == "data":
                    images = self.train_dataset.apply_inverse_scaler(val_x_hats)
                else:
                    raise ValueError("Unsupported scaler_mode")
                images = jnp.clip(images, 0, 255).astype(jnp.uint8)
                model_metrics["val_fid_acts"] = self.fid.compute_acts(images)

            model_metrics.update(self.get_metrics(prefix="val", batch=val_batch, metrics=model_metrics))

            return model_metrics

        p_val_fn = jax.pmap(_get_predictions, in_axes=(0, None), out_axes=0, axis_name="batch")

        # return a random batch to visualize different batches when logging
        # NOTE: validation is not shuffled, and last batch is not dropped
        ret_idx = jax.random.randint(
            state.rng_key, shape=(), minval=0, maxval=self.test_dataloader.num_batches - 1
        )

        # iterator will be exhausted after the first loop, since we aren't using the infinite sampler
        self.test_dataloader.reset()

        all_metrics = []
        all_fid_acts = []

        for i, p_batch in enumerate(self.test_dataloader):
            p_val_metrics = p_val_fn(p_batch, jax.random.PRNGKey(i))

            val_metrics = jax.tree.map(lambda x: jnp.concatenate(jnp.atleast_2d(x)), p_val_metrics)
            x_hats = val_metrics.pop("val_x_hats")

            if i == ret_idx:
                ret_x_hats = x_hats
                ret_batch = jax.tree.map(jnp.concatenate, p_batch)

            if self.fid is not None:
                all_fid_acts.append(val_metrics.pop("val_fid_acts"))

            all_metrics.append(jax.tree.map(jnp.mean, val_metrics))

        combined_metrics = jax.tree.map(lambda *args: jnp.stack(args), *all_metrics)
        ret_metrics = jax.tree.map(jnp.mean, combined_metrics)

        if self.fid is not None:
            stats = self.fid.compute_stats(all_fid_acts)
            fid_score = self.fid.compute_score(stats)
            ret_metrics["val_fid_score"] = fid_score

        ret_metrics["val_x_hats"] = ret_x_hats

        return ret_batch, ret_metrics


@jadex_hydra_main(config_name="vae_config", config_path="./configs")
def main(cfg: DictConfig):
    if cfg.dataset.id == "lafan":
        assert cfg.dataset.in_memory, "LAFAN only supports in-memory dataset!"
        VaeLafanTrainer.submit(cfg)
    else:
        if cfg.dataset.in_memory:
            # Fastest Trainer (used by default)
            assert cfg.dataset.id in ["mnist", "cifar", "imagenet128", "imagenet256"]
            VaeVisionTrainer.submit(cfg)
        else:
            # General Purpose Trainer that works with most dataloaders/datasets
            # NOTE: this is *slow* compared to in-memory trainers
            VaePmapTrainer.submit(cfg)


if __name__ == "__main__":
    main()
