import multiprocessing
import time
import uuid
from functools import partial
from typing import Any, Dict, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from chex import PRNGKey
from flax import core
from jadex.base.base_model import BaseModel
from jadex.base.base_state import BaseState
from jadex.base.registrable import register_all
from jadex.data.dataloader import create_dataloader
from jadex.data.dataloader.jax_sampler import SampleBuffer
from jadex.data.datasets import create_dataset
from jadex.data.datasets.base_dataset import BaseDataset
from jadex.global_configs.constants import JADEX_CHECKPOINT_DIR
from jadex.networks.variational.constants import X
from jadex.networks.variational.variational_network import merge_nn_cfg
from jadex.utils import submit_job
from jadex.utils.dm_pix_metrics import mse, psnr, ssim
from jadex.utils.plotting import plot_prediction
from jadex.utils.printing import print_blue, print_green, print_yellow
from omegaconf import DictConfig, OmegaConf, open_dict


class BaseTrainer:

    def __init__(
        self,
        cfg: DictConfig,
        model: BaseModel,
        train_dataset: BaseDataset,
        test_dataset: Optional[BaseDataset],
    ):
        self.cfg = cfg
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.checkpoint_file = self._create_checkpoint_file(prefix=cfg.wandb.project)
        self._ckpt_idx = 0
        self.last_time = None
        self.last_step = None

    ##### Entrypoint #####

    @classmethod
    def submit(cls, cfg: DictConfig):
        merge_nn_cfg(cfg)

        with open_dict(cfg):
            cfg.wandb = DictConfig(cfg.get("wandb") or {"mode": "disabled"})
            cfg.wandb.project = cfg.wandb.get("project", default_value=f"{cfg.model.id}-{cfg.dataset.id}")

        ##### Resume config from checkpoint #####
        start_checkpoint = cfg.job.get("start_checkpoint")
        if start_checkpoint is not None:
            ckpt_file = JADEX_CHECKPOINT_DIR / start_checkpoint
            assert ckpt_file.exists(), f"No checkpoint {start_checkpoint} found!"
            ckpt_cfg: DictConfig = BaseState.load_cfg(ckpt_file)

            override_keys = cfg.job.checkpoint_overrides + [
                "job.start_checkpoint",
                "job.start_checkpoint_idx",
            ]

            assert "dataset.in_memory" not in override_keys, "cannot use different in_memory setting!"

            for key in override_keys:
                OmegaConf.update(ckpt_cfg, key, OmegaConf.select(cfg, key), merge=False)

            cfg = ckpt_cfg

        if cfg.job.get("uuid", None) is None:
            with open_dict(cfg):
                cfg.job.uuid = str(uuid.uuid4())

        submit_job(cls._run_training, cfg)

    @classmethod
    def _run_training(cls, cfg: DictConfig):
        register_all()

        ##### Setup #####
        ctx = cls.make_ctx(cfg)
        train_dataset, test_dataset = cls.create_datasets(cfg, ctx)

        model = cls.create_model(cfg)
        state: BaseState = model.init(jax.random.PRNGKey(cfg.train.seed))

        wandb.init(**cfg.wandb, config=OmegaConf.to_container(cfg))

        trainer = cls(cfg=cfg, model=model, train_dataset=train_dataset, test_dataset=test_dataset)

        ##### Resume weights #####
        start_checkpoint = cfg.job.get("start_checkpoint")
        if start_checkpoint is not None:
            state = state.load_checkpoint(
                JADEX_CHECKPOINT_DIR / start_checkpoint, cfg.job.start_checkpoint_idx
            )

        ##### Child-only setup hook #####
        trainer.setup(state, ctx)

        print_blue(OmegaConf.to_yaml(cfg))
        state.print_opt_params()

        trainer.train(state)

    ##### Overridables: setup #####

    @classmethod
    def make_ctx(cls, cfg: DictConfig):
        return None

    @classmethod
    def create_datasets(cls, cfg: DictConfig, ctx) -> Tuple[Any, Optional[Any]]:
        train_dataset = create_dataset(cfg, "train", ctx=ctx)
        if cfg.get("test") is not None and cfg.job.get("validation_frequency_nsteps", 0) > 0:
            test_dataset = create_dataset(cfg, "test", ctx=ctx)
        else:
            test_dataset = None
        return train_dataset, test_dataset

    ##### Overridables #####

    @staticmethod
    def create_model(cfg: DictConfig):
        raise NotImplementedError

    def get_train_batch(self, rng_key: PRNGKey) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError

    def run_validation(self, state: BaseState, get_placeholder: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    def get_metrics(self, prefix: str, batch: Dict[str, Any], metrics: Dict[str, Any]):
        raise NotImplementedError

    def log_expensive(
        self, prefix: str, batch: Dict[str, Any], metrics: Dict[str, Any], state: BaseState = None
    ):
        raise NotImplementedError

    def setup(self, state: BaseState, ctx):
        # default: no-op
        return

    @property
    def exclude_metrics(self):
        return []

    ##### Utilities #####

    @staticmethod
    def _uuid_suffix() -> str:
        return "".join(str(uuid.uuid4()).split("-")[:2])

    @classmethod
    def _create_checkpoint_file(cls, prefix: str):
        try:
            run_name = wandb.run.name
            if not run_name or run_name.startswith("dummy"):
                raise ValueError
            suffix = run_name
        except Exception:
            suffix = cls._uuid_suffix()
        checkpoint_file = JADEX_CHECKPOINT_DIR / f"{prefix}_{suffix}.h5"
        print(f"Checkpoints for this run will be saved to: {checkpoint_file}")
        return checkpoint_file

    ##### Checkpointing #####

    def save_checkpoint(self, state: BaseState):
        state.save_checkpoint(self.checkpoint_file, checkpoint_idx=self._ckpt_idx)
        self._ckpt_idx += 1
        print_green(f"##### Saved checkpoint {self._ckpt_idx} #####")

    ##### Steps #####

    def get_step(self, state: BaseState) -> int:
        return state.step

    def get_internal_step(self, state: BaseState) -> int:
        return state.internal_step

    ##### Optimization #####

    def update_model(self, state: BaseState, batch: Dict[str, jnp.ndarray]):
        args_key, grad_key = jax.random.split(state.rng_key)
        loss_args = self.model.get_loss_args(state, batch, args_key)
        state, metrics = state.perform_gradient_update(loss_args, grad_key)
        return state, metrics

    ##### Scheduling #####

    def update_should_run(self, cur_step, should_run: core.FrozenDict, last_steps: core.FrozenDict):
        should_run = should_run.unfreeze()
        for metric_name in last_steps.keys():
            freq_key = f"{metric_name}_frequency_nsteps"
            freq = self.cfg.job.get(freq_key, 0)
            if freq > 0:
                should_run[metric_name] = (cur_step - last_steps[metric_name]) >= freq
        return core.freeze(should_run)

    def update_last_steps(self, cur_step, should_run: core.FrozenDict, last_steps: core.FrozenDict):
        last_steps = last_steps.unfreeze()
        for event in should_run:
            last_steps[event] = jax.lax.cond(should_run[event], lambda: cur_step, lambda: last_steps[event])
        return core.freeze(last_steps)

    ##### Training Loop #####

    def train(self, state: BaseState):
        init_step = self.get_step(state)
        if init_step >= self.cfg.job.total_nsteps:
            print_yellow(
                "Run has already completed! To continue training, increase job.total_nsteps "
                + "(and ensure checkpoint_overrides contains job.total_nsteps)"
            )
            return

        freq_fields = ["print", "wandb_inexpensive", "wandb_expensive", "validation", "checkpoint"]
        init_last_steps = core.FrozenDict([(k, init_step) for k in freq_fields])
        init_should_run = core.FrozenDict([(k, False) for k in freq_fields])

        freq_intervals = [self.cfg.job.get(f"{name}_frequency_nsteps", 0) for name in freq_fields]
        freq_intervals = [x for x in freq_intervals if x != 0]
        min_freq_interval = min(freq_intervals) if freq_intervals else 1

        if freq_intervals:
            assert np.all(
                [x % min_freq_interval == 0 for x in freq_intervals]
            ), "For performance: ensure all *_frequency_nsteps are divisible by the smallest."

        init_carry = (state, init_step, init_should_run, init_last_steps)

        val_frequency = self.cfg.job.get("validation_frequency_nsteps", 0)
        if val_frequency > 0:
            val_batch_ph, val_metrics_ph = jax.jit(partial(self.run_validation, get_placeholder=True))(state)

        def step_fn(carry, _):
            state, prev_step, should_run, last_steps = carry

            new_rng_key, sample_key = jax.random.split(state.rng_key)
            state = state.replace(rng_key=new_rng_key)

            batch = self.get_train_batch(sample_key)

            state, train_metrics = self.update_model(state, batch)

            cur_step = self.get_step(state)
            check_cond = jnp.logical_and(cur_step % min_freq_interval == 0, cur_step > prev_step)

            should_run = jax.lax.cond(
                check_cond,
                lambda: self.update_should_run(cur_step, should_run, last_steps),
                lambda: should_run,
            )

            if val_frequency > 0:
                val_batch, val_metrics = jax.lax.cond(
                    jnp.logical_and(check_cond, should_run["validation"]),
                    lambda: self.run_validation(state),
                    lambda: (val_batch_ph, val_metrics_ph),
                )
            else:
                val_batch = val_metrics = None

            jax.lax.cond(
                check_cond,
                lambda: jax.debug.callback(
                    self.step_callback,
                    state=state,
                    train_metrics=train_metrics,
                    train_batch=batch,
                    should_run=should_run,
                    val_metrics=val_metrics,
                    val_batch=val_batch,
                ),
                lambda: None,
            )

            last_steps = jax.lax.cond(
                check_cond,
                lambda: self.update_last_steps(cur_step, should_run, last_steps),
                lambda: last_steps,
            )

            return (state, cur_step, should_run, last_steps), None

        _carry, _ = jax.lax.scan(step_fn, init_carry, None, self.cfg.job.total_nsteps)
        wandb.finish()

    ##### Callback #####

    def step_callback(self, state, train_metrics, train_batch, should_run, val_metrics, val_batch):
        step = self.get_step(state)
        internal_step = self.get_internal_step(state)

        wandb_metrics: Dict[str, Any] = {}

        if should_run["print"]:
            loss = jnp.mean(train_metrics["loss"]).item()
            step_str = f"{step:07d}"
            if internal_step != step:
                step_str = f"{step:07d}/{internal_step:07d}"

            if self.last_time is None:
                self.last_time = time.time()
                self.last_step = step
                print(f"step {step_str} | loss {loss:.4f}")
            else:
                rate = 60 * (step - self.last_step) / (time.time() - self.last_time)
                print(f"step {step_str} ({round(rate, 2)} steps/min) | loss {loss:.4f}")
                self.last_time = time.time()
                self.last_step = step

        if should_run["validation"]:
            wandb_metrics.update(
                self.log_expensive(prefix="val", batch=val_batch, metrics=val_metrics, state=state)
            )
            wandb_metrics.update(jax.tree.map(jnp.mean, val_metrics))

        if should_run["wandb_inexpensive"]:
            wandb_metrics.update(self.get_metrics(prefix="train", batch=train_batch, metrics=train_metrics))
            wandb_metrics.update(jax.tree.map(jnp.mean, train_metrics))

        if should_run["wandb_expensive"]:
            wandb_metrics.update(
                self.log_expensive(prefix="train", batch=train_batch, metrics=train_metrics, state=state)
            )

        if should_run["checkpoint"]:
            self.save_checkpoint(state)

        if wandb_metrics:
            for k in self.exclude_metrics:
                wandb_metrics.pop(k, None)
            wandb.log(wandb_metrics, step=step)

        if step >= self.cfg.job.total_nsteps:
            print_green("Run finished!")
            if self.cfg.job.checkpoint_frequency_nsteps > 0 and not should_run["checkpoint"]:
                print_green("Saving final checkpoint...")
                self.save_checkpoint(state)


class PmapTrainer(BaseTrainer):
    """
    PMAP trainer that reuses BaseTrainer's callback/logging/metrics semantics,
    but replaces the scan-based loop with a python for-loop over (already-sharded) dataloaders.

    Assumptions:
      - train_dataloader yields batches with shape (n_devices, per_device_batch, ...)
      - test_dataloader yields batches with the same leading (n_devices, per_device_batch, ...) shape
      - create_dataloader(..., wrap_jax=True) is configured so batches are pmap-ready
    """

    def __init__(self, cfg: DictConfig, model: Any, train_dataset: Any, test_dataset: Optional[Any]):
        super().__init__(cfg=cfg, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
        self.train_dataloader = None
        self.test_dataloader = None
        self.fid = None

    ##### Setup overrides #####

    @classmethod
    def make_ctx(cls, cfg: DictConfig):
        return multiprocessing.get_context("forkserver")

    def setup(self, state: BaseState, ctx):
        # Dataloader already returns pmap-ready batches (n_devices, per_device_batch, ...)
        self.train_dataloader = create_dataloader(
            cfg=self.cfg,
            mode="train",
            dataset=self.train_dataset,
            sample_buffer=SampleBuffer.create_from_state(state.sample_buffer_data, ctx),
            ctx=ctx,
        )

        if self.test_dataset is not None and self.cfg.job.get("validation_frequency_nsteps", 0) > 0:
            self.test_dataloader = create_dataloader(
                cfg=self.cfg,
                mode="test",
                dataset=self.test_dataset,
                ctx=ctx,
            )

            if self.cfg.dataset.id in ["cifar", "imagenet128", "imagenet256"]:
                from jadex.utils.fidjax import create_fid  # lazy import

                self.fid = create_fid(self.cfg)

    def get_train_batch(self, rng_key: PRNGKey) -> Dict[str, jnp.ndarray]:
        # Not used in this trainer; train() consumes dataloader directly.
        raise NotImplementedError("PmapTrainer overrides train() and does not use get_train_batch().")

    ##### Train #####

    def train(self, state: BaseState):
        init_step = self.get_step(state)
        if init_step >= self.cfg.job.total_nsteps:
            print_yellow(
                "Run has already completed! To continue training, increase job.total_nsteps "
                + "(and ensure checkpoint_overrides contains job.total_nsteps)"
            )
            return

        assert self.train_dataloader is not None, "PmapTrainer.setup must create train_dataloader"

        ##### Scheduling #####
        freq_fields = ["print", "wandb_inexpensive", "wandb_expensive", "validation", "checkpoint"]
        last_steps = core.FrozenDict([(k, init_step) for k in freq_fields])
        should_run = core.FrozenDict([(k, False) for k in freq_fields])

        freq_intervals = [int(self.cfg.job.get(f"{name}_frequency_nsteps", 0)) for name in freq_fields]
        freq_intervals = [x for x in freq_intervals if x != 0]
        min_freq_interval = min(freq_intervals) if freq_intervals else 1

        if freq_intervals:
            assert all(
                (x % min_freq_interval) == 0 for x in freq_intervals
            ), "For performance: ensure all *_frequency_nsteps are divisible by the smallest."

        p_update_fn = jax.pmap(self.update_model, in_axes=(None, 0), out_axes=(None, 0), axis_name="batch")

        prev_step = self.get_step(state)

        for p_batch in self.train_dataloader:
            state, p_train_metrics = p_update_fn(state, p_batch)

            ##### IMPORTANT: sample buffer must be synchronized when using dataloader iter #####
            state = state.replace(sample_buffer_data=self.train_dataloader.synchronize())

            cur_step = self.get_step(state)
            check_cond = cur_step % min_freq_interval == 0 and cur_step > prev_step

            if check_cond:
                should_run = self.update_should_run(cur_step, should_run, last_steps)

                ##### Run Validation #####
                val_batch = None
                val_metrics = {}
                if should_run["validation"] and self.test_dataloader is not None:
                    val_batch, val_metrics = self.run_validation(state, get_placeholder=False)

                train_metrics = jax.tree.map(lambda x: jnp.concatenate(jnp.atleast_2d(x)), p_train_metrics)
                batch = jax.tree.map(jnp.concatenate, p_batch)

                self.step_callback(
                    state=state,
                    train_metrics=train_metrics,
                    train_batch=batch,
                    should_run=should_run,
                    val_metrics=val_metrics,
                    val_batch=val_batch,
                )

                last_steps = self.update_last_steps(cur_step, should_run, last_steps)

            prev_step = cur_step
            if cur_step >= self.cfg.job.total_nsteps:
                break

        wandb.finish()


class TrainerMixinProtocol(Protocol):
    train_dataset: BaseDataset
    model: BaseModel


class SupervisedTrainerMixin(TrainerMixinProtocol):

    @property
    def exclude_metrics(self):
        return ["train_x_hats", "val_x_hats", "val_xs"]

    @property
    def recon_metrics_include_ssim(self) -> bool:
        return False

    def get_metrics(self, prefix: str, batch: Dict[str, Any], metrics: Dict[str, Any]):
        assert prefix in ["val", "train"]

        xs_scaled = batch[X]
        x_hats_scaled = metrics[f"{prefix}_x_hats"]

        if self.cfg.dataset.scaler_mode != "data":
            raise ValueError("get_metrics currently supports scaler_mode == 'data' only.")

        dxs = self.train_dataset.apply_inverse_scaler(xs_scaled)
        dxh = self.train_dataset.apply_inverse_scaler(x_hats_scaled)

        xs = self.train_dataset._normalize(dxs).clip(0.0, 1.0)
        xh = self.train_dataset._normalize(dxh).clip(0.0, 1.0)

        out: Dict[str, jnp.ndarray] = {
            f"{prefix}_mse": mse(xs, xh).mean(),
            f"{prefix}_psnr": psnr(xs, xh).mean(),
        }
        if self.recon_metrics_include_ssim:
            out[f"{prefix}_ssim"] = ssim(xs, xh).mean()
        return out

    def log_expensive(
        self, prefix: str, batch: Dict[str, Any], metrics: Dict[str, Any], state: BaseState = None
    ):
        assert prefix in ["val", "train"]

        if self.cfg.dataset.scaler_mode == "online":
            assert state is not None
            xs = batch[X]
            x_hats = self.model.apply_inverse_scaler(metrics[f"{prefix}_x_hats"], state.scaler_vars, X)
        elif self.cfg.dataset.scaler_mode == "data":
            xs = self.train_dataset.apply_inverse_scaler(batch[X])
            x_hats = self.train_dataset.apply_inverse_scaler(metrics[f"{prefix}_x_hats"])
        else:
            raise ValueError("Unsupported scaler_mode")

        return plot_prediction(x_hats, xs, self.cfg, prefix)
