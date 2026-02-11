import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig, OmegaConf

MIN_EPS = jnp.finfo(jnp.float32).eps


def create_scheduler(cfg: dict):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    scheduler_cls_name = cfg.pop("type")
    if scheduler_cls_name == "auto_exp_decay":
        init_value = float(cfg["init_value"])
        end_value = float(cfg["end_value"])
        end_pct = float(cfg["end_pct"])
        total_nsteps = float(cfg.pop("total_nsteps"))
        assert init_value > 2 * MIN_EPS, f"start value must be > {2 * MIN_EPS}"

        # clip the ends by EPS to allow for auto decay
        end_value = np.clip(end_value, MIN_EPS, init_value - MIN_EPS)

        # Calculate the step at which the exponential decay ends
        first_step_of_final_decay = int(total_nsteps * end_pct)
        decay_rate = (end_value / init_value) ** (1.0 / first_step_of_final_decay)

        exp_scheduler = optax.schedules.exponential_decay(
            init_value=init_value, end_value=end_value, decay_rate=decay_rate, transition_steps=1
        )

        return exp_scheduler
    elif scheduler_cls_name == "auto_exp_lin_decay":
        exp_init_value = float(cfg["exp_init_value"])
        exp_end_value = float(cfg["exp_end_value"])
        exp_end_pct = float(cfg["exp_end_pct"])
        lin_end_value = float(cfg["lin_end_value"])
        lin_end_pct = float(cfg["lin_end_pct"])
        assert exp_init_value > exp_end_value
        assert lin_end_pct > exp_end_pct

        total_nsteps = float(cfg.pop("total_nsteps"))

        # Calculate the steps at which decay happens based on percentages
        first_step_of_final_decay = int(total_nsteps * exp_end_pct)
        decay_rate = (exp_end_value / exp_init_value) ** (1.0 / first_step_of_final_decay)

        schedule1 = optax.schedules.exponential_decay(
            init_value=exp_init_value, end_value=exp_end_value, decay_rate=decay_rate, transition_steps=1
        )

        second_step_of_final_decay = int(total_nsteps * lin_end_pct)
        schedule2 = optax.schedules.linear_schedule(
            init_value=exp_end_value,
            end_value=lin_end_value,
            transition_steps=second_step_of_final_decay - first_step_of_final_decay,
        )

        auto_scheduler = optax.schedules.join_schedules(
            [schedule1, schedule2], boundaries=[first_step_of_final_decay]
        )
        return auto_scheduler
    else:
        scheduler_cls = eval(f"optax.schedules.{scheduler_cls_name}")
        return scheduler_cls(**cfg)


def create_schedulers(cfg: DictConfig):
    if cfg.get("schedulers", False):
        return {k: create_scheduler(v) for k, v in cfg.schedulers.items()}
    else:
        return None
