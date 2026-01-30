from typing import ClassVar, Dict, Optional

import jax.numpy as jnp
import numpy as np
import optax
from chex import PRNGKey
from omegaconf import DictConfig, open_dict

from jadex.base.base_model import BaseModel
from jadex.base.base_state import BaseState
from jadex.base.registrable import Registrable
from jadex.distributions.base_distribution import BaseDistribution, DistParams, Sample
from jadex.networks.running_mean_std import RunningMeanStd
from jadex.networks.variational.constants import X
from jadex.networks.variational.variational_network import VariationalNetwork, create_networks_and_dists
from jadex.utils import is_power_of_two, non_pytree
from jadex.utils.printing import print_jit_str, print_yellow
from jadex.utils.scheduler import create_schedulers


class BaseVAEModel(BaseModel, Registrable):
    """Base class for Variational Autoencoders"""

    registered: ClassVar[Dict[str, "BaseVAEModel"]] = dict()
    cfg: DictConfig = non_pytree()
    recognition_model: VariationalNetwork = non_pytree()
    generative_model: VariationalNetwork = non_pytree()
    x_dist: BaseDistribution = non_pytree()
    latent_dist: BaseDistribution = non_pytree()
    latent_prior_params: DistParams = non_pytree()
    schedulers: Optional[Dict[str, optax.Schedule]] = non_pytree()

    @classmethod
    def create(cls, cfg):
        cls.set_model_bits(cfg)
        networks, dists = create_networks_and_dists(cfg.networks, cfg.dists, cfg.model.name)
        model_kwargs = cls.create_model_kwargs(cfg, networks, dists)
        return cls(cfg=cfg, **model_kwargs)

    @classmethod
    def create_model_kwargs(cls, cfg, networks, dists) -> Dict:
        """Override this method to generate model-specific kwargs."""
        base_kwargs = dict(
            **networks,
            **dists,
            latent_prior_params=dists["latent_dist"].get_prior_params(),
            schedulers=create_schedulers(cfg),
            scalers={X: RunningMeanStd()} if cfg.dataset.scaler_mode == "online" else {},
        )
        return base_kwargs

    def package_x(self, x_data, variables=None, mutable=False):
        """
        Packages x as a Sample.
        If using online normalization:
         - x is first normalized
         - If mutable is True, then the normalization stats are updated/returned
        """
        mutable_updates = {}
        if self.cfg.dataset.scaler_mode == "online":
            if mutable:
                mutable_updates["scalers"] = {}
                x_data, mutable_updates["scalers"][X] = self.scalers[X].apply(
                    variables["scalers"][X], x_data, mutable=True
                )
            else:
                x_data = self.apply_scaler(x_data, variables["scalers"], X)

        x = self.x_dist.package_sample(x_data)

        if mutable:
            return x, mutable_updates

        return x

    def encode(self, state: BaseState, x: Sample, rng_key: PRNGKey) -> Sample:
        raise NotImplementedError

    def compute_elbo(self, q_z_given_x_params: DistParams, log_p_x_given_z: jnp.ndarray, return_kl=False):
        latent_prior_kl = self.latent_dist.kl(q_z_given_x_params, self.latent_prior_params)
        elbo = jnp.mean(log_p_x_given_z - latent_prior_kl)
        if return_kl:
            return elbo, latent_prior_kl
        return elbo

    @property
    def final_beta(self):
        if self.cfg.schedulers.beta.type == "auto_exp_decay":
            final_beta = self.cfg.schedulers.beta.end_value
        else:
            # NOTE: optax schedules repeat the last value
            # final_beta = self.schedulers["beta"](sys.maxsize)
            raise NotImplementedError

        return final_beta

    def compute_beta_elbo(self, log_p_x_given_z: jnp.ndarray, latent_prior_kl: jnp.ndarray):
        beta_elbo = jnp.mean(log_p_x_given_z - self.final_beta * latent_prior_kl)
        return beta_elbo

    @classmethod
    def set_model_bits(cls, cfg):
        # NOTE: This is the default method for discrete VAEs.
        # Override this for other VAES (e.g. standard VAE and FSQ)
        vocab_size = cfg.model.get("vocab_size")
        num_bits = cfg.model.get("num_bits")
        block_size = cfg.model.get("block_size")

        # Case 1: num_bits is specified (and optionally block_size)
        if num_bits is not None:
            # If all are specified, check consistency
            if block_size is not None and vocab_size is not None:
                true_num_bits = int(block_size * np.log2(vocab_size))
                correct_num_bits = true_num_bits == num_bits
                if not correct_num_bits:
                    print(
                        f"WARNING: {num_bits} bits does not match {block_size} blocks and {vocab_size} vocab. "
                        + f"Changing num_bits to {true_num_bits}..."
                    )
                    with open_dict(cfg):
                        cfg.model.num_bits = true_num_bits
                else:
                    print_jit_str(
                        f"All specified values are consistent: "
                        + f"{num_bits} bits, {block_size} blocks, and {vocab_size} vocab"
                    )

            # If block_size is specified but vocab_size is not, calculate vocab_size
            elif block_size is not None:
                assert num_bits >= block_size, "num_bits must be greater than or equal to block_size."
                assert is_power_of_two(block_size), "block_size must be a power of 2"
                with open_dict(cfg):
                    cfg.model.vocab_size = int(2 ** (num_bits / block_size))
                assert cfg.model.vocab_size < 100_000, "too many vocab selected"
                print_jit_str(f"Calculated vocab_size: {cfg.model.vocab_size}")

            # If block_size is not specified but vocab_size is, calculate block_size
            elif vocab_size is not None:
                assert num_bits >= int(np.log2(vocab_size)), "num_bits must be >= log2(vocab_size)."
                with open_dict(cfg):
                    cfg.model.block_size = int(num_bits / np.log2(vocab_size))
                print_jit_str(f"Calculated block_size: {cfg.model.block_size}")

            # If neither block_size nor vocab_size is provided, raise an error
            else:
                raise ValueError(
                    "If num_bits is specified, either block_size or vocab_size must be provided."
                )

        # Case 2: vocab_size is specified and num_bits is None
        elif vocab_size is not None:
            if block_size is not None:
                if not is_power_of_two(block_size):
                    print_yellow("Warning! block_size is not a power of 2.")
                with open_dict(cfg):
                    cfg.model.num_bits = int(block_size * np.log2(vocab_size))
                print_jit_str(f"Calculated num_bits: {cfg.model.num_bits}")
            else:
                raise ValueError(
                    "If vocab_size is specified, block_size must also be provided to calculate num_bits."
                )

        # Case 3: Neither num_bits nor vocab_size is specified
        else:
            raise ValueError("You must provide either vocab_size, block_size, or num_bits.")

        # Final setup message
        print_jit_str(
            f"Setting up: {cfg.model.block_size} blocks, {cfg.model.vocab_size} vocab, and {cfg.model.num_bits} bits for {cfg.model.name}\n"
        )
