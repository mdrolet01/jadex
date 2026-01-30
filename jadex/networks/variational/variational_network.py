import uuid
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, Type

import flax.linen as nn
from chex import PRNGKey
from flax.typing import VariableDict
from omegaconf import DictConfig, OmegaConf, open_dict

from jadex.base.registrable import Registrable
from jadex.distributions import create_distribution
from jadex.distributions.base_distribution import BaseDistribution, DistParams, Sample
from jadex.distributions.bernoulli import BernoulliParams
from jadex.distributions.categorical import CategoricalParams
from jadex.distributions.diagonal_gaussian import DiagonalGaussianParams
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit, print_jit_str


class VariationalNetwork(nn.Module, ABC, Registrable):
    registered: ClassVar[Dict[str, "VariationalNetwork"]] = dict()
    cfg: DictConfig = non_pytree()
    input_dists: Dict[str, BaseDistribution] = non_pytree()
    output_dists: Dict[str, BaseDistribution] = non_pytree()
    output_modality: str = non_pytree()
    print_info: dict = non_pytree()
    parent_model_name: str = non_pytree()

    # Core network operations
    @classmethod
    def create(cls, cfg, input_dists, output_dists, parent_model_name=""):
        assert len(output_dists.values()) == 1, "Multiple output modalities are not currently supported"
        output_modality = list(output_dists.keys())[0]
        print_info = dict(name=cfg.name, uuid=str(uuid.uuid4()))
        network_kwargs = cls.create_network_kwargs(
            cfg, input_dists, output_dists, print_info, parent_model_name
        )
        return cls(
            cfg=cfg,
            input_dists=input_dists,
            output_dists=output_dists,
            output_modality=output_modality,
            print_info=print_info,
            parent_model_name=parent_model_name,
            **network_kwargs,
        )

    @classmethod
    @abstractmethod
    def create_network_kwargs(
        cls,
        cfg: DictConfig,
        input_dists: Dict[str, BaseDistribution],
        output_dists: Dict[str, BaseDistribution],
        print_info: Dict,
        parent_model_name: str,
    ):
        raise NotImplementedError

    @abstractmethod
    def output(self, samples: Dict[str, Sample], train: bool):
        raise NotImplementedError

    def sample_autoregressive(
        self,
        variables: VariableDict,
        inputs: Dict[str, Sample],
        temperature: float,
        train: bool,
        rng_key: PRNGKey,
    ):
        raise NotImplementedError

    @nn.compact
    def __call__(self, samples: Dict[str, Sample], train: bool):
        assert len(samples.keys()) == len(
            self.input_dists.keys()
        ), f"Input mismatch! : {list(samples.keys())} {list(self.input_dists.keys())}"

        return self.output(samples, train=train)

    def _print_output(self, modality: str, params: DistParams, constant_variance: bool = False):
        all_attrs = {
            DiagonalGaussianParams: ["mean", "variance"],
            BernoulliParams: ["logits"],
            CategoricalParams: ["logits"],
        }

        assert params.__class__ in all_attrs, f"Unsupported type {params.__class__}"
        print_attrs = all_attrs[params.__class__]

        for i, attr in enumerate(print_attrs):
            if constant_variance and attr == "variance":
                attr_name = "constant variance"
            else:
                attr_name = attr

            print_jit(
                f"{modality} {attr_name} ({params.__class__.__name__})",
                getattr(params, attr).shape,
                self.print_info,
                output=True,
                footer=i == len(print_attrs) - 1,
            )

    def _print_input(self, modality: str, samples: Sample):
        print_jit(
            f"{modality} ({samples.__class__.__name__})",
            samples.value.shape,
            self.print_info,
            input=True,
            header=True,
        )

    @classmethod
    def raise_not_supported(cls, io_type: str, obj_cls: Type, modality, print_info):
        cls_name = obj_cls.__class__.__name__
        raise ValueError(f"{print_info['name']} unsupported {io_type} type: {cls_name} for {modality}")


def create_network(
    cfg: DictConfig,
    input_dists: Dict[str, BaseDistribution],
    output_dists: Dict[str, BaseDistribution],
    parent_model_name: str = "",
):
    nn_cls = VariationalNetwork.registered[cfg.name]
    return nn_cls.create(
        cfg=cfg, input_dists=input_dists, output_dists=output_dists, parent_model_name=parent_model_name
    )


def create_networks_and_dists(networks_cfg: DictConfig, dists_cfg: DictConfig, parent_model_name=""):
    dists: Dict[str, BaseDistribution] = {}
    for dist_name, dist_cfg in dists_cfg.items():
        dists[dist_name] = create_distribution(dist_cfg)

    def _create_dists(target_dists):
        ret_dists = {}
        for dist_key, dist_val in target_dists.items():
            ret_dists[dist_key] = dists[dist_val] if dist_val else None
        return ret_dists

    networks = {}
    for nn_tag, nn_cfg in networks_cfg.items():
        input_dists = _create_dists(nn_cfg.input_dists)
        output_dists = _create_dists(nn_cfg.output_dists)

        dist_str = f"Setting up distributions for {nn_tag}: {nn_cfg.name}\n"
        dist_str += "Input distributions:\n"
        for k, v in input_dists.items():
            dist_str += f"  {k}: {v.__class__.__name__}\n"
        dist_str += "Output distributions:\n"
        for k, v in output_dists.items():
            dist_str += f"  {k}: {v.__class__.__name__}"
            if k != list(output_dists.keys())[-1]:  # Add newline except for last item
                dist_str += "\n"

        print_jit_str(dist_str, with_header_footer=True)

        networks[nn_tag] = create_network(
            nn_cfg, input_dists, output_dists, parent_model_name=parent_model_name
        )

    return networks, dists


def merge_nn_cfg(cfg: DictConfig) -> DictConfig:
    if not hasattr(cfg, "networks") or cfg.networks is None:
        return cfg

    for network_name, network_cfg in cfg.networks.items():
        if not hasattr(network_cfg, "nn") or network_cfg.nn is None:
            continue

        with open_dict(cfg):
            nn_cfg = network_cfg.nn
            OmegaConf.resolve(nn_cfg)
            merged = OmegaConf.merge(nn_cfg, network_cfg)
            del merged.nn
            cfg.networks[network_name] = merged

    # Now safe to delete top-level nn
    if hasattr(cfg, "nn"):
        with open_dict(cfg):
            del cfg["nn"]

    return cfg
