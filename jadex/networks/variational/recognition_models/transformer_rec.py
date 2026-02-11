from abc import abstractmethod
from typing import Callable, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax.typing import VariableDict
from omegaconf import DictConfig

from jadex.base.base_state import combine_mutable_updates, get_model_variables_and_mutable
from jadex.data.utils.image_patcher import Patcher
from jadex.distributions.base_distribution import BaseDistribution, Sample
from jadex.distributions.bernoulli import Bernoulli, BernoulliSample
from jadex.distributions.categorical import (
    Categorical,
    CategoricalSample,
    GRMCKCategorical,
    GumbelSoftmaxCategorical,
    GumbelSoftmaxSample,
)
from jadex.distributions.diagonal_gaussian import (
    DiagonalGaussian,
    DiagonalGaussianConstantVariance,
    DiagonalGaussianSample,
)
from jadex.networks.autoregressive_sampling import Transformer, TransformerConfig, create_predict_fn
from jadex.networks.conv1d import ResampleConv1D, ResampleConv1DConfig
from jadex.networks.nn_utils import FeedForwardNetwork
from jadex.networks.transformer import TransformerDecoder, TransformerDecoderConfig
from jadex.networks.variational.constants import LATENT, X
from jadex.networks.variational.variational_network import VariationalNetwork
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit


class BaseTransformerRecognitionModel(VariationalNetwork):
    """Recognition Model used for all VAEs (except the vanilla VAE)"""

    transformer: Transformer = non_pytree()
    predict_fn: Optional[Callable] = non_pytree()
    latent_output_head: Optional[nn.Dense] = non_pytree()

    @classmethod
    @abstractmethod
    def setup_x_input(cls, x_dist: BaseDistribution, cfg: DictConfig, print_info: Dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def process_x_samples(
        self,
        x: Sample,
        variables: VariableDict = None,
        train: bool = True,
        rng_key: PRNGKey = None,
    ) -> jnp.ndarray:
        raise NotImplementedError

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP LATENT INPUT (DAPS/Gumbel) #####
        if LATENT in input_dists.keys():
            latent_dist = input_dists[LATENT]
            if latent_dist.matches([Categorical, GumbelSoftmaxCategorical, GRMCKCategorical]):
                # create any necessary models here
                pass
            else:
                cls.raise_not_supported("input", latent_dist, LATENT, print_info)

        ##### SETUP X INPUT #####
        x_network_kwargs = cls.setup_x_input(input_dists[X], cfg, print_info)

        ##### SETUP LATENT OUTPUT #####
        latent_dist = output_dists[LATENT]

        predict_fn = latent_output_head = None
        if latent_dist.matches([Categorical, GumbelSoftmaxCategorical, GRMCKCategorical]):
            # AUTOREGRESSIVE MODELS
            if LATENT in input_dists.keys():
                transformer_config = TransformerConfig(**cfg.transformer, sample_dist=latent_dist)
                transformer = Transformer(transformer_config, print_info=print_info)
                predict_fn = create_predict_fn(config=transformer_config, print_info=print_info)
            else:
                transformer_config = TransformerDecoderConfig(**cfg.transformer)
                transformer = TransformerDecoder(transformer_config)
                latent_output_head = nn.Dense(latent_dist.num_classes)
        elif latent_dist.matches(DiagonalGaussianConstantVariance):
            assert parent_model_name in ["VQVAEModel", "FSQModel"]
            transformer_config = TransformerDecoderConfig(**cfg.transformer)
            transformer = TransformerDecoder(transformer_config)
            if latent_dist.shape[-1] != cfg.transformer.embed_dim:
                latent_output_head = nn.Dense(latent_dist.shape[-1])
        else:
            cls.raise_not_supported("output", latent_dist, LATENT, print_info)

        return dict(
            transformer=transformer,
            predict_fn=predict_fn,
            latent_output_head=latent_output_head,
            **x_network_kwargs,
        )

    def output(self, samples, train):
        ##### PROCESS LATENT (DAPS/Gumbel) #####
        if LATENT in self.input_dists.keys():
            latent = samples[LATENT]
            self._print_input(LATENT, latent)
            if latent.matches(CategoricalSample):
                latent = samples[LATENT].value
            elif latent.matches([GumbelSoftmaxSample, GRMCKCategorical]):
                latent = samples[LATENT].onehot
            else:
                self.raise_not_supported("sample", latent, LATENT, self.print_info)

            print_jit(LATENT, latent.shape, self.print_info)

        ##### PROCESS X #####
        x = samples[X]
        self._print_input(X, x)
        x = self.process_x_samples(x=x, train=train)

        ##### OUTPUT LATENT #####
        output_dist = self.output_dists[LATENT]
        if output_dist.matches([Categorical, GumbelSoftmaxCategorical, GRMCKCategorical]):
            if LATENT in self.input_dists.keys():
                logits = self.transformer(inputs=x, targets=latent, train=train)
                params = output_dist.package_params(logits)
                self._print_output(LATENT, params)
            else:
                latent_emb = self.transformer(target_embs=x, train=train)
                logits = self.latent_output_head(latent_emb)
                params = output_dist.package_params(logits)
                self._print_output(LATENT, params)
        elif output_dist.matches(DiagonalGaussianConstantVariance):
            ##### VQVAE / FSQ #####
            latent_emb = self.transformer(target_embs=x, train=train)
            if self.latent_output_head is not None:
                latent_emb = self.latent_output_head(latent_emb)
                print_jit(f"{LATENT} emb", latent_emb.shape, self.print_info)
            params = output_dist.package_params(latent_emb)
            self._print_output(LATENT, params, constant_variance=True)
        else:
            self.raise_not_supported("output", output_dist, LATENT, self.print_info)

        return params

    def sample_autoregressive(self, variables, inputs, temperature, train, rng_key):
        x_key, predict_key = jax.random.split(rng_key)
        mutable_updates = {}

        ##### PROCESS X #####
        x, process_x_mutable_update_dict = self.process_x_samples(inputs[X], variables, train, x_key)
        mutable_updates = combine_mutable_updates(process_x_mutable_update_dict, mutable_updates)

        apply_variables, mutable = get_model_variables_and_mutable(variables, "transformer")
        z_given_x_data, logits, tf_mutable_update_dict = self.predict_fn(
            variables=apply_variables,
            train=train,
            mutable=mutable,
            inputs=x,
            temperature=temperature,
            rng_key=predict_key,
        )

        mutable_updates = combine_mutable_updates(
            tf_mutable_update_dict, mutable_updates, prefix="transformer"
        )

        ##### OUTPUT LATENT #####
        output_dist = self.output_dists[LATENT]
        z_given_x_params = output_dist.package_params(logits)
        if output_dist.matches(Categorical):
            z_given_x = self.output_dists[LATENT].package_sample(value=z_given_x_data)
        elif output_dist.matches([GumbelSoftmaxCategorical, GRMCKCategorical]):
            value = jnp.argmax(z_given_x_data, axis=-1)  # used for non-gradient-based operations
            z_given_x = self.output_dists[LATENT].package_sample(value=value, onehot=z_given_x_data)
        else:
            self.raise_not_supported("output", output_dist, LATENT, self.print_info)

        return z_given_x, z_given_x_params, mutable_updates


class VisionTransformerRecognitionModel(BaseTransformerRecognitionModel):
    patcher: Patcher = non_pytree()
    patch_ffwd: FeedForwardNetwork = non_pytree()

    @classmethod
    def setup_x_input(cls, x_dist, cfg, print_info):
        if x_dist.matches([Bernoulli, DiagonalGaussian, DiagonalGaussianConstantVariance]):
            patcher = Patcher.create(*x_dist.shape, cfg.num_image_patches)
            patch_ffwd = FeedForwardNetwork(cfg.patch_ffwd_layers, print_info)
        else:
            cls.raise_not_supported("input", x_dist, X, print_info)

        return dict(patcher=patcher, patch_ffwd=patch_ffwd)

    def process_x_samples(self, x, variables=None, train=True, rng_key=None):
        if x.matches([BernoulliSample, DiagonalGaussianSample]):
            H, W, C = x.value.shape[-3:]
            assert self.input_dists[X].shape == (H, W, C)
            flat_pad_patches = self.patcher.patchify_pad_flat(x.value)
        else:
            self.raise_not_supported("sample", x, X, self.print_info)

        if variables is not None:
            apply_variables, mutable = get_model_variables_and_mutable(variables, "patch_ffwd")
            x, ffwd_mutable_updates = self.patch_ffwd.apply(
                apply_variables,
                flat_pad_patches,
                train=train,
                rngs={"dropout": rng_key},
                mutable=mutable,
            )
            process_x_mutable_update_dict = {"patch_ffwd": ffwd_mutable_updates}
            return x, process_x_mutable_update_dict
        else:
            x = self.patch_ffwd(flat_pad_patches, train)
            return x


class TrajTransformerRecognitionModel(BaseTransformerRecognitionModel):
    resample_conv: Optional[ResampleConv1D] = non_pytree()

    @classmethod
    def setup_x_input(cls, x_dist, cfg, print_info):
        resample_conv = None
        if x_dist.matches([DiagonalGaussian, DiagonalGaussianConstantVariance]):
            resample_conv_config = ResampleConv1DConfig(**cfg.resample_conv1d)
            resample_conv = ResampleConv1D(resample_conv_config, print_info)
        else:
            cls.raise_not_supported("input", x_dist, X, print_info)

        return dict(resample_conv=resample_conv)

    def process_x_samples(self, x, variables=None, train=True, rng_key=None):
        if x.matches(DiagonalGaussianSample):
            T, C = x.value.shape[-2:]
            assert self.input_dists[X].shape == (T, C)
            x = x.value
        else:
            self.raise_not_supported("sample", x, X, self.print_info)

        if variables is not None:
            model_variables, mutable = get_model_variables_and_mutable(variables, "resample_conv")
            x, mutable_updates = self.resample_conv.apply(
                model_variables,
                x,
                train=train,
                rngs={"dropout": rng_key},
                mutable=mutable,
            )
            mutable_update_dict = {"resample_conv": mutable_updates}
            return x, mutable_update_dict
        else:
            x = self.resample_conv(x, train=train)
            return x
