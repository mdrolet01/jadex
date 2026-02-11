from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from jadex.base.base_state import combine_mutable_updates, get_model_variables_and_mutable
from jadex.distributions.base_distribution import Sample
from jadex.distributions.categorical import (
    Categorical,
    CategoricalSample,
    GRMCKCategorical,
    GumbelSoftmaxCategorical,
    GumbelSoftmaxSample,
)
from jadex.distributions.uniform import UniformSample
from jadex.networks.autoregressive_sampling import create_predict_fn
from jadex.networks.diffusion_transformer import DiT
from jadex.networks.transformer import Transformer, TransformerConfig
from jadex.networks.variational.constants import LABEL, LATENT, TIME
from jadex.networks.variational.mixins.embed_latent_mixin import EmbedLatentMixin
from jadex.networks.variational.variational_network import VariationalNetwork
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit


class LabelTransformerRecognitionModel(VariationalNetwork):
    embed_label: nn.Embed = non_pytree()
    transformer: Transformer = non_pytree()
    predict_fn: Callable = non_pytree()

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP LATENT INPUT #####
        latent_dist = input_dists[LATENT]
        if latent_dist.matches([Categorical, GumbelSoftmaxCategorical]):
            pass
        else:
            cls.raise_not_supported("input", latent_dist, LATENT, print_info)

        ##### SETUP LABEL INPUT #####
        label_dist = input_dists[LABEL]
        if label_dist.matches(Categorical):
            embed_label = nn.Embed(label_dist.num_classes, cfg.transformer.embed_dim)
        else:
            cls.raise_not_supported("input", label_dist, LABEL, print_info)

        ##### SETUP LATENT OUTPUT #####
        latent_dist = output_dists[LATENT]
        if latent_dist.matches([Categorical, GumbelSoftmaxCategorical]):
            transformer_config = TransformerConfig(**cfg.transformer, sample_dist=latent_dist)
            transformer = Transformer(transformer_config, print_info=print_info)
            predict_fn = create_predict_fn(config=transformer_config, print_info=print_info)
        else:
            cls.raise_not_supported("output", latent_dist, LATENT, print_info)

        return dict(
            embed_label=embed_label,
            transformer=transformer,
            predict_fn=predict_fn,
        )

    def output(self, samples, train):
        ##### PROCESS LATENT #####
        latent = samples[LATENT]
        self._print_input(LATENT, latent)
        if latent.matches(CategoricalSample):
            latent = latent.value
        elif latent.matches(GumbelSoftmaxSample):
            latent = latent.onehot
        else:
            self.raise_not_supported("sample", latent, LATENT, self.print_info)

        ##### PROCESS LABEL #####
        label = samples[LABEL]
        self._print_input(LABEL, label)
        label_emb = self.process_label_samples(label=label, train=train)

        ##### OUTPUT LATENT #####
        output_dist = self.output_dists[LATENT]
        if output_dist.matches(Categorical):
            logits = self.transformer(inputs=label_emb, targets=latent, train=train)
            print_jit("transformer final", latent.shape, self.print_info)
            params = output_dist.package_params(logits)
            self._print_output(LATENT, params)
        else:
            self.raise_not_supported("output", output_dist, LATENT, self.print_info)

        return params

    def process_label_samples(self, label: Sample, variables=None, train=True, rng_key=None):
        if label.matches(CategoricalSample):
            label = label.value
        else:
            self.raise_not_supported("sample", label, LABEL, self.print_info)

        if variables is not None:
            apply_variables, mutable = get_model_variables_and_mutable(variables, "embed_label")
            label_emb, embed_mutable_updates = self.embed_label.apply(
                apply_variables,
                label,
                # train=train,
                # rngs={"dropout": rng_key},
                mutable=mutable,
            )
            process_latent_mutable_update_dict = {"embed_label": embed_mutable_updates}
            return label_emb, process_latent_mutable_update_dict
        else:
            label_emb = self.embed_label(label)
            return label_emb

    def sample_autoregressive(self, variables, inputs, temperature, train, rng_key):
        x_key, predict_key = jax.random.split(rng_key)
        mutable_updates = {}

        ##### PROCESS LABEL #####
        label_emb, process_label_mutable_update_dict = self.process_label_samples(
            inputs[LABEL], variables, train, x_key
        )

        mutable_updates = combine_mutable_updates(process_label_mutable_update_dict, mutable_updates)

        apply_variables, mutable = get_model_variables_and_mutable(variables, "transformer")
        z_given_x_data, logits, tf_mutable_update_dict = self.predict_fn(
            variables=apply_variables,
            train=train,
            mutable=mutable,
            inputs=label_emb,
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


class LabelTransformerDiscreteFlowModel(VariationalNetwork, EmbedLatentMixin):
    transformer: DiT = non_pytree()
    output_head: nn.Dense = non_pytree()

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP LATENT INPUT #####
        latent_kwargs = cls.setup_latent_input(input_dists[LATENT], cfg, print_info)

        ##### SETUP LABEL INPUT #####
        label_dist = input_dists[LABEL]
        if label_dist.matches(Categorical):
            pass
        else:
            cls.raise_not_supported("input", label_dist, LABEL, print_info)

        ##### SETUP LATENT OUTPUT #####
        latent_dist = output_dists[LATENT]
        if latent_dist.matches([Categorical, GumbelSoftmaxCategorical]):
            transformer = DiT(
                hidden_size=cfg.transformer.hidden_size,
                num_classes=label_dist.num_classes,
                num_layers=cfg.transformer.num_layers,
                num_heads=cfg.transformer.num_heads,
                mlp_ratio=cfg.transformer.mlp_ratio,
                class_dropout_prob=cfg.transformer.class_dropout_prob,
            )
            output_head = nn.Dense(cfg.vocab_size)
        else:
            cls.raise_not_supported("output", latent_dist, LATENT, print_info)

        return dict(transformer=transformer, output_head=output_head, **latent_kwargs)

    def output(self, samples, train):
        ##### PROCESS LATENT #####
        latent_emb = self.process_latent_samples(samples[LATENT], train=train)

        ##### PROCESS TIME #####
        t = samples[TIME]
        self._print_input(TIME, t)
        if t.matches(UniformSample):
            t = rearrange(t.value, "b 1 -> b")
        else:
            self.raise_not_supported("sample", t, TIME, self.print_info)

        ##### PROCESS LABEL #####
        label = samples[LABEL]
        self._print_input(LABEL, label)
        if label.matches(CategoricalSample):
            label = rearrange(label.value, "b 1 -> b")
        else:
            self.raise_not_supported("sample", label, LABEL, self.print_info)

        ##### OUTPUT LATENT #####
        output_dist = self.output_dists[LATENT]
        if output_dist.matches([Categorical, GumbelSoftmaxCategorical]):
            output = self.transformer(x=latent_emb, t=t, y=label, train=train)
            logits = self.output_head(output)
            params = output_dist.package_params(logits)
            self._print_output(LATENT, params)
        else:
            self.raise_not_supported("output", output_dist, LATENT, self.print_info)

        return params
