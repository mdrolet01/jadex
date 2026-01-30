from typing import Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from omegaconf import open_dict

from jadex.base.base_state import combine_mutable_updates, get_model_variables_and_mutable
from jadex.distributions.base_distribution import Sample
from jadex.distributions.categorical import (
    Categorical,
    CategoricalSample,
    GRMCKCategorical,
    GumbelSoftmaxCategorical,
    GumbelSoftmaxSample,
)
from jadex.distributions.diagonal_gaussian import DiagonalGaussianConstantVariance, DiagonalGaussianSample
from jadex.networks.autoregressive_sampling import create_predict_fn
from jadex.networks.transformer import Transformer, TransformerConfig
from jadex.networks.variational.constants import CUR_LATENT, GOAL, LATENT_HIST, TEXT
from jadex.networks.variational.variational_network import VariationalNetwork
from jadex.utils import non_pytree
from jadex.utils.printing import print_jit


class TrajGptPolicy(VariationalNetwork):
    """Recognition Model used for all methods (except VAE)"""

    bert_model: nn.Module = non_pytree()
    embed_latent_hist: nn.Embed = non_pytree()
    embed_text: nn.Dense = non_pytree()
    embed_goal: nn.Dense = non_pytree()
    proj_flatcat_token: nn.Dense = non_pytree()
    transformer: Transformer = non_pytree()
    predict_fn: Optional[Callable] = non_pytree()
    latent_output_head: Optional[nn.Dense] = non_pytree()

    @classmethod
    def create_network_kwargs(cls, cfg, input_dists, output_dists, print_info, parent_model_name):
        ##### SETUP TEXT INPUT #####
        text_dist = input_dists[TEXT]
        if text_dist.matches(Categorical):
            from transformers import FlaxDistilBertModel

            bert_model = FlaxDistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
            embed_text = nn.Dense(cfg.transformer.embed_dim)
        else:
            cls.raise_not_supported("input", text_dist, TEXT, print_info)

        ##### SETUP GOAL INPUT #####
        goal_dist = input_dists[GOAL]
        if goal_dist.matches(DiagonalGaussianConstantVariance):
            embed_goal = nn.Dense(cfg.transformer.embed_dim)
        else:
            cls.raise_not_supported("input", goal_dist, GOAL, print_info)

        ##### SETUP LATENT_HIST, CUR_LATENT INPUT/OUPTUT #####
        latent_dist = input_dists[CUR_LATENT]  # or LATENT_HIST

        predict_fn = latent_output_head = None
        if latent_dist.matches([Categorical, GumbelSoftmaxCategorical, GRMCKCategorical]):
            with open_dict(cfg):
                cfg.transformer.encoder_block_size = (
                    goal_dist.shape[-2] + text_dist.shape[-1] + latent_dist.shape[-1]
                )
            transformer_config = TransformerConfig(**cfg.transformer, sample_dist=latent_dist)
            transformer = Transformer(transformer_config, print_info=print_info)
            predict_fn = create_predict_fn(config=transformer_config, print_info=print_info)
            proj_flatcat_token = nn.Dense(cfg.transformer.embed_dim)
        else:
            cls.raise_not_supported("input", latent_dist, f"{CUR_LATENT}/{LATENT_HIST}", print_info)

        embed_latent_hist = nn.Embed(cfg.transformer.vocab_size, cfg.transformer.embed_dim)

        return dict(
            bert_model=bert_model,
            embed_latent_hist=embed_latent_hist,
            embed_text=embed_text,
            embed_goal=embed_goal,
            proj_flatcat_token=proj_flatcat_token,
            transformer=transformer,
            predict_fn=predict_fn,
            latent_output_head=latent_output_head,
        )

    def process_text_samples(self, text: Sample, variables=None, train=True, rng_key=None):
        if text.matches([CategoricalSample]):
            # TODO: Pass attention mask to this model
            attention_mask = jnp.zeros_like(text.value, dtype=jnp.int32)
            text_state = self.bert_model(text.value, attention_mask).last_hidden_state
        else:
            self.raise_not_supported("sample", text, TEXT, self.print_info)

        if variables is not None:
            apply_variables, mutable = get_model_variables_and_mutable(variables, "embed_text")
            text_emb, mutable_updates = self.embed_text.apply(
                apply_variables,
                text_state,
                # train=train,
                # rngs={"dropout": rng_key},
                mutable=mutable,
            )
            mutable_update_dict = {"embed_text": mutable_updates}
            return text_emb, mutable_update_dict
        else:
            text_emb = self.embed_text(text_state)
            return text_emb

    def process_goal_samples(self, goal: Sample, variables=None, train=True, rng_key=None):
        if goal.matches(DiagonalGaussianSample):
            pass
        else:
            self.raise_not_supported("sample", goal, GOAL, self.print_info)

        if variables is not None:
            apply_variables, mutable = get_model_variables_and_mutable(variables, "embed_goal")
            goal_emb, mutable_updates = self.embed_goal.apply(
                apply_variables,
                goal.value,
                # train=train,
                # rngs={"dropout": rng_key},
                mutable=mutable,
            )
            mutable_update_dict = {"embed_goal": mutable_updates}
            return goal_emb, mutable_update_dict
        else:
            goal_emb = self.embed_goal(goal.value)
            return goal_emb

    def process_latent_hist_samples(self, latent_hist: Sample, variables=None, train=True, rng_key=None):
        if latent_hist.matches(CategoricalSample):
            pass
        else:
            self.raise_not_supported("sample", latent_hist, LATENT_HIST, self.print_info)

        if variables is not None:
            apply_variables, mutable = get_model_variables_and_mutable(variables, "embed_latent_hist")
            latent_hist_emb, mutable_updates = self.embed_latent_hist.apply(
                apply_variables,
                latent_hist.value,
                # train=train,
                # rngs={"dropout": rng_key},
                mutable=mutable,
            )
            mutable_update_dict = {"embed_latent_hist": mutable_updates}
            return latent_hist_emb, mutable_update_dict
        else:
            latent_hist_emb = self.embed_latent_hist(latent_hist.value)
            return latent_hist_emb

    def output(self, samples, train):
        ##### PROCESS CUR LATENT #####
        cur_latent = samples[CUR_LATENT]
        self._print_input(CUR_LATENT, cur_latent)
        if cur_latent.matches(CategoricalSample):
            cur_latent = cur_latent.value
        else:
            self.raise_not_supported("sample", cur_latent, CUR_LATENT, self.print_info)

        print_jit(f"{CUR_LATENT} emb", cur_latent.shape, self.print_info)

        ##### PROCESS LATENT HIST #####
        latent_hist = samples[LATENT_HIST]
        self._print_input(LATENT_HIST, latent_hist)
        latent_hist_emb = self.process_latent_hist_samples(latent_hist)
        print_jit(f"{LATENT_HIST} emb", latent_hist_emb.shape, self.print_info)

        ##### PROCESS TEXT #####
        text = samples[TEXT]
        self._print_input(TEXT, text)
        text_emb = self.process_text_samples(text, train=train)
        print_jit(f"{TEXT} emb", text_emb.shape, self.print_info)

        ##### PROCESS GOAL #####
        goal = samples[GOAL]
        self._print_input(GOAL, goal)
        goal_emb = self.process_goal_samples(goal, train=train)
        print_jit(f"{GOAL} emb", goal_emb.shape, self.print_info)

        ##### OUTPUT LATENT #####
        latent_dist = self.output_dists[CUR_LATENT]  # or LATENT_HIST
        if latent_dist.matches([Categorical, GumbelSoftmaxCategorical]):
            context = jnp.concatenate((latent_hist_emb, text_emb, goal_emb), axis=1)
            logits = self.transformer(inputs=context, targets=cur_latent, train=train)
            params = latent_dist.package_params(logits)
            self._print_output(CUR_LATENT, params)
        else:
            self.raise_not_supported("output", latent_dist, CUR_LATENT, self.print_info)

        return params

    def sample_autoregressive(self, variables, inputs, temperature, train, rng_key):
        latent_hist_key, goal_key, text_key, predict_key = jax.random.split(rng_key, 4)
        mutable_updates = {}

        latent_hist_emb, latent_hist_updates = self.process_latent_hist_samples(
            latent_hist=inputs[LATENT_HIST], variables=variables, train=train, rng_key=latent_hist_key
        )
        mutable_updates = combine_mutable_updates(latent_hist_updates, mutable_updates)

        goal_emb, goal_updates = self.process_goal_samples(
            goal=inputs[GOAL], variables=variables, train=train, rng_key=goal_key
        )
        mutable_updates = combine_mutable_updates(goal_updates, mutable_updates)

        text_emb, text_updates = self.process_text_samples(
            text=inputs[TEXT], variables=variables, train=train, rng_key=text_key
        )
        mutable_updates = combine_mutable_updates(text_updates, mutable_updates)

        context = jnp.concatenate((latent_hist_emb, text_emb, goal_emb), axis=1)

        apply_variables, mutable = get_model_variables_and_mutable(variables, "transformer")
        z_given_x_data, logits, tf_updates = self.predict_fn(
            variables=apply_variables,
            train=train,
            mutable=mutable,
            inputs=context,
            temperature=temperature,
            rng_key=predict_key,
        )

        mutable_updates = combine_mutable_updates(tf_updates, mutable_updates, prefix="transformer")

        ##### OUTPUT LATENT #####
        output_dist = self.output_dists[CUR_LATENT]
        z_given_x_params = output_dist.package_params(logits)
        if output_dist.matches(Categorical):
            z_given_x = output_dist.package_sample(value=z_given_x_data)
        elif output_dist.matches([GumbelSoftmaxCategorical, GRMCKCategorical]):
            value = jnp.argmax(z_given_x_data, axis=-1)  # used for non-gradient-based operations
            z_given_x = output_dist.package_sample(value=value, onehot=z_given_x_data)
        else:
            self.raise_not_supported("output", output_dist, CUR_LATENT, self.print_info)

        return z_given_x, z_given_x_params, mutable_updates
