"""
Fast Autoregressive Sampling of Transformer

Code is mostly sourced from:
    https://colab.research.google.com/gist/thisiscam/f3d849ff989ecc504681aeb52b0c13f1/transformer-ar-cache.ipynb

"""

import math
from typing import Any, Callable, List, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey
from flax.typing import VariableDict
from jax import lax

from jadex.distributions.categorical import Categorical
from jadex.networks.transformer import Transformer, TransformerConfig

TokenToLogits = Callable[[chex.Array, chex.ArrayTree], chex.Array]
SampleStepFn = Callable[[chex.ArrayTree, int], Tuple[chex.ArrayTree, chex.ArrayTree]]

PAD_ID = 0
EXPECTED_AUTOREGRESSIVE_MUTABLE_KEYS = ["batch_stats", "cache"]


def _get_mutable_updates_for_model(all_mutables: dict, model_name: str):
    model_mutables = {}
    assert isinstance(all_mutables, dict)
    for mutable_key, val_dict in all_mutables.items():
        if mutable_key in EXPECTED_AUTOREGRESSIVE_MUTABLE_KEYS and mutable_key != "cache":
            assert isinstance(val_dict, dict)
            if model_name in val_dict.keys():
                model_mutables[mutable_key] = val_dict[model_name]

    return model_mutables


def sample(
    batch_size: int,
    max_decode_len: int,
    vocab_size: int,
    tokens_to_logits: TokenToLogits,
    prng_key: chex.PRNGKey,
    vars_with_cache: chex.ArrayTree,
    sample_dist: Categorical,
    use_gumbel: bool,
    temperature: float = 1.0,
    start_id: Optional[Union[int, chex.Array]] = PAD_ID,
    return_logits: bool = False,
):
    """
    Temperature-based sampling for autoregressive generation.

    Args:
        batch_size: Number of sequences to sample.
        max_decode_len: Maximum length to decode.
        vocab_size: Size of the output vocabulary.
        tokens_to_logits: Function mapping tokens and cache to logits and updated cache.
        prng_key: JAX PRNG key for randomness.
        cache: Initial attention cache.
        use_gumbel: Whether to use Gumbel sampling.
        temperature: Softmax temperature; lower values make output more deterministic.
        start_id: Initial token(s) or token distribution.
        return_logits: If True, also return logits from each step.

    Returns:
        Tuple of (updated cache, sampled sequences, logits if requested)
    """

    # Initialize first token
    if isinstance(start_id, int):
        if use_gumbel:
            token = jnp.zeros((batch_size, vocab_size))
            token = token.at[..., start_id].set(1)
        else:
            token = jnp.full((batch_size,), start_id, dtype=jnp.int32)
    else:
        token = start_id

    chex.assert_shape(token, (batch_size, vocab_size) if use_gumbel else (batch_size,))

    def sampling_step(carry, _):
        vars_with_cache, cur_token, rng = carry
        next_rng, logits_rng, samp_rng = jax.random.split(rng, 3)
        logits, new_vars_with_cache = tokens_to_logits(cur_token, vars_with_cache, logits_rng)

        sample = sample_dist.sample_single_categorical(logits, temperature, samp_rng)
        next_token = sample.onehot if use_gumbel else sample.value

        output = (next_token, logits) if return_logits else (next_token,)
        return (new_vars_with_cache, next_token, next_rng), output

    carry_init = (vars_with_cache, token, prng_key)
    carry_final, outputs = lax.scan(sampling_step, carry_init, None, length=max_decode_len)

    final_vars_with_cache = carry_final[0]
    outputs = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), outputs)
    sequences = outputs[0]
    logits_out = outputs[1] if return_logits else None

    return final_vars_with_cache, sequences, logits_out


def loop_helper(
    sample_step_fn: SampleStepFn, start_state: chex.ArrayTree, length: int = 1, length_axis: int = 1
):
    """A decodeing loop helper.

    This function is similar to `jax.lax.scan`, except:
      1) The computation is done entirely in Python
      2) It only returns the output (not the carry)
      3) It only takes in a length parameter.
      4) The outputs for each call to `sample_step_fn` is joined on along the
        `length_axis`.

    The purpose of the function is to wrap any of the decoding methods, so that
    the cache size can grow dynamically within a single end-to-end decode.
    """
    outputs = []
    state = start_state
    for _ in range(length):
        (state, output) = sample_step_fn(state)
        outputs.append(output)

    concat_outputs = jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=length_axis), *outputs)
    new_vars_with_cache = state[2]
    return concat_outputs, new_vars_with_cache


def update_leftmost(cache: chex.ArrayTree, prev_cache: chex.ArrayTree) -> chex.ArrayTree:
    """Updates cache by prev_cache at the leftmost dimension."""
    chex.assert_trees_all_equal_structs(cache, prev_cache)

    def write_leftmost(cache, prev_cache):
        return cache.at[tuple(slice(0, d) for d in prev_cache.shape)].set(prev_cache)

    return jax.tree.map(write_leftmost, cache, prev_cache)


def create_predict_fn(config: TransformerConfig, print_info=None):
    config = config.replace(decode=True)
    transformer = Transformer(config, print_info)

    def predict_fn(
        variables: VariableDict,
        inputs: jnp.ndarray,
        temperature: float,
        train: bool,
        rng_key: PRNGKey,
        mutable: List,
    ):
        encoded_inputs, enc_mutable_updates = transformer.apply(
            variables,
            inputs=inputs,
            train=train,
            method=Transformer.encode,
            rngs={"dropout": rng_key},
            mutable=mutable,
        )

        enc_mutable_updates = _get_mutable_updates_for_model(enc_mutable_updates, "encoder")

        def initialize_vars_with_cache(max_len):
            batch_size = encoded_inputs.shape[0]
            target_shape = (batch_size, max_len)
            if config.use_gumbel:
                target_shape += (config.vocab_size,)

            init_vars_with_cache = transformer.init(
                jax.random.PRNGKey(0),
                inputs=jnp.zeros(encoded_inputs.shape),
                targets=jnp.zeros(target_shape),
                train=train,
            )
            # remove params, since we will use the params from "variables"
            init_vars_with_cache.pop("params")
            return init_vars_with_cache

        def tokens_to_logits(tokens, vars_with_cache, rng_key):
            logits, new_vars_with_cache = transformer.apply(
                {"params": variables["params"], **vars_with_cache},
                encoded=encoded_inputs,
                inputs=inputs,
                targets=jnp.expand_dims(tokens, axis=-2 if config.use_gumbel else -1),
                rngs={"dropout": rng_key},
                train=train,
                method=Transformer.decode,
                mutable=mutable + ["cache"],
            )

            return logits.squeeze(axis=1), new_vars_with_cache

        steps = int(math.ceil(config.decoder_block_size / config.grow_target_every))
        prng_keys = jax.random.split(rng_key, steps)

        def sample_step(state):
            i, idx, prev_vars_with_cache, prev_tok = state
            end = min(idx + config.grow_target_every, config.decoder_block_size)

            cur_vars_with_cache = initialize_vars_with_cache(end)
            cur_vars_with_cache["cache"] = update_leftmost(
                cur_vars_with_cache["cache"], prev_vars_with_cache["cache"]
            )

            next_vars_with_cache, seqs, logits = sample(
                batch_size=encoded_inputs.shape[0],
                max_decode_len=end - idx,
                vocab_size=config.vocab_size,
                tokens_to_logits=tokens_to_logits,
                prng_key=prng_keys[i],
                vars_with_cache=cur_vars_with_cache,
                sample_dist=config.sample_dist,
                use_gumbel=config.use_gumbel,
                temperature=temperature,
                start_id=prev_tok,
                return_logits=True,
            )

            next_tok = seqs[:, -1, :] if config.use_gumbel else seqs[:, -1]

            return (i + 1, end, next_vars_with_cache, next_tok), (seqs, logits)

        init_vars_with_cache = initialize_vars_with_cache(config.grow_target_every)

        (seqs, logits), new_vars_with_cache = loop_helper(
            sample_step, (0, 0, init_vars_with_cache, PAD_ID), steps
        )

        # NOTE: There probably shouldn't be any mutable updates (batch_stats) for the decoder.
        # Long story short: Autoregressive sampling with caching makes this tricky.
        dec_mutable_updates = _get_mutable_updates_for_model(new_vars_with_cache, "decoder")

        mutable_update_dict = {"encoder": enc_mutable_updates, "decoder": dec_mutable_updates}

        return seqs, logits, mutable_update_dict

    return predict_fn
