import jax
import jax.numpy as jnp


def clamp_probs(probs: jnp.ndarray) -> jnp.ndarray:
    """Clamp probabilities to avoid numerical instability."""
    eps = jnp.finfo(probs.dtype).eps
    return jnp.clip(probs, eps, 1 - eps)


def logits_to_probs(logits, is_binary=False, axis=-1):
    """Convert logits to probabilities."""

    if is_binary:
        return jax.nn.sigmoid(logits)
    return jax.nn.softmax(logits, axis=axis)


def probs_to_logits(probs, is_binary=False):
    """Convert probabilities to logits."""
    ps_clamped = clamp_probs(probs)
    if is_binary:
        return jnp.log(ps_clamped) - jnp.log1p(-ps_clamped)
    return jnp.log(ps_clamped)
