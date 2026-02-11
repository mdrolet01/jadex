import flax.linen as nn
import jax.numpy as jnp
from jax.lax import stop_gradient as sg

MIN_VAR = 1e-5


class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and variance for input normalization."""

    @staticmethod
    def inverse_scale(maybe_normalized_x, mean, var, should_norm_mask=None):
        var = jnp.where(var < MIN_VAR, 1.0, var)
        apply_inverse_scalerd_x = maybe_normalized_x * sg(jnp.sqrt(var)) + sg(mean)
        if should_norm_mask is None:
            x = apply_inverse_scalerd_x
        else:
            should_norm = should_norm_mask.reshape([1] * (maybe_normalized_x.ndim - 1) + [-1])
            x = jnp.where(should_norm, apply_inverse_scalerd_x, maybe_normalized_x)
        return x

    @staticmethod
    def scale(x, mean, var, should_norm_mask=None):
        var = jnp.where(var < MIN_VAR, 1.0, var)
        normalized_x = (x - sg(mean)) / sg(jnp.sqrt(var))
        if should_norm_mask is None:
            maybe_normalized_x = normalized_x
        else:
            should_norm = should_norm_mask.reshape([1] * (x.ndim - 1) + [-1])
            maybe_normalized_x = jnp.where(should_norm, normalized_x, x)
        return maybe_normalized_x

    @staticmethod
    def compute_new_stats(cur_mean, cur_var, cur_count, batch_mean, batch_var, batch_count):
        # Update counts
        updated_count = cur_count + batch_count

        # Numerically stable mean and variance update
        delta = batch_mean - cur_mean
        new_mean = cur_mean + delta * batch_count / updated_count

        # Compute the new variance using Welford's method
        m_a = cur_var * cur_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * cur_count * batch_count / updated_count
        new_var = M2 / updated_count

        return new_mean, new_var, updated_count

    @nn.compact
    def __call__(self, x, should_norm_mask=None, update=True):
        x = jnp.atleast_2d(x)

        # Initialize running mean, variance, and count
        mean = self.variable("run_stats", "mean", lambda: jnp.zeros(x.shape[-1]))
        var = self.variable("run_stats", "var", lambda: jnp.ones(x.shape[-1]))
        count = self.variable("run_stats", "count", lambda: jnp.array(1e-6))

        if update:
            # Compute batch mean and variance
            batch_mean = jnp.mean(x, axis=tuple(range(x.ndim - 1)))
            batch_var = jnp.var(x, axis=tuple(range(x.ndim - 1)))
            batch_count = x.shape[0]

            new_mean, new_var, updated_count = self.compute_new_stats(
                mean.value, var.value, count.value, batch_mean, batch_var, batch_count
            )
        else:
            new_mean = mean.value
            new_var = var.value

        # Standardize input
        scaled_x = self.scale(x, new_mean, new_var, should_norm_mask)

        # Update state variables
        if update:
            mean.value = new_mean
            var.value = new_var
            count.value = updated_count

        return scaled_x
