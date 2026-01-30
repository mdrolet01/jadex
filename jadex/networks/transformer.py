from typing import Any, Optional

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import struct
from jax import lax
from omegaconf import DictConfig

from jadex.distributions.base_distribution import BaseDistribution
from jadex.distributions.categorical import Categorical, GRMCKCategorical, GumbelSoftmaxCategorical
from jadex.networks.nn_utils import get_bias_init, get_dtype, get_embed_init, get_kernel_init
from jadex.utils.printing import print_jit


@struct.dataclass
class TransformerConfig:
    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    qkv_dim: int
    mlp_ratio: int
    decoder_block_size: int
    encoder_block_size: int
    grow_target_every: Optional[int] = None
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    decode: bool = False
    sample_dist: Categorical = None
    train_posemb: bool = False
    logits_via_embedding: bool = False
    share_embeddings: bool = False
    kernel_init_name: str = "xavier"
    bias_init_name: str = "normal"
    posemb_init_name: str = "default"
    dtype_name: str = "float32"

    @property
    def use_gumbel(self):
        if self.sample_dist is None:
            return False
        else:
            assert isinstance(self.sample_dist, BaseDistribution)
            return self.sample_dist.matches([GumbelSoftmaxCategorical, GRMCKCategorical])

    @property
    def mlp_dim(self):
        return self.mlp_ratio * self.qkv_dim

    @property
    def kernel_init(self):
        return get_kernel_init(self.kernel_init_name)

    @property
    def bias_init(self):
        return get_bias_init(self.bias_init_name)

    @property
    def posemb_init(self):
        return get_embed_init(self.posemb_init_name)

    @property
    def dtype(self):
        return get_dtype(self.dtype_name)

    @property
    def max_posemb_length(self):
        return max(self.encoder_block_size, self.decoder_block_size)


@struct.dataclass
class TransformerDecoderConfig:
    embed_dim: int
    num_heads: int
    num_layers: int
    qkv_dim: int
    mlp_ratio: int
    decoder_block_size: int
    dropout_rate: float = 0.0
    attention_dropout_rate: float = 0.0
    train_posemb: bool = False
    kernel_init_name: str = "xavier"
    bias_init_name: str = "normal"
    posemb_init_name: str = "default"
    dtype_name: str = "float32"

    @property
    def mlp_dim(self):
        return self.mlp_ratio * self.qkv_dim

    @property
    def kernel_init(self):
        return get_kernel_init(self.kernel_init_name)

    @property
    def bias_init(self):
        return get_bias_init(self.bias_init_name)

    @property
    def posemb_init(self):
        return get_embed_init(self.posemb_init_name)

    @property
    def dtype(self):
        return get_dtype(self.dtype_name)

    @property
    def max_posemb_length(self):
        return self.decoder_block_size


def shift_right(x, axis=1):
    """Shift the input to the right by padding on axis 1."""
    pad_widths = [(0, 0)] * len(x.shape)
    pad_widths[axis] = (1, 0)
    padded = jnp.pad(x, pad_widths, mode="constant", constant_values=x.dtype.type(0))
    return padded[:, :-1]


def sinusoidal_init(max_len, min_scale=1.0, max_scale=10000.0):
    """1D Sinusoidal Position Embedding Initializer.

    Args:
        max_len: maximum possible length for the input.
        min_scale: float: minimum frequency-scale in sine grating.
        max_scale: float: maximum frequency-scale in sine grating.

    Returns:
        output: init function returning `(1, max_len, d_feature)`
    """

    def init(key, shape, dtype=np.float32):
        """Sinusoidal init."""
        del key, dtype
        d_feature = shape[-1]
        pe = np.zeros((max_len, d_feature), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
        div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
        pe[:, : d_feature // 2] = np.sin(position * div_term)
        pe[:, d_feature // 2 : 2 * (d_feature // 2)] = np.cos(position * div_term)
        pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
        return jnp.array(pe)

    return init


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      decode: whether to run in single-position autoregressive mode.
    """

    config: TransformerConfig | TransformerDecoderConfig
    decode: bool = False

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, inputs_positions=None):
        """Applies AddPositionEmbs module.

        By default this layer uses a fixed sinusoidal embedding table. If a
        learned position embedding is desired, pass an initializer to
        posemb_init in the configuration.

        Args:
          inputs: input data.
          inputs_positions: input position indices for packed sequences.

        Returns:
          output: `(bs, timesteps, in_dim)`
        """
        cfg = self.config
        # inputs.shape is (batch_size, seq_len, embed_dim)
        assert inputs.ndim == 3, "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        length = inputs.shape[1]
        max_len = cfg.max_posemb_length
        pos_emb_shape = (1, max_len, inputs.shape[-1])
        if cfg.train_posemb:
            pos_embedding = self.param("pos_embedding", cfg.posemb_init, pos_emb_shape)
        else:
            # Use a fixed (non-learned) sinusoidal position embedding.
            pos_embedding = sinusoidal_init(max_len=max_len)(None, pos_emb_shape, None)

        pe = pos_embedding[:, :length, :]

        # We use a cache position index for tracking decoding position.
        if self.decode:
            is_initialized = self.has_variable("cache", "cache_index")
            cache_index = self.variable("cache", "cache_index", lambda: jnp.array(0, dtype=jnp.uint32))
            if is_initialized:
                i = cache_index.value
                cache_index.value = i + 1
                _, _, df = pos_embedding.shape
                pe = lax.dynamic_slice(pos_embedding, jnp.array((0, i, 0)), (1, 1, df))
        if inputs_positions is None:
            # normal unpacked case:
            return inputs + pe
        else:
            # for packed data we need to use known position indices:
            return inputs + jnp.take(pe[0], inputs_positions, axis=0)


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      out_dim: optionally specify out dimension.
    """

    config: TransformerConfig
    out_dim: Optional[int] = None

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool):
        """Applies Transformer MlpBlock module."""
        cfg = self.config
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = nn.Dense(cfg.mlp_dim, dtype=cfg.dtype, kernel_init=cfg.kernel_init, bias_init=cfg.bias_init)(
            inputs
        )
        x = nn.relu(x)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
        output = nn.Dense(
            actual_out_dim, dtype=cfg.dtype, kernel_init=cfg.kernel_init, bias_init=cfg.bias_init
        )(x)
        output = nn.Dropout(rate=cfg.dropout_rate)(output, deterministic=not train)
        return output


class Encoder1DBlock(nn.Module):
    """Transformer encoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig | TransformerDecoderConfig

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool, encoder_mask=None):
        """Applies Encoder1DBlock module.

        Args:
          inputs: input data.
          encoder_mask: encoder self-attention mask.

        Returns:
          output after transformer encoder block.
        """
        cfg = self.config

        # Attention block.
        assert inputs.ndim == 3
        x = nn.LayerNorm(dtype=cfg.dtype)(inputs)
        x = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            qkv_features=cfg.qkv_dim,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=cfg.attention_dropout_rate,
            deterministic=not train,
        )(inputs_q=x, mask=encoder_mask)

        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
        x = x + inputs

        # MLP block.
        y = nn.LayerNorm(dtype=cfg.dtype)(x)
        y = MlpBlock(config=cfg)(y, train=train)

        return x + y


class EncoderDecoder1DBlock(nn.Module):
    """Transformer encoder-decoder layer.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig

    @nn.compact
    def __call__(
        self,
        targets: jnp.ndarray,
        encoded: jnp.ndarray,
        train: bool,
        decoder_mask=None,
        encoder_decoder_mask=None,
    ):
        """Applies EncoderDecoder1DBlock module.

        Args:
          targets: input data for decoder
          encoded: input data from encoder
          decoder_mask: decoder self-attention mask.
          encoder_decoder_mask: encoder-decoder attention mask.

        Returns:
          output after transformer encoder-decoder block.
        """
        cfg = self.config

        # Decoder block.
        assert targets.ndim == 3
        x = nn.LayerNorm(dtype=cfg.dtype)(targets)
        x = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            qkv_features=cfg.qkv_dim,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=cfg.attention_dropout_rate,
            deterministic=not train,
            decode=cfg.decode,
        )(inputs_q=x, mask=decoder_mask)

        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
        x = x + targets

        # Encoder-Decoder block.
        y = nn.LayerNorm(dtype=cfg.dtype)(x)
        y = nn.MultiHeadDotProductAttention(
            num_heads=cfg.num_heads,
            dtype=cfg.dtype,
            qkv_features=cfg.qkv_dim,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
            use_bias=False,
            broadcast_dropout=False,
            dropout_rate=cfg.attention_dropout_rate,
            deterministic=not train,
        )(inputs_q=y, inputs_k=encoded, inputs_v=encoded, mask=encoder_decoder_mask)

        y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=not train)
        y = y + x

        # MLP block.
        z = nn.LayerNorm(dtype=cfg.dtype)(y)
        z = MlpBlock(config=cfg)(z, train=train)

        return y + z


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      shared_embedding: a shared embedding layer to use.
    """

    config: TransformerConfig
    shared_embedding: Any = None
    print_info: dict = DictConfig({"name": "Transformer_Encoder", "uuid": "TRANSFORMER_ENC"})

    @nn.compact
    def __call__(self, inputs: jnp.ndarray, train: bool, inputs_positions=None, encoder_mask=None):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data
          inputs_positions: input subsequence positions for packed examples.
          encoder_mask: decoder self-attention mask.

        Returns:
          output of a transformer encoder.
        """
        cfg = self.config

        if inputs.ndim == 2:
            assert inputs.dtype == jnp.int32
            # Input Embedding
            if self.shared_embedding is None:
                input_embed = nn.Embed(
                    num_embeddings=cfg.vocab_size,
                    features=cfg.embed_dim,
                    embedding_init=nn.initializers.normal(stddev=1.0),
                )
            else:
                input_embed = self.shared_embedding
            x = input_embed(inputs)
            print_jit(f"transformer encoder input embed", x.shape, self.print_info)
        else:
            assert inputs.dtype == jnp.float32
            assert inputs.ndim == 3  # (batch, len, embed_dim)
            x = inputs

        x = AddPositionEmbs(config=cfg, decode=False)(x, inputs_positions=inputs_positions)
        x = nn.Dropout(rate=cfg.dropout_rate)(x, deterministic=not train)
        x = x.astype(cfg.dtype)

        # Input Encoder
        print_jit(f"transformer encoder input to {cfg.num_layers} layers", x.shape, self.print_info)
        for lyr in range(cfg.num_layers):
            x = Encoder1DBlock(config=cfg)(inputs=x, train=train, encoder_mask=encoder_mask)

        encoded = nn.LayerNorm(dtype=cfg.dtype)(x)

        return encoded


class Decoder(nn.Module):
    """Transformer Model Decoder for sequence to sequence translation.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
      shared_embedding: a shared embedding layer to use.
    """

    config: TransformerConfig
    shared_embedding: Any = None
    print_info: dict = DictConfig({"name": "Transformer_Decoder", "uuid": "TRANSFORMER_DEC"})

    @nn.compact
    def __call__(
        self,
        encoded: jnp.ndarray,
        targets: jnp.ndarray,
        train: bool,
        targets_positions=None,
        decoder_mask=None,
        encoder_decoder_mask=None,
    ):
        """Applies Transformer model on the inputs.

        NOTE: If this decoder is used for autoregressive sampling, then caution must be taken when using BatchNorm.
        BatchNorm/mutables are not properly setup for autoregressive sampling, but a possible workaround is to
        to always have use_running_average=False.

        Args:
          encoded: encoded input data from encoder.
          targets: target inputs.
          targets_positions: input subsequence positions for packed examples.
          decoder_mask: decoder self-attention mask.
          encoder_decoder_mask: encoder-decoder attention mask.

        Returns:
          output of a transformer decoder.
        """
        cfg = self.config

        assert encoded.ndim == 3  # (batch, len, depth)

        # Target Embedding
        if self.shared_embedding is None:
            output_embed = nn.Embed(
                num_embeddings=cfg.vocab_size,
                features=cfg.embed_dim,
                embedding_init=nn.initializers.normal(stddev=1.0),
            )
        else:
            output_embed = self.shared_embedding

        if cfg.use_gumbel:
            assert targets.ndim == 3  # (batch, len, vocab size)
            assert targets.dtype == jnp.float32
            if not cfg.decode:
                # shift right
                B, T, K = targets.shape
                zero_pad = jnp.zeros((B, 1, K))
                zero_pad = zero_pad.at[..., 0].set(1)
                onehot_y = jnp.concatenate((zero_pad, targets[:, 1:, :]), axis=1)
            else:
                onehot_y = targets
            all_embs = output_embed(jnp.arange(cfg.vocab_size))
            y = jnp.einsum("b t v, v c -> b t c", onehot_y, all_embs)
        else:
            assert targets.ndim == 2  # (batch, len)
            y = targets.astype("int32")
            if not cfg.decode:
                y = shift_right(y)
            y = output_embed(y)

        y = AddPositionEmbs(config=cfg, decode=cfg.decode)(y, inputs_positions=targets_positions)
        y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=not train)
        y = y.astype(cfg.dtype)
        print_jit(f"transformer decoder target embed", y.shape, self.print_info)

        # Target-Input Decoder
        print_jit(
            f"transformer encoder/decoder input to {cfg.num_layers} layers",
            (encoded.shape, y.shape),
            self.print_info,
        )
        for lyr in range(cfg.num_layers):
            y = EncoderDecoder1DBlock(config=cfg)(
                targets=y,
                encoded=encoded,
                train=train,
                decoder_mask=decoder_mask,
                encoder_decoder_mask=encoder_decoder_mask,
            )

        y = nn.LayerNorm(dtype=cfg.dtype)(y)

        # Decoded Logits
        if cfg.logits_via_embedding:
            # Use the transpose of embedding matrix for logit transform.
            logits = output_embed.attend(y.astype(jnp.float32))
            # Correctly normalize pre-softmax logits for this shared case.
            logits = logits / jnp.sqrt(y.shape[-1])
        else:
            logits = nn.Dense(
                cfg.vocab_size,
                dtype=cfg.dtype,
                kernel_init=cfg.kernel_init,
                bias_init=cfg.bias_init,
            )(y)
            print_jit(f"transformer decoder logits output", logits.shape, self.print_info)
        return logits


class Transformer(nn.Module):
    """Transformer Model for sequence to sequence translation.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerConfig
    print_info: dict = DictConfig({"name": "Transformer", "uuid": "TRANSFORMER"})

    def setup(self):
        cfg = self.config

        if cfg.share_embeddings:
            self.shared_embedding = nn.Embed(
                num_embeddings=cfg.vocab_size,
                features=cfg.embed_dim,
                embedding_init=nn.initializers.normal(stddev=1.0),
            )
        else:
            self.shared_embedding = None

        self.encoder = Encoder(
            config=cfg, shared_embedding=self.shared_embedding, print_info=self.print_info
        )

        self.decoder = Decoder(
            config=cfg, shared_embedding=self.shared_embedding, print_info=self.print_info
        )

    def encode(self, inputs: jnp.ndarray, train: bool, inputs_positions=None, inputs_segmentation=None):
        """Applies Transformer encoder-branch on the inputs.

        Args:
          inputs: input data.
          inputs_positions: input subsequence positions for packed examples.
          inputs_segmentation: input segmentation info for packed examples.

        Returns:
          encoded feature array from the transformer encoder.
        """
        cfg = self.config
        print_jit(f"transformer ENCODER receieved inputs", inputs.shape, self.print_info)

        # Make padding attention mask.
        if inputs.ndim == 2:
            # token id inputs
            assert inputs.dtype == jnp.int32
            input_cond = inputs > 0
        else:
            # embedding inputs
            assert inputs.dtype == jnp.float32
            assert inputs.ndim == 3
            input_cond = jnp.ones(inputs.shape[:2], dtype=bool)

        encoder_mask = nn.make_attention_mask(input_cond, input_cond, dtype=cfg.dtype)

        # Add segmentation block-diagonal attention mask if using segmented data.
        if inputs_segmentation is not None:
            encoder_mask = nn.combine_masks(
                encoder_mask,
                nn.make_attention_mask(inputs_segmentation, inputs_segmentation, jnp.equal, dtype=cfg.dtype),
            )

        return self.encoder(
            inputs=inputs,
            train=train,
            inputs_positions=inputs_positions,
            encoder_mask=encoder_mask,
        )

    def decode(
        self,
        encoded: jnp.ndarray,
        inputs: jnp.ndarray,  # only needed for masks
        targets: jnp.ndarray,
        train: bool,
        targets_positions=None,
        inputs_segmentation=None,
        targets_segmentation=None,
    ):
        """Applies Transformer decoder-branch on encoded-input and target.

        Args:
          encoded: encoded input data from encoder.
          inputs: input data (only needed for masking).
          targets: target data.
          targets_positions: target subsequence positions for packed examples.
          inputs_segmentation: input segmentation info for packed examples.
          targets_segmentation: target segmentation info for packed examples.

        Returns:
          logits array from transformer decoder.
        """
        cfg = self.config
        print_jit(f"transformer DECODER receieved encoded", encoded.shape, self.print_info)
        print_jit(f"transformer DECODER receieved targets", targets.shape, self.print_info)

        # Padding condition/mask for input
        if inputs.ndim == 2:
            # if input is token sequence (translation task)
            assert inputs.dtype == jnp.int32
            input_cond = inputs > 0
        else:
            # if input is embedding sequence (e.g. vision transformer)
            assert inputs.dtype == jnp.float32
            assert inputs.ndim == 3
            input_cond = jnp.ones(inputs.shape[:2], dtype=bool)

        # Make padding attention masks. Determine whether inputs are onehot (Gumbel) or not
        target_idxs = lax.stop_gradient(jnp.argmax(targets, axis=-1)) if cfg.use_gumbel else targets

        # Fast autoregressive decoding: use encoder-decoder mask only
        if cfg.decode:
            decoder_mask = None
            encoder_decoder_mask = nn.make_attention_mask(
                jnp.ones_like(target_idxs) > 0, input_cond, dtype=cfg.dtype
            )
        else:
            decoder_mask = nn.combine_masks(
                nn.make_attention_mask(target_idxs > 0, target_idxs > 0, dtype=cfg.dtype),
                nn.make_causal_mask(target_idxs, dtype=cfg.dtype),
            )
            encoder_decoder_mask = nn.make_attention_mask(target_idxs > 0, input_cond, dtype=cfg.dtype)

        # Add segmentation block-diagonal attention masks if using segmented data.
        if inputs_segmentation is not None:
            decoder_mask = nn.combine_masks(
                decoder_mask,
                nn.make_attention_mask(
                    targets_segmentation, targets_segmentation, jnp.equal, dtype=cfg.dtype
                ),
            )
            encoder_decoder_mask = nn.combine_masks(
                encoder_decoder_mask,
                nn.make_attention_mask(
                    targets_segmentation, inputs_segmentation, jnp.equal, dtype=cfg.dtype
                ),
            )

        logits = self.decoder(
            encoded=encoded,
            targets=targets,
            train=train,
            targets_positions=targets_positions,
            decoder_mask=decoder_mask,
            encoder_decoder_mask=encoder_decoder_mask,
        )

        return logits.astype(self.config.dtype)

    def __call__(
        self,
        inputs: jnp.ndarray,
        targets: jnp.ndarray,
        train: bool,
        inputs_positions=None,
        targets_positions=None,
        inputs_segmentation=None,
        targets_segmentation=None,
    ):
        """Applies Transformer model on the inputs.

        Args:
          inputs: input data.
          targets: target data.
          inputs_positions: input subsequence positions for packed examples.
          targets_positions: target subsequence positions for packed examples.
          inputs_segmentation: input segmentation info for packed examples.
          targets_segmentation: target segmentation info for packed examples.

        Returns:
          logits array from full transformer.
        """

        encoded = self.encode(
            inputs=inputs,
            train=train,
            inputs_positions=inputs_positions,
            inputs_segmentation=inputs_segmentation,
        )

        return self.decode(
            encoded=encoded,
            inputs=inputs,  # only used for masks
            targets=targets,
            train=train,
            targets_positions=targets_positions,
            inputs_segmentation=inputs_segmentation,
            targets_segmentation=targets_segmentation,
        )


class TransformerDecoder(nn.Module):
    """Transformer Decoder Model for continuous (vector-based) inputs.

    Attributes:
      config: TransformerConfig dataclass containing hyperparameters.
    """

    config: TransformerDecoderConfig

    @nn.compact
    def __call__(self, target_embs: jnp.ndarray, train: bool):
        """Applies Transformer model on the target embeddings.

        Args:
          target_embs: embedded targets.

        Returns:
          output of a transformer decoder.
        """

        cfg = self.config

        y = AddPositionEmbs(config=cfg, decode=False)(target_embs)
        y = nn.Dropout(rate=cfg.dropout_rate)(y, deterministic=not train)
        y = y.astype(cfg.dtype)

        # Target Decoder
        for lyr in range(cfg.num_layers):
            y = Encoder1DBlock(config=cfg)(inputs=y, train=train)

        y = nn.LayerNorm(dtype=cfg.dtype)(y)

        y = nn.Dense(
            cfg.embed_dim,
            dtype=cfg.dtype,
            kernel_init=cfg.kernel_init,
            bias_init=cfg.bias_init,
        )(y)

        return y
