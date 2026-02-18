# Copyright 2023 Meta AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax DINOv2 model."""

import collections.abc
import math
from typing import Optional, Sequence

import numpy as np
from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers import Dinov2WithRegistersModel, Dinov2WithRegistersConfig
from transformers.modeling_flax_outputs import (
    FlaxBaseModelOutput,
    FlaxBaseModelOutputWithPooling,
)
from transformers.modeling_flax_utils import ACT2FN


class FlaxDinov2WithRegistersPatchEmbeddings(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        image_size = self.config.image_size
        patch_size = self.config.patch_size
        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.num_patches = num_patches
        self.num_channels = self.config.num_channels
        self.projection = nn.Conv(
            self.config.hidden_size,
            kernel_size=patch_size,
            strides=patch_size,
            padding="VALID",
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )

    # Copied from transformers.models.vit.modeling_flax_vit.FlaxViTPatchEmbeddings.__call__
    def __call__(self, pixel_values):
        num_channels = pixel_values.shape[-1]
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        embeddings = self.projection(pixel_values)
        batch_size, _, _, channels = embeddings.shape
        return jnp.reshape(embeddings, (batch_size, -1, channels))


class FlaxDinov2WithRegistersEmbeddings(nn.Module):
    """Construct the CLS token, position and patch embeddings."""

    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            (1, 1, self.config.hidden_size),
        )
        # if self.config.use_mask_token:
        self.mask_token = self.param(
            "mask_token",
            jax.nn.initializers.zeros,
            (1, self.config.hidden_size),
        )
        self.register_tokens = self.param(
            "register_tokens",
            jax.nn.initializers.zeros,
            (1, self.config.num_register_tokens, self.config.hidden_size),
        )
        self.patch_embeddings = FlaxDinov2WithRegistersPatchEmbeddings(
            self.config, dtype=self.dtype
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = self.param(
            "position_embeddings",
            jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            (1, num_patches + 1, self.config.hidden_size),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, config, hidden_states, height, width, position_embeddings):
        num_patches = hidden_states.shape[1] - 1
        num_positions = position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return position_embeddings

        class_pos_embed = position_embeddings[:, 0]
        patch_pos_embed = position_embeddings[:, 1:]
        dim = hidden_states.shape[-1]

        height = height // config.patch_size
        width = width // config.patch_size

        patch_pos_embed = patch_pos_embed.reshape(
            (1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        )
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))
        target_dtype = patch_pos_embed.dtype

        patch_pos_embed = jax.image.resize(
            patch_pos_embed.astype(jnp.float32),
            shape=(patch_pos_embed.shape[0], patch_pos_embed.shape[1], height, width),
            method="bicubic",
            antialias=True,
        ).astype(target_dtype)

        patch_pos_embed = patch_pos_embed.astype(target_dtype)
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 2, 3, 1)).reshape(
            (position_embeddings.shape[0], -1, dim)
        )
        patch_pos_embed_expanded = jnp.tile(patch_pos_embed, (hidden_states.shape[0], 1, 1))
        class_pos_embed_expanded = jnp.tile(class_pos_embed, (hidden_states.shape[0], 1, 1))

        return jnp.concatenate((class_pos_embed_expanded, patch_pos_embed_expanded), axis=1)

    def __call__(self, pixel_values, bool_masked_pos=None, deterministic=True):
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embeddings.projection.dtype
        height, width = pixel_values.shape[1], pixel_values.shape[2]

        embeddings = self.patch_embeddings(pixel_values.astype(target_dtype))

        if bool_masked_pos is not None:
            # bool_masked_pos: (B, N) -> (B, N, 1)
            mask = bool_masked_pos.astype(jnp.bool_)[..., None]
            # (1, H) -> (B, N, H)
            mask_tok = jnp.broadcast_to(
                self.mask_token.astype(embeddings.dtype),
                (embeddings.shape[0], embeddings.shape[1], embeddings.shape[2]),
            )
            embeddings = jnp.where(mask, mask_tok, embeddings)

        cls_tokens = jnp.broadcast_to(self.cls_token, (batch_size, 1, self.config.hidden_size))
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)

        embeddings = embeddings + self.interpolate_pos_encoding(
            self.config, embeddings, height, width, self.position_embeddings
        )

        embeddings = jnp.concatenate(
            (
                embeddings[:, :1],
                jnp.repeat(self.register_tokens, embeddings.shape[0], axis=0),
                embeddings[:, 1:],
            ),
            axis=1,
        )

        embeddings = self.dropout(embeddings, deterministic=deterministic)
        return embeddings


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTSelfAttention with ViT->Dinov2
class FlaxDinov2WithRegistersSelfAttention(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                "`config.hidden_size`: {self.config.hidden_size} has to be a multiple of `config.num_attention_heads`:"
                " {self.config.num_attention_heads}"
            )

        self.query = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.key = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )
        self.value = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, mode="fan_in", distribution="truncated_normal"
            ),
            use_bias=self.config.qkv_bias,
        )

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        query_states = self.query(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        value_states = self.value(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )
        key_states = self.key(hidden_states).reshape(
            hidden_states.shape[:2] + (self.config.num_attention_heads, head_dim)
        )

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = dot_product_attention_weights(
            query_states,
            key_states,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTSelfOutput with ViT->Dinov2
class FlaxDinov2WithRegistersSelfOutput(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTAttention with ViT->Dinov2
class FlaxDinov2WithRegistersAttention(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.attention = FlaxDinov2WithRegistersSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxDinov2WithRegistersSelfOutput(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True, output_attentions: bool = False):
        attn_outputs = self.attention(
            hidden_states, deterministic=deterministic, output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output, hidden_states, deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


def ones_with_scale(key, shape, scale, dtype=jnp.float32):
    return jnp.ones(shape, dtype) * scale


class FlaxDinov2WithRegistersLayerScale(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.lambda1 = self.config.layerscale_value * self.param(
            "lambda1",
            jax.nn.initializers.ones,
            (self.config.hidden_size,),
        )
        self.lambda1 = self.lambda1 * self.config.layerscale_value

    def __call__(self, hidden_states):
        return self.lambda1 * hidden_states


# Copied from transformers.models.beit.modeling_flax_beit.FlaxBeitDropPath with Beit -> Dinov2
class FlaxDinov2WithRegistersDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    rate: float

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = True):
        if self.rate == 0.0:
            return inputs
        keep_prob = 1.0 - self.rate
        if deterministic:
            return inputs
        else:
            shape = (inputs.shape[0],) + (1,) * (
                inputs.ndim - 1
            )  # work with diff dim tensors, not just 2D ConvNets
            rng = self.make_rng("droppath")
            random_tensor = keep_prob + jax.random.uniform(rng, shape=shape, dtype=inputs.dtype)
            binary_tensor = jnp.floor(random_tensor)
            output = inputs / keep_prob * binary_tensor
            return output


class FlaxDinov2WithRegistersMLP(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.fc1 = nn.Dense(
            self.config.hidden_size * self.config.mlp_ratio,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.fc2 = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        if isinstance(self.config.hidden_act, str):
            self.act = ACT2FN[self.config.hidden_act]
        else:
            self.act = self.config.hidden_act

    def __call__(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class FlaxDinov2WithRegistersSwiGLUFFN(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        hidden_features = int(self.config.hidden_size * self.config.mlp_ratio)
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8

        self.weights_in = nn.Dense(
            2 * hidden_features,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )
        self.weights_out = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.config.initializer_range**2, "fan_in", "truncated_normal"
            ),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        hidden_states = self.weights_in(hidden_states)
        x1, x2 = jnp.split(hidden_states, 2, axis=-1)
        hidden = nn.silu(x1) * x2
        return self.weights_out(hidden)


class FlaxDinov2WithRegistersLayer(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.norm1 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)
        self.attention = FlaxDinov2WithRegistersAttention(self.config, dtype=self.dtype)
        self.layer_scale1 = FlaxDinov2WithRegistersLayerScale(self.config, dtype=self.dtype)
        self.drop_path = FlaxDinov2WithRegistersDropPath(self.config.drop_path_rate)
        self.norm2 = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=self.dtype)

        if self.config.use_swiglu_ffn:
            self.mlp = FlaxDinov2WithRegistersSwiGLUFFN(self.config, dtype=self.dtype)
        else:
            self.mlp = FlaxDinov2WithRegistersMLP(self.config, dtype=self.dtype)

        self.layer_scale2 = FlaxDinov2WithRegistersLayerScale(self.config, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic: bool = True, output_attentions: bool = False):
        self_attention_outputs = self.attention(
            self.norm1(hidden_states),  # in Dinov2, layernorm is applied before self-attention
            deterministic=deterministic,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]

        attention_output = self.layer_scale1(attention_output)

        outputs = self_attention_outputs[1:]

        # first residual connection
        hidden_states = self.drop_path(attention_output) + hidden_states

        # in Dinov2, layernorm is also applied after self-attention
        layer_output = self.norm2(hidden_states)
        layer_output = self.mlp(layer_output)
        layer_output = self.layer_scale2(layer_output)

        # second residual connection
        layer_output = self.drop_path(layer_output) + hidden_states

        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTLayerCollection with ViT->Dinov2
class FlaxDinov2WithRegistersLayerCollection(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxDinov2WithRegistersLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states, deterministic=deterministic, output_attentions=output_attentions
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# Copied from transformers.models.vit.modeling_flax_vit.FlaxViTEncoder with ViT->Dinov2
class FlaxDinov2WithRegistersEncoder(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxDinov2WithRegistersLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxDinov2WithRegistersModule(nn.Module):
    config: Dinov2WithRegistersConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embeddings = FlaxDinov2WithRegistersEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxDinov2WithRegistersEncoder(self.config, dtype=self.dtype)
        # LayerNorm is disabled for RAE
        self.layernorm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps, dtype=self.dtype, use_bias=False, use_scale=False
        )

    def __call__(
        self,
        pixel_values,
        bool_masked_pos: jnp.ndarray | None = None,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, deterministic=deterministic
        )

        encoder_outputs = self.encoder(
            hidden_states,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output[:, 0, :]

        if not return_dict:
            head_outputs = (sequence_output, pooled_output)
            return head_outputs + encoder_outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def convert_weights(flax_params, th_encoder):
    flat_params = dict(flatten_dict(flax_params))
    th_params = th_encoder.state_dict()
    marker = {k: 0 for k in th_params.keys()}
    for name, param in flat_params.items():
        th_name = ".".join(name)

        # kernel -> weight
        th_name = th_name.replace("kernel", "weight")

        # norm scale
        th_name = th_name.replace("norm1.scale", "norm1.weight")
        th_name = th_name.replace("norm2.scale", "norm2.weight")

        if th_name not in th_params:
            print(f"Missing key: {th_name}\n")
        else:
            if "weight" in th_name:
                if "norm" in th_name:
                    flat_params[name] = th_params[th_name].cpu().numpy()
                else:
                    if param.ndim == 2:
                        assert param.shape == th_params[th_name].T.shape
                        flat_params[name] = th_params[th_name].cpu().numpy().T
                    elif param.ndim == 4:
                        assert (
                            param.shape
                            == th_params[th_name].cpu().numpy().transpose(2, 3, 1, 0).shape
                        )
                        flat_params[name] = th_params[th_name].cpu().numpy().transpose(2, 3, 1, 0)
                    elif param.ndim == 1:
                        # 1D tensors (biases inside weight-named modules like weights_in)
                        flat_params[name] = th_params[th_name].cpu().numpy()
                    else:
                        raise ValueError(f"Unsupported shape for {name}: {param.shape}")
            else:
                flat_params[name] = th_params[th_name].cpu().numpy()

        marker[th_name] = 1

    for k, v in marker.items():
        if v == 0:
            print(f"Missing key to port in: {k}\n")

    return unflatten_dict(flat_params)


__all__ = [
    "Dinov2WithRegistersConfig",
    "FlaxDinov2WithRegistersModule",
    "convert_weights",
    "DinoWithRegisters",
]


class DinoWithRegisters(nnx.Module):
    def __init__(
        self,
        pretrained_path: str = "facebook/dinov2-with-registers-base",
        resolution: int = 224,
        dtype: jnp.dtype = jnp.float32,
    ):
        self.resolution = resolution
        encoder = Dinov2WithRegistersModel.from_pretrained(pretrained_path)
        encoder.layernorm.elementwise_affine = False
        encoder.layernorm.bias = None
        encoder.layernorm.weight = None

        self.config = Dinov2WithRegistersConfig(**encoder.config.to_dict())
        network = FlaxDinov2WithRegistersModule(config=self.config, dtype=dtype)

        network_params = network.init(
            jax.random.PRNGKey(0), jnp.zeros((1, resolution, resolution, 3), dtype=dtype)
        )
        network_params["params"] = convert_weights(network_params["params"], encoder)
        self.network = network.bind(network_params)

        del encoder

    def encode(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        layers: Sequence[int] | None = None,
    ) -> jnp.ndarray | tuple[jnp.ndarray, list[jnp.ndarray]]:
        """Encode images to DINO features.

        Args:
            x: Input images of shape (B, H, W, C).
            deterministic: Whether to use deterministic mode.
            layers: If specified, also return intermediate layer activations.
                Layer indices are 0-based (e.g., layer 0 is after the first transformer block).

        Returns:
            If layers is None: final features of shape (B, T, D).
            If layers is specified: (final_features, [layer_activations]) where each
                layer activation has shape (B, T, D).
        """
        output_hidden_states = layers is not None
        outputs = self.network(
            x, output_hidden_states=output_hidden_states, deterministic=deterministic
        )

        if layers is None:
            return outputs.last_hidden_state

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Hidden states were not returned by the DINOv2 encoder.")

        num_layers = self.config.num_hidden_layers
        activations = []
        for layer_idx in layers:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(
                    f"Layer index {layer_idx} out of bounds for model with {num_layers} layers."
                )
            # hidden_states[0] is the embedding output, hidden_states[i+1] is after layer i
            activations.append(hidden_states[layer_idx + 1])

        return outputs.last_hidden_state, activations

    def __call__(
        self,
        x: jnp.ndarray,
        deterministic: bool = True,
        layers: Sequence[int] | None = None,
    ) -> jnp.ndarray | tuple[jnp.ndarray, list[jnp.ndarray]]:
        return self.encode(x, deterministic=deterministic, layers=layers)


if __name__ == "__main__":
    import torch
    from einops import rearrange

    huggingface_names = [
        "facebook/dinov2-with-registers-small",
        "facebook/dinov2-with-registers-base",
        "facebook/dinov2-with-registers-large",
        "facebook/dinov2-with-registers-giant",
    ]
    facebook_names = [
        "dinov2_vits14_reg",
        "dinov2_vitb14_reg",
        "dinov2_vitl14_reg",
        "dinov2_vitg14_reg",
    ]

    @torch.no_grad()
    def check_dino_model(hf_name: str, fb_name: str, res: int = 224):
        image = np.random.default_rng(0).normal(size=(1, res, res, 3))
        torch_image = rearrange(torch.tensor(image, dtype=torch.float32), "b h w c -> b c h w")
        jax_image = jnp.array(image, dtype=jnp.float32)

        hf_dino = DinoWithRegisters(hf_name, resolution=res)
        fb_dino = torch.hub.load("facebookresearch/dinov2", fb_name)

        torch_output = fb_dino.forward_features(torch_image)["x_prenorm"]
        torch_output = (torch_output - torch.mean(torch_output, dim=-1, keepdim=True)) / (
            torch.sqrt(torch.var(torch_output, dim=-1, keepdim=True) + 1e-5)
        )
        jax_output = hf_dino(jax_image)

        torch_output, jax_output = torch_output.cpu().numpy(), np.array(jax_output)
        diff = np.abs(torch_output - jax_output)
        print("jax_output.shape: ", jax_output.shape)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        print(f"model: {fb_name}\tmean err: {mean_diff}\tmax err: {max_diff}")

    for hf_name, fb_name in zip(huggingface_names, facebook_names):
        check_dino_model(hf_name, fb_name)
