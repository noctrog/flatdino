# TAKEN FROM: https://github.dev/willisma/diffuse_nnx
"""Flax/nnx implementation of the ViT-MAE decoder."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("ORBAX_USE_FAKE_ASYNC", "1")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from flax import struct
import torch
from flatdino.pretrained.rae_decoder_utils import (
    ACT2FN,
    ModelOutput,
    ViTMAEConfig,
    get_2d_sincos_pos_embed,
    convert_weights,
)
from flatdino.pretrained.rae_decoder_vitxl import download_rae_decoder

Array = jnp.ndarray


class Buffer(nnx.Variable):
    """Non-trainable container for fixed arrays."""


@struct.dataclass
class ViTMAEDecoderOutput(ModelOutput):
    """Outputs produced by the ViT-MAE decoder."""

    logits: Array
    hidden_states: Optional[Tuple[Array, ...]] = None
    attentions: Optional[Tuple[Array, ...]] = None


class ViTMAESelfAttention(nnx.Module):
    """Multi-head self-attention for the decoder blocks."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"The hidden size {hidden_size} is not a multiple of the number of heads {num_heads}."
            )

        self.num_attention_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nnx.Linear(
            hidden_size,
            self.all_head_size,
            use_bias=config.qkv_bias,
            dtype=dtype,
            rngs=rngs,
        )
        self.key = nnx.Linear(
            hidden_size,
            self.all_head_size,
            use_bias=config.qkv_bias,
            dtype=dtype,
            rngs=rngs,
        )
        self.value = nnx.Linear(
            hidden_size,
            self.all_head_size,
            use_bias=config.qkv_bias,
            dtype=dtype,
            rngs=rngs,
        )
        # self.dropout = nnx.Dropout(config.attention_probs_dropout_prob, rngs=rngs)
        self.scale = 1.0 / math.sqrt(self.attention_head_size)

    def _reshape_for_scores(self, x: Array) -> Array:
        new_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = jnp.reshape(x, new_shape)
        return jnp.transpose(x, (0, 2, 1, 3))

    def __call__(
        self,
        hidden_states: Array,
        head_mask: Optional[Array] = None,
        *,
        output_attentions: bool = False,
    ) -> Tuple[Array, ...]:
        query_layer = self._reshape_for_scores(self.query(hidden_states))
        key_layer = self._reshape_for_scores(self.key(hidden_states))
        value_layer = self._reshape_for_scores(self.value(hidden_states))

        attention_scores = jnp.matmul(query_layer, jnp.swapaxes(key_layer, -1, -2)) * self.scale
        attention_probs = jax.nn.softmax(attention_scores, axis=-1)
        # attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = jnp.matmul(attention_probs, value_layer)
        context_layer = jnp.transpose(context_layer, (0, 2, 1, 3))
        new_context_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = jnp.reshape(context_layer, new_context_shape)

        if output_attentions:
            return context_layer, attention_probs
        return (context_layer,)


class ViTMAESelfOutput(nnx.Module):
    """Output projection for the attention block (residual handled in the layer)."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        hidden_size = config.hidden_size
        self.dense = nnx.Linear(hidden_size, hidden_size, dtype=dtype, rngs=rngs)
        # self.dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTMAEAttention(nnx.Module):
    """Attention block with pre/post projections."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.attention = ViTMAESelfAttention(config, rngs=rngs, dtype=dtype)
        self.output = ViTMAESelfOutput(config, rngs=rngs, dtype=dtype)

    def __call__(
        self,
        hidden_states: Array,
        head_mask: Optional[Array] = None,
        *,
        output_attentions: bool = False,
    ) -> Tuple[Array, ...]:
        self_outputs = self.attention(
            hidden_states, head_mask=head_mask, output_attentions=output_attentions
        )
        attention_output = self.output(self_outputs[0])
        outputs: Tuple[Array, ...] = (attention_output,) + self_outputs[1:]
        return outputs


class ViTMAEIntermediate(nnx.Module):
    """Feed-forward network expansion."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.dense = nnx.Linear(hidden_size, intermediate_size, dtype=dtype, rngs=rngs)
        hidden_act = config.hidden_act
        if isinstance(hidden_act, str):
            if hidden_act not in ACT2FN:
                raise ValueError(f"Unsupported activation string: {hidden_act}")
            self.activation = ACT2FN[hidden_act]
        elif callable(hidden_act):
            self.activation = hidden_act
        else:
            raise ValueError("hidden_act must be either a string or a callable")

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = self.dense(hidden_states)
        return self.activation(hidden_states)


class ViTMAEOutput(nnx.Module):
    """Feed-forward network projection and residual merge."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.dense = nnx.Linear(intermediate_size, hidden_size, dtype=dtype, rngs=rngs)
        # self.dropout = nnx.Dropout(config.hidden_dropout_prob, rngs=rngs)

    def __call__(self, hidden_states: Array, input_tensor: Array) -> Array:
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class ViTMAELayer(nnx.Module):
    """Single transformer block used in the decoder."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
    ) -> None:
        self.attention = ViTMAEAttention(config, rngs=rngs, dtype=dtype)
        self.intermediate = ViTMAEIntermediate(config, rngs=rngs, dtype=dtype)
        self.output = ViTMAEOutput(config, rngs=rngs, dtype=dtype)
        self.layernorm_before = nnx.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs
        )
        self.layernorm_after = nnx.LayerNorm(
            config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs
        )

    def __call__(
        self,
        hidden_states: Array,
        head_mask: Optional[Array] = None,
        *,
        output_attentions: bool = False,
    ) -> Tuple[Array, ...]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs: Tuple[Array, ...] = self_attention_outputs[1:]

        hidden_states = attention_output + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = self.output(layer_output, hidden_states)

        return (layer_output,) + outputs


class GeneralDecoder(nnx.Module):
    """ViT-MAE decoder implemented with Flax/nnx."""

    def __init__(
        self,
        config: ViTMAEConfig,
        *,
        num_patches: int,
        rngs: nnx.Rngs | None = None,
        dtype: jnp.dtype = jnp.float32,
        pretrained_path: Path | str | None = None,
    ) -> None:
        if rngs is None:
            rngs = nnx.Rngs(0)
        decoder_config = copy.deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size

        self.config = config
        self.decoder_config = decoder_config
        self.num_patches = num_patches
        self.dtype = dtype

        self.decoder_embed = nnx.Linear(
            config.hidden_size, decoder_config.hidden_size, dtype=dtype, rngs=rngs
        )
        pos_embed = get_2d_sincos_pos_embed(
            decoder_config.hidden_size, int(math.sqrt(num_patches)), add_cls_token=True
        )
        self.decoder_pos_embed = Buffer(jnp.asarray(pos_embed, dtype=dtype)[None, ...])

        self.decoder_layers = nnx.List(
            [
                ViTMAELayer(decoder_config, rngs=rngs, dtype=dtype)
                for _ in range(decoder_config.num_hidden_layers)
            ]
        )
        self.decoder_norm = nnx.LayerNorm(
            decoder_config.hidden_size, epsilon=config.layer_norm_eps, dtype=dtype, rngs=rngs
        )
        self.decoder_pred = nnx.Linear(
            decoder_config.hidden_size,
            config.patch_size * config.patch_size * config.num_channels,
            dtype=dtype,
            rngs=rngs,
        )

        self.set_trainable_cls_token()

        if pretrained_path is None:
            pretrained_path = download_rae_decoder()

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)
            self._resize_decoder_pos_embed()

    def set_trainable_cls_token(self, tensor: Optional[Array] = None) -> None:
        if tensor is None:
            tensor = jnp.zeros((1, 1, self.decoder_config.hidden_size), dtype=self.dtype)
        self.trainable_cls_token = nnx.Param(tensor)

    def interpolate_pos_encoding(self, embeddings: Array) -> Array:
        embeddings_positions = embeddings.shape[1] - 1
        num_positions = self.decoder_pos_embed.shape[1] - 1

        if embeddings_positions == num_positions:
            return self.decoder_pos_embed

        class_pos_embed = self.decoder_pos_embed[:, :1, :]
        patch_pos_embed = self.decoder_pos_embed[:, 1:, :]
        patch_pos_embed = jnp.reshape(patch_pos_embed, (1, 1, num_positions, -1))
        patch_pos_embed = jnp.transpose(patch_pos_embed, (0, 3, 1, 2))
        resized = jax.image.resize(
            patch_pos_embed,
            shape=(1, patch_pos_embed.shape[1], 1, embeddings_positions),
            method="linear",
            antialias=False,
        )
        resized = jnp.transpose(resized, (0, 2, 3, 1)).reshape(1, embeddings_positions, -1)
        return jnp.concatenate([class_pos_embed, resized], axis=1)

    def _resize_decoder_pos_embed(self) -> None:
        """Resize decoder positional embeddings to match the configured patch grid."""

        target_tokens = self.num_patches + 1
        decoder_pos = self.decoder_pos_embed.value
        current_tokens = decoder_pos.shape[1]

        if current_tokens == target_tokens:
            return

        if current_tokens <= 1:
            raise ValueError("Decoder positional embedding tensor is malformed.")

        cls_pos = decoder_pos[:, :1, :]
        patch_pos = decoder_pos[:, 1:, :]

        current_num_patches = patch_pos.shape[1]
        current_hw = int(math.sqrt(current_num_patches))
        target_hw = int(math.sqrt(self.num_patches))

        if current_hw * current_hw != current_num_patches:
            raise ValueError("Current decoder positional embedding does not form a square grid.")
        if target_hw * target_hw != self.num_patches:
            raise ValueError("Target number of patches does not form a square grid.")

        patch_pos = jnp.reshape(patch_pos, (1, current_hw, current_hw, -1))
        patch_pos = jnp.transpose(patch_pos, (0, 3, 1, 2))
        resized = jax.image.resize(
            patch_pos,
            shape=(1, patch_pos.shape[1], target_hw, target_hw),
            method="linear",
            antialias=False,
        )
        resized = jnp.transpose(resized, (0, 2, 3, 1)).reshape(1, self.num_patches, -1)
        resized = resized.astype(decoder_pos.dtype)

        self.decoder_pos_embed.value = jnp.concatenate([cls_pos, resized], axis=1)

    def interpolate_latent(self, x: Array) -> Array:
        batch_size, length, channels = x.shape
        if length == self.num_patches:
            return x
        height = width = int(math.sqrt(length))
        target_hw = int(math.sqrt(self.num_patches))
        x_img = jnp.reshape(x, (batch_size, height, width, channels))
        x_img = jax.image.resize(
            x_img,
            shape=(batch_size, target_hw, target_hw, channels),
            method="linear",
            antialias=False,
        )
        return jnp.reshape(x_img, (batch_size, self.num_patches, channels))

    def unpatchify(
        self,
        patchified_pixel_values: Array,
        original_image_size: Optional[Tuple[int, int]] = None,
    ) -> Array:
        """Reconstruct an image from decoder patch predictions."""

        if patchified_pixel_values.ndim != 3:
            raise ValueError("patchified_pixel_values must be of shape (batch, num_patches, dim).")

        batch_size, num_patches, patch_dim = patchified_pixel_values.shape
        num_channels = self.config.num_channels

        patch_area, remainder = divmod(patch_dim, num_channels)
        if remainder != 0:
            raise ValueError(
                "Patch embedding dimension is not divisible by the number of channels. "
                "Cannot infer patch resolution."
            )

        patch_size = int(math.sqrt(patch_area))
        if patch_size * patch_size != patch_area:
            raise ValueError("Inferred patch size is not an integer square.")

        grid_size = int(math.sqrt(num_patches))
        if grid_size * grid_size != num_patches:
            raise ValueError("Number of patches must form a square grid.")

        x = jnp.reshape(
            patchified_pixel_values,
            (
                batch_size,
                grid_size,
                grid_size,
                patch_size,
                patch_size,
                num_channels,
            ),
        )
        x = jnp.transpose(x, (0, 5, 1, 3, 2, 4))
        x = jnp.reshape(
            x, (batch_size, num_channels, grid_size * patch_size, grid_size * patch_size)
        )

        if original_image_size is None:
            original_image_size = (self.config.image_size, self.config.image_size)

        target_height, target_width = original_image_size
        current_height = grid_size * patch_size
        current_width = grid_size * patch_size

        if (target_height, target_width) != (current_height, current_width):
            if target_height > current_height or target_width > current_width:
                # Use bilinear resize when upscaling is required.
                x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
                x_nhwc = jax.image.resize(
                    x_nhwc,
                    (batch_size, target_height, target_width, num_channels),
                    method="linear",
                    antialias=False,
                )
                x = jnp.transpose(x_nhwc, (0, 3, 1, 2))
            else:
                offset_h = (current_height - target_height) // 2
                offset_w = (current_width - target_width) // 2
                x = x[
                    :,
                    :,
                    offset_h : offset_h + target_height,
                    offset_w : offset_w + target_width,
                ]

        return x

    def __call__(
        self,
        hidden_states: Array,
        *,
        head_mask: Optional[Array] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        interpolate_pos_encoding: bool = False,
        drop_cls_token: bool = False,
    ) -> ViTMAEDecoderOutput | Tuple[Array, ...]:
        x = self.decoder_embed(hidden_states)

        if drop_cls_token:
            x_ = x[:, 1:, :]
            x_ = self.interpolate_latent(x_)
        else:
            x_ = self.interpolate_latent(x)

        cls_token = jnp.broadcast_to(
            self.trainable_cls_token.value, (x_.shape[0],) + self.trainable_cls_token.shape[1:]
        )
        x = jnp.concatenate([cls_token, x_], axis=1)

        if interpolate_pos_encoding:
            if not drop_cls_token:
                raise ValueError("interpolate_pos_encoding requires drop_cls_token=True")
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed

        hidden_states = x + decoder_pos_embed

        all_hidden_states: Optional[Tuple[Array, ...]] = () if output_hidden_states else None
        all_self_attentions: Optional[Tuple[Array, ...]] = () if output_attentions else None

        for layer_module in self.decoder_layers:
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, head_mask=head_mask, output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions and all_self_attentions is not None and len(layer_outputs) > 1:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)
        logits = self.decoder_pred(hidden_states)
        logits = logits[:, 1:, :]

        if not return_dict:
            outputs: Tuple[Array, ...] = (logits,)
            if output_hidden_states and all_hidden_states is not None:
                outputs = outputs + (all_hidden_states,)
            if output_attentions and all_self_attentions is not None:
                outputs = outputs + (all_self_attentions,)
            return outputs

        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

    def load_pretrained(self, path: str | Path) -> None:
        """Load decoder parameters from a serialized nnx state file."""

        checkpoint_path = Path(path)
        torch_state = torch.load(checkpoint_path)
        decoder = convert_weights(torch_state, self)
        nnx.update(self, decoder)


def make_rae_decoder(
    num_patches: int, *, image_size: int, dtype: jnp.dtype, seed: int = 0
) -> GeneralDecoder:
    config_path = Path(__file__).resolve().parents[2] / "pretrained" / "rae_decoder" / "config.yaml"

    if config_path.exists():
        decoder_cfg = ViTMAEConfig.from_pretrained(str(config_path))
    else:
        raise FileNotFoundError

    decoder_cfg.image_size = image_size
    decoder_cfg.patch_size = 16

    decoder = GeneralDecoder(decoder_cfg, num_patches=num_patches, rngs=nnx.Rngs(seed), dtype=dtype)
    return decoder


def _init_test() -> None:
    """Runs a decoding smoke test using Imagenet samples and the decoder checkpoint."""

    import matplotlib.pyplot as plt

    from flatdino.data import (
        DataConfig,
        IMAGENET_DEFAULT_MEAN,
        IMAGENET_DEFAULT_STD,
        create_dataloaders,
    )
    from flatdino.augmentations import FlatDinoAugConfig, FlatDinoValAugmentations
    from flatdino.pretrained import DinoWithRegisters

    config_path = Path(__file__).resolve().with_name("config.yaml")
    output_dir = Path(os.environ.get("VIT_DECODER_OUTPUT", "/tmp"))

    if config_path.exists():
        config = ViTMAEConfig.from_pretrained(str(config_path))
    else:
        config = ViTMAEConfig()

    config.image_size = 256
    config.patch_size = 16

    num_patches = (config.image_size // config.patch_size) ** 2
    rngs = nnx.Rngs(0)
    checkpoint_path = download_rae_decoder()
    print(f"checkpoint_path: {checkpoint_path}")
    decoder = GeneralDecoder(
        config,
        num_patches=num_patches,
        rngs=rngs,
        pretrained_path=checkpoint_path,
    )

    data_cfg = DataConfig(num_workers=0)
    aug_cfg = FlatDinoAugConfig(image_size=(config.image_size, config.image_size))
    data = create_dataloaders(
        data_cfg,
        batch_size=4,
        val_aug=FlatDinoValAugmentations(aug_cfg, data_cfg),
        val_epochs=1,
    )
    images = next(iter(data.val_loader))["image"].astype(np.float32)
    b, _, _, d = images.shape

    dino = DinoWithRegisters(resolution=224)
    images_dino = jax.image.resize(
        jnp.asarray(images),
        (b, dino.resolution, dino.resolution, d),
        method="linear",
        antialias=True,
    )
    tokens = dino(images_dino)
    hidden_states = tokens[:, 5:, :]

    output = decoder(
        hidden_states,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )

    recon = decoder.unpatchify(output.logits)
    recon_np = np.asarray(recon)
    images_np = np.asarray(images)

    mean = np.array(IMAGENET_DEFAULT_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_DEFAULT_STD, dtype=np.float32)

    def denormalize(x: np.ndarray) -> np.ndarray:
        return x * std.reshape(1, 1, 3) + mean.reshape(1, 1, 3)

    orig = denormalize(images_np)
    orig = np.clip(orig, 0.0, 1.0)
    orig = (orig * 255.0).astype(np.uint8)

    recon_hwc = np.transpose(recon_np, (0, 2, 3, 1))
    recon_hwc = denormalize(recon_hwc)
    recon_hwc = np.clip(recon_hwc, 0.0, 1.0)
    recon_hwc = (recon_hwc * 255.0).astype(np.uint8)

    print("decoded shape: ", recon_hwc.shape)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "decoder_reconstructions.png"

    num_images = orig.shape[0]
    fig, axes = plt.subplots(2, num_images, figsize=(4 * num_images, 8))
    if num_images == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for idx in range(num_images):
        axes[0, idx].imshow(orig[idx])
        axes[0, idx].axis("off")
        axes[0, idx].set_title(f"Original {idx}")
        axes[1, idx].imshow(recon_hwc[idx])
        axes[1, idx].axis("off")
        axes[1, idx].set_title(f"Reconstruction {idx}")

    fig.tight_layout()
    fig.savefig(comparison_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {comparison_path}")


if __name__ == "__main__":
    _init_test()
