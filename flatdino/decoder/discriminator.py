from typing import Callable
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import chex
import jmp
from einops import rearrange

from flatdino.pretrained.dino import DinoViT

# ImageNet normalization constants for DINO discriminator
# Converts input from [-1, 1] to ImageNet normalized space
# Formula: x_norm = x * x_scale + x_shift where x in [-1, 1]
# Derived from: ((x+1)/2 - mean) / std = 0.5*x/std + (0.5-mean)/std
IMAGENET_MEAN = jnp.array([0.485, 0.456, 0.406])
IMAGENET_STD = jnp.array([0.229, 0.224, 0.225])
DINO_X_SCALE = (0.5 / IMAGENET_STD).reshape(1, 1, 1, 3)  # For BHWC format
DINO_X_SHIFT = ((0.5 - IMAGENET_MEAN) / IMAGENET_STD).reshape(1, 1, 1, 3)


class BatchNormLocal(nnx.Module):
    def __init__(
        self,
        num_features: int,
        mp: jmp.Policy,  # Assuming jmp is imported
        affine: bool = True,
        virtual_bs: int = 1,
        eps: float = 1e-6,
        *,
        rngs: nnx.Rngs,
    ):
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine
        self.mp = mp

        if self.affine:
            self.weight = nnx.Param(jnp.ones((num_features,), dtype=mp.param_dtype))
            self.bias = nnx.Param(jnp.zeros((num_features,), dtype=mp.param_dtype))
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: jax.Array) -> jax.Array:
        chex.assert_rank(x, 3)  # Input: (Batch, Time, Features)
        x = jnp.astype(x, jnp.float32)
        x = rearrange(x, "(g v) t d -> g v t d", v=self.virtual_bs)

        mean = jnp.mean(x, axis=(1, 2), keepdims=True)
        var = jnp.var(x, axis=(1, 2), keepdims=True)
        scale = jax.lax.rsqrt(var + self.eps)
        x = (x - mean) * scale

        if self.affine:
            w = self.weight.value[None, None, None, :]
            b = self.bias.value[None, None, None, :]
            x = x * w + b

        x = rearrange(x, "g v t d -> (g v) t d")
        x = self.mp.cast_to_compute(x)
        return x


class ResidualBlock(nnx.Module):
    def __init__(self, fn: Callable):
        self.fn = fn
        self.ratio = 2**-0.5

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        return (self.fn(x, **kwargs) + x) * self.ratio


class DinoDiscBlock(nnx.Module):
    def __init__(
        self, mp: jmp.Policy, channels: int, kernel_size: int, norm_type: str, *, rngs: nnx.Rngs
    ):
        # This value is autamatically set with nnx.Module.{train(), eval()}. But nnx.SpectralNorm
        # takes an `update_stats` argument for training/eval. We store this dummy variable here
        # to track the training/eval state instead. By default, it's on training state
        self.use_running_average = False

        match norm_type:
            case "bn":
                self.norm = BatchNormLocal(channels, mp, rngs=rngs)
            case "gn":
                self.norm = nnx.GroupNorm(channels, num_groups=32, rngs=rngs)
            case _:
                raise ValueError("Invalid norm type")

        self.conv = nnx.SpectralNorm(
            nnx.Conv(channels, channels, kernel_size=(kernel_size,), padding="CIRCULAR", rngs=rngs),
            rngs=rngs,
        )
        self.relu = partial(nnx.leaky_relu, negative_slope=0.2)

    def __call__(self, x: jax.Array):
        is_training = not self.use_running_average
        x = self.conv(x, update_stats=is_training)
        x = self.norm(x)
        return self.relu(x)


# def make_block(
#     mp: jmp.Policy,
#     channels: int,
#     kernel_size: int,
#     norm_type: str,
#     *,
#     rngs: nnx.Rngs,
# ) -> nnx.Module:
#     match norm_type:
#         case "bn":
#             norm = BatchNormLocal(channels, mp, rngs=rngs)
#         case "gn":
#             norm = nnx.GroupNorm(channels, num_groups=32, rngs=rngs)
#         case _:
#             raise ValueError("Invalid norm type")

#     # TODO: this applies the spectral norm also to the bias, and the ref. impl. does not
#     conv = nnx.SpectralNorm(
#         nnx.Conv(channels, channels, kernel_size=(kernel_size,), padding="CIRCULAR", rngs=rngs),
#         rngs=rngs,
#     )

#     return nnx.Sequential(conv, norm, partial(nnx.leaky_relu, negative_slope=0.2))


class DinoDisc(nnx.Module):
    def __init__(
        self,
        ks: int,
        mp: jmp.Policy,
        dino_name: str = "facebook/dino-vits8",
        key_depths=(2, 5, 8, 11),
        *,
        rngs: nnx.Rngs,
    ):
        # This value is autamatically set with nnx.Module.{train(), eval()}. But nnx.SpectralNorm
        # takes an `update_stats` argument for training/eval. We store this dummy variable here
        # to track the training/eval state instead. By default, it's on training state
        self.use_running_average = False

        self.mp = mp
        self.key_depths = key_depths
        self.dino = DinoViT(dino_name, resolution=224)

        # Infer embed_dim from DINO config
        embed_dim = self.dino.config.hidden_size
        self.heads = nnx.List(
            [
                nnx.Sequential(
                    DinoDiscBlock(self.mp, embed_dim, 1, "bn", rngs=rngs),
                    ResidualBlock(DinoDiscBlock(self.mp, embed_dim, ks, "bn", rngs=rngs)),
                )
                for _ in range(len(key_depths) + 1)
            ]
        )
        self.norms = nnx.List(
            [
                nnx.SpectralNorm(
                    nnx.Conv(embed_dim, 1, kernel_size=1, padding=0, rngs=rngs), rngs=rngs
                )
                for _ in range(len(key_depths) + 1)
            ]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through DINO discriminator.

        Args:
            x: Input images in [-1, 1] range with shape (B, H, W, C).

        Returns:
            Discriminator logits with shape (B, num_outputs).
        """
        chex.assert_rank(x, 4)
        b, h, w, c = x.shape
        is_training = not self.use_running_average

        # Resize to DINO input resolution
        x = jax.image.resize(x, (b, 224, 224, c), method="bilinear")

        # Convert from [-1, 1] to ImageNet normalized space
        # This matches the reference implementation's internal normalization
        x = x * DINO_X_SCALE + DINO_X_SHIFT

        # Get intermediate activations at key_depths
        final_output, intermediate_activations = self.dino(
            x, deterministic=True, capture_layers=self.key_depths
        )

        # Remove CLS token from all activations (CLS is at index 0)
        # Reference: x[:, 1:, :].transpose(1, 2) but we keep (B, T, C) for JAX Conv
        activations = [act[:, 1:, :] for act in intermediate_activations]

        # Add final DINO output to activations (reference includes final output at index 0)
        activations.insert(0, final_output[:, 1:, :])

        outputs = []
        for head, norm, act in zip(self.heads, self.norms, activations):
            tmp = head(act)
            tmp = norm(tmp, update_stats=is_training)
            outputs.append(rearrange(tmp, "b ... -> b (...)"))

        return jnp.concatenate(outputs, axis=1)
