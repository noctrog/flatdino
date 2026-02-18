from typing import Literal
from dataclasses import dataclass, field
from functools import reduce
import math
import operator

import jax
import jax.numpy as jnp
import jmp
import flax.nnx as nnx
from flatdino.data import DataConfig
from flatdino.models.fsq import FSQ
from flatdino.models.transformer import TransformerConfig
from flatdino.models.vit import ViTConfig, ViTEncoder
from flatdino.augmentations import FlatDinoAugConfig


@dataclass
class VQOptimConfig:
    epochs: int = 100
    batch_size: int = 1024

    adam_b1: float = 0.9
    adam_b2: float = 0.999
    lr_start: float = 1e-6
    lr_peak: float = 1e-4
    lr_final: float = 1e-8
    weight_decay: float = 0.02
    """Weight decay for AdamW. Applied to 2D params only, excluding embeddings."""
    warmup_epochs: int = 10
    decay_epochs: int | None = None

    recon_weight: float = 1.0

    lr_schedule: Literal["warmup_cosine", "wsd"] = "warmup_cosine"


@dataclass
class VQDinoConfig:
    dino_name: str = "facebook/dinov2-with-registers-base"
    train: VQOptimConfig = field(default_factory=lambda: VQOptimConfig())
    data: DataConfig = field(default_factory=lambda: DataConfig())
    encoder: ViTConfig = field(
        default_factory=lambda: ViTConfig(
            patch=None,
            num_patches=256,
            input_dim=768,
            num_registers=32,
            transformer=TransformerConfig(
                embed_dim=768, num_layers=12, mlp_hidden_dim=3072, num_heads=12
            ),
        )
    )
    decoder: ViTConfig = field(
        default_factory=lambda: ViTConfig(
            patch=None,
            num_patches=32,
            num_registers=256,
            transformer=TransformerConfig(
                embed_dim=768, num_layers=12, mlp_hidden_dim=3072, num_heads=12
            ),
        )
    )
    aug: FlatDinoAugConfig = field(default_factory=lambda: FlatDinoAugConfig())
    levels: list[int] = field(default_factory=lambda: [8, 5, 5, 5])
    """FSQ levels per channel. Codebook size = prod(levels). Each level should be 3-9."""

    encoder_disposable_registers: int = 0
    """Number of disposable registers for encoder. These are prepended to latent tokens
    and discarded after encoding. Sequence order: [disposable, latents, patches]."""
    decoder_disposable_registers: int = 0
    """Number of disposable registers for decoder. These are prepended to output patches
    and discarded after decoding. Sequence order: [disposable, output_patches, latent_input]."""

    nested_dropout: bool = False
    """Enable nested dropout on quantized latent tokens. Replaces dropped tokens with
    a learned mask embedding at decoder hidden dim. Enables causal register attention in encoder."""

    @property
    def num_latents(self) -> int:
        """Number of actual latent tokens (excluding disposable registers)."""
        return self.encoder.num_registers - self.encoder_disposable_registers

    @property
    def num_output_patches(self) -> int:
        """Number of actual output patches (excluding disposable registers)."""
        return self.decoder.num_registers - self.decoder_disposable_registers

    @property
    def codebook_size(self) -> int:
        return reduce(operator.mul, self.levels, 1)


class VQDinoAutoencoder(nnx.Module):
    def __init__(
        self,
        cfg: VQDinoConfig,
        *,
        mp: jmp.Policy,
        rngs: nnx.Rngs,
    ):
        assert cfg.encoder.patch is None
        assert cfg.decoder.patch is None

        self.cfg = cfg
        self.fsq = FSQ(cfg.levels)

        if cfg.nested_dropout:
            decoder_embed_dim = cfg.decoder.transformer.embed_dim
            kinit = nnx.initializers.truncated_normal(0.02)
            self.latent_proj = nnx.Linear(
                len(cfg.levels),
                decoder_embed_dim,
                use_bias=False,
                dtype=mp.compute_dtype,
                param_dtype=mp.param_dtype,
                kernel_init=kinit,
                rngs=rngs,
            )
            self.mask_token = nnx.Param(
                kinit(rngs(), (1, 1, decoder_embed_dim), dtype=mp.param_dtype),
                sharding=(None, None, None),
            )
        else:
            self.latent_proj = None
            self.mask_token = None

        self.encoder = ViTEncoder(cfg.encoder, mp, rngs=rngs)
        self.decoder = ViTEncoder(cfg.decoder, mp, rngs=rngs)

    @property
    def trainable_pytree(self):
        pytree: dict = {"encoder": self.encoder, "decoder": self.decoder}
        if self.mask_token is not None:
            pytree["mask_token"] = self.mask_token
        if self.latent_proj is not None:
            pytree["latent_proj"] = self.latent_proj
        return pytree

    @property
    def trainable_state(self):
        return nnx.state(self.trainable_pytree, nnx.Param)

    def _encoder_causal_mask(self) -> jax.Array:
        """Build attention mask for encoder with causal register attention.

        Sequence layout: [reg_0, ..., reg_{R-1}, patch_0, ..., patch_{N-1}]

        Rules:
        - Register↔Register (top-left R×R): lower-triangular (causal)
        - Register→Patch (top-right R×N): all True
        - Patch→Register (bottom-left N×R): all True
        - Patch↔Patch (bottom-right N×N): all True

        Returns:
            Boolean mask of shape (1, 1, R+N, R+N).
        """
        R = self.cfg.encoder.num_registers
        N = self.cfg.encoder.num_patches
        S = R + N

        # Start with all True
        mask = jnp.ones((S, S), dtype=jnp.bool_)
        # Apply causal (lower-triangular) to register↔register block
        reg_causal = jnp.tril(jnp.ones((R, R), dtype=jnp.bool_))
        mask = mask.at[:R, :R].set(reg_causal)

        return mask[None, None, :, :]

    def _sample_keep_mask(
        self, num_latents: int, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Sample which latent tokens to keep using pow2 sampling.

        K_keep is sampled uniformly from {1, 2, 4, ..., num_latents}.

        Returns:
            keep_mask: (1, K, 1) boolean — True = keep this token
            k_keep: scalar int — number of tokens kept
        """
        max_exp = int(math.log2(num_latents))
        pow2_options = jnp.array([2**i for i in range(max_exp + 1)])
        idx = jax.random.randint(key, (), 0, len(pow2_options))
        k_keep = pow2_options[idx]
        keep_mask = jnp.arange(num_latents)[None, :, None] < k_keep
        return keep_mask, k_keep

    def encode(
        self, x: jax.Array, *, key: jax.Array | None = None
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Encode DINO patches to FSQ-quantized latents.

        Args:
            x: DINO normalized patches of shape (B, 256, 768)
            key: Optional PRNG key for nested dropout sampling.

        Returns:
            z_quantized: Quantized codes in ~[-1, 1], shape (B, num_latents, len(levels))
            aux: Dict containing 'indices', 'z_pre', and optionally 'keep_mask'/'k_keep'.
        """
        mask = self._encoder_causal_mask() if self.cfg.nested_dropout else None
        encoded = self.encoder(x, mask=mask)
        num_latents = self.cfg.num_latents
        skip = self.cfg.encoder_disposable_registers

        z_pre = encoded[:, skip : skip + num_latents]
        z_quantized = self.fsq.quantize(z_pre)

        if key is not None and self.cfg.nested_dropout:
            keep_mask, k_keep = self._sample_keep_mask(num_latents, key)
        else:
            keep_mask, k_keep = None, None

        aux = {
            "indices": self.fsq.codes_to_indices(z_quantized),
            "z_pre": z_pre,
            "keep_mask": keep_mask,
            "k_keep": k_keep,
        }
        return z_quantized, aux

    def decode(
        self, z: jax.Array, *, keep_mask: jax.Array | None = None
    ) -> jax.Array:
        """Decode quantized latents to DINO patch reconstructions.

        Args:
            z: Quantized codes of shape (B, num_latents, len(levels))
            keep_mask: Optional (1, K, 1) boolean mask. Dropped tokens are replaced
                with the learned mask_token at decoder embed_dim.

        Returns:
            Reconstructed DINO patches of shape (B, num_output_patches, 768)
        """
        if self.latent_proj is not None:
            z = self.latent_proj(z)

        if keep_mask is not None and self.mask_token is not None:
            z = jnp.where(keep_mask, z, self.mask_token.value)

        decoded = self.decoder(z)
        num_output_patches = self.cfg.num_output_patches
        skip = self.cfg.decoder_disposable_registers
        return decoded[:, skip : skip + num_output_patches]
