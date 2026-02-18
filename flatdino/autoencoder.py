from typing import Literal
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jmp
import flax.nnx as nnx
import orbax.checkpoint as ocp

from flatdino.data import DataConfig
from flatdino.models.transformer import TransformerConfig
from flatdino.models.vit import ViTConfig, ViTEncoder
from flatdino.augmentations import FlatDinoAugConfig


@dataclass
class OptimConfig:
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
    kl_weight: float | None = None
    """KL divergence weight. If None, computed as 1e-6 * (512 / latent_dim) to normalize per-dimension."""
    kl_anneal_steps: int | None = None
    """Number of steps to linearly anneal kl_weight from 0 to its full value. None disables annealing."""
    free_bits: float | None = None
    """Minimum KL per dimension (in nats) before penalizing. Prevents posterior collapse. None disables."""
    kl_var_weight: float | None = None
    """Weight for KL variance loss (variance across tokens, mean over batch). Encourages balanced
    KL distribution across tokens. None disables. Specified via -klvarN in experiment name (1e-N)."""

    lr_schedule: Literal["warmup_cosine", "wsd"] = "warmup_cosine"


@dataclass
class FlatDinoConfig:
    dino_name: str = "facebook/dinov2-with-registers-base"
    train: OptimConfig = field(default_factory=lambda: OptimConfig())
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
    decoder: ViTConfig = field(default_factory=lambda: ViTConfig(patch=None, num_patches=196))
    aug: FlatDinoAugConfig = field(default_factory=lambda: FlatDinoAugConfig())

    # Disposable registers: extra learnable tokens prepended to the sequence that are
    # discarded after the forward pass. These can help attention patterns without
    # contributing to the latent representation. The ViTConfig.num_registers field
    # should be set to (num_latents + disposable_registers) for encoder, and
    # (num_output_patches + disposable_registers) for decoder.
    encoder_disposable_registers: int = 0
    """Number of disposable registers for encoder. These are prepended to latent tokens
    and discarded after encoding. Sequence order: [disposable, latents, patches]."""
    decoder_disposable_registers: int = 0
    """Number of disposable registers for decoder. These are prepended to output patches
    and discarded after decoding. Sequence order: [disposable, output_patches, latent_input]."""

    tanh: bool = False
    """If True, apply tanh activation to the latents after sampling. This bounds latents to [-1, 1]."""

    intermediate_layer: int | None = None
    """If set, also reconstruct DINO features from this intermediate layer (0-indexed).
    The decoder output_dim should be doubled (768 * 2) to output both final and intermediate."""
    intermediate_weight: float = 1.0
    """Weight for intermediate layer reconstruction loss (relative to final layer loss)."""

    @property
    def num_latents(self) -> int:
        """Number of actual latent tokens (excluding disposable registers)."""
        return self.encoder.num_registers - self.encoder_disposable_registers

    @property
    def num_output_patches(self) -> int:
        """Number of actual output patches (excluding disposable registers)."""
        return self.decoder.num_registers - self.decoder_disposable_registers


class FlatDinoAutoencoder(nnx.Module):
    def __init__(
        self,
        cfg: FlatDinoConfig,
        mesh: jax.sharding.Mesh | None = None,
        *,
        mngr: ocp.CheckpointManager | None = None,
        step: int | None = None,
        mp: jmp.Policy,
        rngs: nnx.Rngs,
    ):
        assert cfg.encoder.patch is None
        assert cfg.decoder.patch is None

        self.cfg = cfg

        if mngr is None:
            self.encoder = ViTEncoder(cfg.encoder, mp, rngs=rngs)
            self.decoder = ViTEncoder(cfg.decoder, mp, rngs=rngs)
        else:
            step = mngr.latest_step() if step is None else step
            assert mesh is not None, "mesh must be specified when restoring"
            assert step is not None, "could not restore checkpoint"

            self.encoder = ViTEncoder.restore(mngr, step, "encoder", mesh, cfg=cfg.encoder, mp=mp)
            self.decoder = ViTEncoder.restore(mngr, step, "decoder", mesh, cfg=cfg.decoder, mp=mp)

    @staticmethod
    def get_item_names() -> list[str]:
        return ["encoder", "decoder"]

    def get_state(self):
        return {
            "encoder": nnx.to_pure_dict(nnx.state(self.encoder)),
            "decoder": nnx.to_pure_dict(nnx.state(self.decoder)),
        }

    @property
    def trainable_pytree(self):
        # Must use dict with same keys as module attributes so structure matches
        # the gradients returned by nnx.value_and_grad
        return {"encoder": self.encoder, "decoder": self.decoder}

    @property
    def trainable_state(self):
        return nnx.state(self.trainable_pytree, nnx.Param)

    def encode(
        self, x: jax.Array, *, key: jax.Array | None = None
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Encode DINO patches to latents.

        Args:
            x: DINO normalized patches of shape (B, 256, 768)
            key: Optional RNG key for stochastic sampling. If None, returns mu (deterministic).

        Returns:
            z: Latent codes of shape (B, num_latents, latent_dim), with tanh applied if configured.
            aux: Dict containing 'mu' and 'logvar' for KL computation.
        """
        encoded = self.encoder(x)
        num_latents = self.cfg.num_latents
        skip = self.cfg.encoder_disposable_registers

        # Split encoder output into mu and logvar
        mu, logvar = jnp.split(encoded, 2, axis=-1)
        mu = mu[:, skip : skip + num_latents]
        logvar = logvar[:, skip : skip + num_latents]

        if key is not None:
            std = jnp.exp(0.5 * logvar)
            eps = jax.random.normal(key, mu.shape, dtype=mu.dtype)
            z = mu + eps * std
        else:
            z = mu

        if self.cfg.tanh:
            z = jnp.tanh(z)

        return z, {"mu": mu, "logvar": logvar}

    def decode(
        self, z: jax.Array, return_intermediate: bool = False
    ) -> jax.Array | dict[str, jax.Array]:
        """Decode latents to DINO patch reconstructions.

        Args:
            z: Latent codes of shape (B, num_latents, latent_dim)
            return_intermediate: If True and intermediate_layer is configured, return dict
                with both 'final' and 'intermediate' reconstructions.

        Returns:
            If return_intermediate=False: Reconstructed DINO patches of shape (B, num_output_patches, 768)
            If return_intermediate=True and intermediate_layer is set: Dict with 'final' and
                'intermediate' reconstructions, each of shape (B, num_output_patches, 768)
        """
        decoded = self.decoder(z)
        num_output_patches = self.cfg.num_output_patches
        skip = self.cfg.decoder_disposable_registers
        output = decoded[:, skip : skip + num_output_patches]

        if not return_intermediate or self.cfg.intermediate_layer is None:
            # When not using intermediate, output_dim is 768, return as-is
            # When using intermediate but not requested, split and return final only
            if self.cfg.intermediate_layer is not None:
                output, _ = jnp.split(output, 2, axis=-1)
            return output

        # Split concatenated output into final and intermediate
        final, intermediate = jnp.split(output, 2, axis=-1)
        return {"final": final, "intermediate": intermediate}
