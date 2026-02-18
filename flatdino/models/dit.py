from typing import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jmp
import flax.nnx as nnx
from einops import rearrange


from flatdino.distributed import Restorable
from flatdino.models.transformer import TransformerConfig, TransformerDecoderLayer, LayerNorm
from flatdino.models.vit import (
    PatchEmbed,
    PatchEmbedConfig,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed,
    rotate_half,
)


# Original DiT configurations (Peebles & Xie, "Scalable Diffusion Models with Transformers")
# Uses GELU activation with mlp_ratio=4
DIT_CONFIGS = {
    "dit-s": {"embed_dim": 384, "num_layers": 12, "mlp_hidden_dim": 1536, "num_heads": 6},
    "dit-b": {"embed_dim": 768, "num_layers": 12, "mlp_hidden_dim": 3072, "num_heads": 12},
    "dit-l": {"embed_dim": 1024, "num_layers": 24, "mlp_hidden_dim": 4096, "num_heads": 16},
    "dit-xl": {"embed_dim": 1152, "num_layers": 28, "mlp_hidden_dim": 4608, "num_heads": 16},
}

# Lightning DiT configurations (parameter-matched with SwiGLU)
# SwiGLU has 3 projections vs GELU's 2, so mlp_hidden = 8/3 * embed_dim to match params
LIGHTNING_DIT_CONFIGS = {
    "dit-s": {"embed_dim": 384, "num_layers": 12, "mlp_hidden_dim": 1024, "num_heads": 6},
    "dit-b": {"embed_dim": 768, "num_layers": 12, "mlp_hidden_dim": 2048, "num_heads": 12},
    "dit-l": {"embed_dim": 1024, "num_layers": 24, "mlp_hidden_dim": 2731, "num_heads": 16},
    "dit-xl": {"embed_dim": 1152, "num_layers": 28, "mlp_hidden_dim": 3072, "num_heads": 16},
}


@dataclass
class DiTConfig:
    in_channels: int = 768
    num_registers: int = 0
    wo_shift: bool = False
    learn_sigma: bool = False
    use_rope: bool = True

    num_classes: int = 1_000
    class_dropout_prob: float = 0.1

    num_patches: int | None = None
    """This can be set only if patch_embed is set to None. It stores the number of input
    patches the model will get."""
    patch_embed: PatchEmbedConfig | None = field(
        default_factory=lambda: PatchEmbedConfig(img_size=16, patch_size=1, in_channels=768)
    )
    transformer: TransformerConfig = field(
        default_factory=lambda: TransformerConfig(
            embed_dim=384,
            num_heads=6,
            num_layers=12,
            mlp_hidden_dim=3702,
            linear_kernel_init="trunc_normal",
            mlp_type="swiglu",
            norm_type="rms",
            qk_norm=False,
        )
    )

    in_cls_dim: int | None = None
    """If set, it should be the dimension of the CLS token of the representation encoder.
    Then, we must input the CLS token in the forward pass, since it will be diffused alongside
    the other tokens."""

    def __post_init__(self):
        if self.patch_embed is not None and self.num_patches is not None:
            raise ValueError(
                "DITConfig: both patch_embed and num_patches specified, only one should be set."
            )
        if self.patch_embed is None and self.num_patches is None:
            raise ValueError("DiTConfig: please spicify num_patches or patch_embed")


def modulate(x: jax.Array, shift: jax.Array | None, scale: jax.Array | None):
    """AdaLN modulation.

    Args:
      x (jax.Array): tensor to be modulated BTD.
      shift (jax.Array | None): optional BD shift tensor
      scale (jax.Array | None): optional BD scale tensor

    Returns:
      m (jax.Array): modulated tensor
    """
    match (shift is None, scale is None):
        case (True, True):
            return x
        case (False, False):
            return x * (1 + jnp.expand_dims(scale, axis=1)) + jnp.expand_dims(shift, axis=1)
        case (True, False):
            return x * (1 + jnp.expand_dims(scale, axis=1))
        case (False, True):
            return x + jnp.expand_dims(shift, axis=1)
        case (_, _):
            raise ValueError("how did you get here?!")


class GaussianFourierEmbedding(nnx.Module):
    def __init__(
        self, hidden_size: int, embedding_size: int = 256, scale: float = 1.0, *, rngs: nnx.Rngs
    ):
        self.embedding_size = embedding_size
        self.scale = scale

        # FSDP Megatron pattern:
        # - First layer: column-parallel (None, 'model')
        # - Second layer: row-parallel ('model', None) to consume sharded input
        col_kernel_init = nnx.with_partitioning(
            nnx.initializers.truncated_normal(0.02), (None, "model")
        )
        row_kernel_init = nnx.with_partitioning(
            nnx.initializers.truncated_normal(0.02), ("model", None)
        )

        self.W = nnx.Variable(scale * jax.random.normal(rngs(), (embedding_size,)))
        self.mlp = nnx.Sequential(
            nnx.Linear(
                embedding_size * 2,
                hidden_size,
                use_bias=True,
                kernel_init=col_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            ),
            nnx.silu,
            nnx.Linear(
                hidden_size,
                hidden_size,
                use_bias=True,
                kernel_init=row_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            ),
        )

    def __call__(self, t: jax.Array) -> jax.Array:
        t = t[:, None] * self.W[None, :] * 2 * jnp.pi
        t_embed = jnp.concatenate((jnp.sin(t), jnp.cos(t)), axis=-1)
        t_embed = self.mlp(t_embed)
        return t_embed


class LabelEmbedder(nnx.Module):
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float, *, rngs: nnx.Rngs):
        use_cfg_embedding = dropout_prob > 0
        # FSDP: shard embedding along hidden dimension
        embed_init = nnx.with_partitioning(
            nnx.initializers.truncated_normal(0.02), (None, "model")
        )
        self.embedding_table = nnx.Embed(
            num_classes + use_cfg_embedding,
            hidden_size,
            embedding_init=embed_init,
            rngs=rngs,
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.rngs = rngs

    def token_drop(self, labels: jax.Array, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = jax.random.uniform(self.rngs(), (labels.shape[0],)) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels

    def __call__(self, labels: jax.Array, train: bool, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class FlatRotaryEmbedding(nnx.Module):
    """RoPE for flat (1D) token sequences."""

    def __init__(self, head_dim: int, seq_len: int, theta: float = 10_000.0):
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings.")
        base_freqs = 1.0 / (theta ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
        pos = jnp.arange(seq_len, dtype=jnp.float32)
        freqs = jnp.einsum("n,f->n f", pos, base_freqs)
        freqs = jnp.repeat(freqs, 2, axis=-1)
        self.freqs_cos = nnx.Variable(jnp.cos(freqs))
        self.freqs_sin = nnx.Variable(jnp.sin(freqs))

    def __call__(self, t: jax.Array) -> jax.Array:
        _, seq_len, _, _ = t.shape  # B, T, num_heads, dim
        if seq_len % self.freqs_cos.shape[0] != 0:
            raise ValueError("sequence length not compatible with precomputed RoPE table")
        repeat = seq_len // self.freqs_cos.shape[0]
        cos = self.freqs_cos if repeat == 1 else jnp.repeat(self.freqs_cos, repeat, axis=0)
        sin = self.freqs_sin if repeat == 1 else jnp.repeat(self.freqs_sin, repeat, axis=0)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return t * cos + rotate_half(t) * sin


class LightningDiTBlock(TransformerDecoderLayer):
    def __init__(self, cfg: DiTConfig, mp: jmp.Policy, *, rngs: nnx.Rngs):
        super().__init__(cfg.transformer, mp, rngs=rngs)
        self.wo_shift = cfg.wo_shift
        cfg_t = cfg.transformer
        input_dim: int = cfg_t.embed_dim if cfg_t.embed_dim is not None else cfg_t.input_dim  # ty: ignore
        adaln_dim = 4 * input_dim if cfg.wo_shift else 6 * input_dim
        # FSDP: shard along output dimension
        kernel_init = nnx.with_partitioning(nnx.initializers.zeros_init(), (None, "model"))
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                input_dim,
                adaln_dim,
                param_dtype=mp.param_dtype,
                dtype=mp.compute_dtype,
                kernel_init=kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            ),
        )

    def __call__(
        self, x: jax.Array, c: jax.Array, rope: Callable | None = None, deterministic: bool = True
    ) -> jax.Array:
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = jnp.split(
                self.adaLN_modulation(c), 4, axis=-1
            )
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
                self.adaLN_modulation(c), 6, axis=-1
            )

        attn = self.attention(modulate(self.att_norm(x), shift_msa, scale_msa), rope=rope)
        attn = jnp.expand_dims(gate_msa, axis=1) * attn
        x = x + self.drop_path(attn, deterministic=deterministic)

        mlp = self.mlp(modulate(self.mlp_norm(x), shift_mlp, scale_mlp))
        mlp = jnp.expand_dims(gate_mlp, axis=1) * mlp
        x = x + self.drop_path(mlp, deterministic=deterministic)

        return x


# TODO: fix the use of patch when there is no patchembed
class LightningFinalLayer(nnx.Module):
    def __init__(
        self, cfg: DiTConfig, mp: jmp.Policy, *, num_tokens: int | None = None, rngs: nnx.Rngs
    ):
        out_channels = cfg.in_channels if not cfg.learn_sigma else 2 * cfg.in_channels
        norm_type = cfg.transformer.norm_type
        if cfg.patch_embed is not None:
            assert num_tokens is None
            patch_size = cfg.patch_embed.patch_size
            out_dim = patch_size * patch_size * out_channels
        else:
            assert num_tokens is not None
            out_dim = out_channels

        # FSDP Megatron pattern:
        # - adaLN_modulation: column-parallel (modulation applied to sharded hidden states)
        # - linear: row-parallel (final output should be replicated after all-reduce)
        col_kernel_init = nnx.with_partitioning(nnx.initializers.zeros_init(), (None, "model"))
        row_kernel_init = nnx.with_partitioning(nnx.initializers.zeros_init(), ("model", None))

        self.norm_final = LayerNorm(
            cfg.transformer.residual_dim,
            norm_type,
            epsilon=1e-6,
            param_dtype=mp.param_dtype,
            use_scale=norm_type == "rms",
            use_bias=False,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        # TODO: this should be the output patch size!
        self.linear = nnx.Linear(
            cfg.transformer.residual_dim,
            out_dim,
            use_bias=True,
            kernel_init=row_kernel_init,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                cfg.transformer.residual_dim,
                2 * cfg.transformer.residual_dim,
                use_bias=True,
                kernel_init=col_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            ),
        )

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class LightningDiT(nnx.Module, Restorable):
    def __init__(
        self,
        cfg: DiTConfig,
        mp: jmp.Policy,
        *,
        num_tokens: int | None = None,
        rngs: nnx.Rngs,
        latent_mean: jax.Array | None = None,
        latent_std: jax.Array | None = None,
    ):
        """LightningDiT.
        num_tokens (int | None): if set, PatchEmbed must be dissabled.
        """
        assert cfg.num_registers == 0, (
            "LightningDiT reference does not use register tokens; set num_registers=0."
        )
        self.cfg = cfg
        cfg_t = cfg.transformer
        self.out_channels = cfg.in_channels if not cfg.learn_sigma else 2 * cfg.in_channels

        # FSDP Megatron pattern:
        # - Entry layers (x_embedder, in_cls_proj): column-parallel (None, 'model')
        # - Exit layers (out_cls_proj): row-parallel ('model', None)
        col_kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), (None, "model"))
        row_kernel_init = nnx.with_partitioning(nnx.initializers.xavier_uniform(), ("model", None))
        embed_init_base = nnx.initializers.truncated_normal(0.02)

        if cfg.num_registers > 0:
            self.registers = nnx.Param(
                embed_init_base(rngs(), (1, cfg.num_registers, cfg_t.residual_dim)),
                sharding=(None, None, "model"),
            )
        else:
            self.registers = None

        if cfg.patch_embed is not None:
            assert num_tokens is None
            self.x_embedder = PatchEmbed(cfg.patch_embed, cfg_t.residual_dim, mp, rngs=rngs)
            num_pos_embeds = int(self.x_embedder.num_patches**0.5)
            hw_seq_len = cfg.patch_embed.img_size // cfg.patch_embed.patch_size
            self.pos_embed = nnx.Variable(
                jnp.asarray(get_2d_sincos_pos_embed(cfg_t.residual_dim, num_pos_embeds))
            )
        else:
            assert num_tokens is not None
            self.x_embedder = nnx.Linear(
                cfg.in_channels,
                cfg_t.residual_dim,
                use_bias=cfg.transformer.mlp_bias,
                kernel_init=col_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                param_dtype=mp.param_dtype,
                rngs=rngs,
            )
            hw_seq_len = num_tokens
            self.pos_embed = nnx.Variable(
                jnp.asarray(get_1d_sincos_pos_embed(cfg_t.residual_dim, num_tokens))
            )
        self.t_embedder = GaussianFourierEmbedding(
            cfg_t.residual_dim, embedding_size=256, rngs=rngs
        )
        self.y_embedder = LabelEmbedder(
            cfg.num_classes, cfg_t.residual_dim, cfg.class_dropout_prob, rngs=rngs
        )

        if self.cfg.in_cls_dim is not None:
            self.in_cls_proj = nnx.Linear(
                self.cfg.in_cls_dim, cfg_t.residual_dim, kernel_init=col_kernel_init, rngs=rngs
            )
            self.out_cls_proj = nnx.Linear(
                cfg_t.residual_dim, self.cfg.in_cls_dim, kernel_init=row_kernel_init, rngs=rngs
            )

        if self.cfg.use_rope:
            if cfg.patch_embed is not None:
                half_head_dim = cfg_t.inner_dim // cfg_t.num_heads // 2
                self.feat_rope = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=hw_seq_len)
            else:
                self.feat_rope = FlatRotaryEmbedding(cfg.transformer.h_dim, cfg.num_patches)
        else:
            self.feat_rope = None

        self.blocks = nnx.List(
            [LightningDiTBlock(cfg, mp, rngs=rngs) for _ in range(cfg.transformer.num_layers)]
        )
        self.final_layer = LightningFinalLayer(cfg, mp, num_tokens=num_tokens, rngs=rngs)

        # Latent normalization statistics (only created when provided)
        # This ensures backward compatibility with checkpoints that don't have these variables
        if latent_mean is not None and latent_std is not None:
            self.latent_mean = nnx.Variable(jnp.asarray(latent_mean, dtype=mp.param_dtype))
            self.latent_std = nnx.Variable(jnp.asarray(latent_std, dtype=mp.param_dtype))

    def normalize(self, z: jax.Array) -> jax.Array:
        """Normalize latents to zero mean, unit variance.

        If no stats were provided at init, returns z unchanged (identity transform).
        """
        if not hasattr(self, "latent_mean"):
            return z
        return (z - self.latent_mean.value) / self.latent_std.value

    def denormalize(self, z: jax.Array) -> jax.Array:
        """Denormalize latents back to original scale.

        If no stats were provided at init, returns z unchanged (identity transform).
        """
        if not hasattr(self, "latent_std"):
            return z
        return z * self.latent_std.value + self.latent_mean.value

    def unpatchify(self, x: jax.Array) -> jax.Array:
        assert self.cfg.patch_embed is not None
        b, t, _ = x.shape
        c = self.out_channels
        p = self.x_embedder.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        return rearrange(x, "b (h w) (p1 p2 c) -> b (h p1) (w p2) c", h=h, w=w, p1=p, p2=p, c=c)

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array | None = None,
        y: jax.Array | None = None,
        *,
        train: bool = True,
        cls_: jax.Array | None = None,
    ):
        x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        y = self.y_embedder(y, train=train)
        c = t + y

        if self.cfg.num_registers > 0:
            regs = jnp.repeat(self.registers, c.shape[0], 0)
            x = jnp.concatenate((regs, x), axis=1)

        if cls_ is not None:
            assert self.cfg.in_cls_dim is not None
            cls_in = self.in_cls_proj(cls_)
            x = jnp.concatenate((jnp.expand_dims(cls_in, axis=1), x), axis=1)

        rope = self.feat_rope
        if cls_ is not None and self.feat_rope is not None:
            # Skip RoPE for CLS so sequence length matches the precomputed table.
            def rope_skip_cls(t: jax.Array) -> jax.Array:
                cls_tok = t[:, :1]
                rest = t[:, 1:]
                if rest.shape[1] == 0:
                    return t
                rest = self.feat_rope(rest)
                return jnp.concatenate((cls_tok, rest), axis=1)

            rope = rope_skip_cls

        for block in self.blocks:
            x = block(x, c, rope=rope)

        if cls_ is not None:
            cls_out = self.out_cls_proj(x[:, 0])
            x = x[:, 1:]

        x = self.final_layer(x, c)  # (N, T, p ** 2 * out_channels)

        if self.cfg.patch_embed is not None:
            x = self.unpatchify(x)

        if self.cfg.learn_sigma:
            x, _ = jnp.array_split(x, 2, axis=-1)

        # If using registers, do not return them
        if self.cfg.num_registers > 0:
            x = x[:, self.cfg.num_registers :]

        if cls_ is not None:
            return {"cls": cls_out, "x": x}
        else:
            return {"x": x}
