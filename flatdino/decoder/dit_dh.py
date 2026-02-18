from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import jmp
from einops import rearrange

from flatdino.models.transformer import TransformerConfig, Attention, MLP, LayerNorm
from flatdino.models.vit import (
    PatchEmbed,
    PatchEmbedConfig,
    VisionRotaryEmbeddingFast,
    get_2d_sincos_pos_embed,
)
from flatdino.models.dit import GaussianFourierEmbedding, LabelEmbedder, FlatRotaryEmbedding
from flatdino.distributed import Restorable


@dataclass
class DiTDHConfig:
    input_size: int = 16
    """When patch_size is set, this refers to the side of an image. Otherwise it refers
    to the number of tokens.
    """
    patch_size: int | tuple[int, int] | None = 1
    """If None, the input is expected to be flat (e.g. FlatDINO)."""
    in_channels: int = 768

    encoder_dim: int = 384
    decoder_dim: int = 2048

    encoder_depth: int = 12
    decoder_depth: int = 2

    encoder_heads: int = 6
    decoder_heads: int = 16

    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000

    use_qknorm: bool = False
    use_swiglu: bool = True
    use_rope: bool = True
    use_rmsnorm: bool = True
    wo_shift: bool = False
    use_pos_embed: bool = True

    num_registers: int = 0
    """Registers used in the DH head (no registers are used in the normal layers!)"""

    in_cls_dim: int | None = None
    """If set, it should be the dimension of the CLS token of the representation encoder.
    Then, we must input the CLS token in the forward pass, since it will be diffused alongside
    the other tokens."""



def _repeat_to_match_length(x: jax.Array | None, length: int) -> jax.Array | None:
    if x is None:
        return None
    if x.shape[1] == length:
        return x
    if length % x.shape[1] != 0:
        raise ValueError(
            f"Token length {length} is not divisible by modulation length {x.shape[1]}"
        )
    repeat = length // x.shape[1]
    return jnp.repeat(x, repeat, axis=1)


def ddt_modulate(x: jax.Array, shift: jax.Array | None, scale: jax.Array | None) -> jax.Array:
    shift = _repeat_to_match_length(shift, x.shape[1])
    scale = _repeat_to_match_length(scale, x.shape[1])
    if shift is None:
        shift = 0.0
    if scale is None:
        scale = 0.0
    return x * (1 + scale) + shift


def ddt_gate(x: jax.Array, gate: jax.Array) -> jax.Array:
    gate = _repeat_to_match_length(gate, x.shape[1])
    return x * gate


class LightningDDTBlock(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        norm_type: str,
        wo_shift: bool,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        cfg = TransformerConfig(
            embed_dim=hidden_size,
            num_heads=num_heads,
            mlp_hidden_dim=int(hidden_size * mlp_ratio),
            qkv_bias=True,
            mlp_bias=True,
            mlp_type="swiglu" if use_swiglu else "gelu",
            norm_type=norm_type,
            qk_norm=use_qknorm,
            linear_kernel_init="xavier_uniform",
            drop_mlp=0.0,
            drop_proj=0.0,
            drop_rate=0.0,
        )
        self.attention = Attention(cfg, mp, rngs=rngs)
        self.mlp = MLP(cfg, mp, rngs=rngs)
        self.att_norm = LayerNorm(
            hidden_size,
            type=norm_type,
            param_dtype=mp.param_dtype,
            use_scale=norm_type == "rms",
            use_bias=False,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.mlp_norm = LayerNorm(
            hidden_size,
            type=norm_type,
            param_dtype=mp.param_dtype,
            use_scale=norm_type == "rms",
            use_bias=False,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        adaln_dim = 4 * hidden_size if wo_shift else 6 * hidden_size
        # FSDP: adaLN modulation is column-parallel (produces conditioning)
        col_kernel_init = nnx.with_partitioning(nnx.initializers.zeros_init(), (None, "model"))
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                hidden_size,
                adaln_dim,
                param_dtype=mp.param_dtype,
                dtype=mp.compute_dtype,
                kernel_init=col_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            ),
        )
        self.wo_shift = wo_shift

    def __call__(self, x: jax.Array, c: jax.Array, rope: nnx.Module | None = None) -> jax.Array:
        if c.ndim < x.ndim:
            c = jnp.expand_dims(c, axis=1)

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

        attn = self.attention(ddt_modulate(self.att_norm(x), shift_msa, scale_msa), rope=rope)
        attn = ddt_gate(attn, gate_msa)
        x = x + attn

        mlp_out = self.mlp(ddt_modulate(self.mlp_norm(x), shift_mlp, scale_mlp))
        mlp_out = ddt_gate(mlp_out, gate_mlp)
        x = x + mlp_out
        return x


class DDTFinalLayer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        norm_type: str,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        out_dim = patch_size * patch_size * out_channels
        self.norm_final = LayerNorm(
            hidden_size,
            type=norm_type,
            epsilon=1e-6,
            param_dtype=mp.param_dtype,
            use_scale=norm_type == "rms",
            use_bias=False,
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        # FSDP: final linear is row-parallel (exit layer)
        row_kernel_init = nnx.with_partitioning(nnx.initializers.zeros_init(), ("model", None))
        self.linear = nnx.Linear(
            hidden_size,
            out_dim,
            use_bias=True,
            kernel_init=row_kernel_init,
            bias_init=nnx.initializers.zeros_init(),
            param_dtype=mp.param_dtype,
            rngs=rngs,
        )
        # FSDP: adaLN modulation is column-parallel (produces conditioning)
        col_kernel_init = nnx.with_partitioning(nnx.initializers.zeros_init(), (None, "model"))
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                hidden_size,
                2 * hidden_size,
                use_bias=True,
                kernel_init=col_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                param_dtype=mp.param_dtype,
                dtype=mp.compute_dtype,
                rngs=rngs,
            ),
        )

    def __call__(self, x: jax.Array, c: jax.Array) -> jax.Array:
        if c.ndim < x.ndim:
            c = jnp.expand_dims(c, axis=1)
        shift, scale = jnp.split(self.adaLN_modulation(c), 2, axis=-1)
        x = ddt_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTDH(nnx.Module, Restorable):
    def __init__(
        self,
        cfg: DiTDHConfig,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
        latent_mean: jax.Array | None = None,
        latent_std: jax.Array | None = None,
    ):
        self.cfg = cfg
        self.mp = mp

        # FSDP: Megatron pattern - column-parallel for entry, row-parallel for exit
        col_kernel_init = nnx.with_partitioning(
            nnx.initializers.truncated_normal(), (None, "model")
        )
        row_kernel_init = nnx.with_partitioning(
            nnx.initializers.truncated_normal(), ("model", None)
        )
        col_xavier_init = nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), (None, "model")
        )

        if cfg.patch_size is not None:
            s_patch, x_patch = (
                (cfg.patch_size, cfg.patch_size)
                if isinstance(cfg.patch_size, int)
                else (cfg.patch_size[0], cfg.patch_size[1])
            )
            self.s_patch_size = s_patch
            self.x_patch_size = x_patch

            self.x_channel_per_token = cfg.in_channels * self.x_patch_size * self.x_patch_size
            norm_type = "rms" if cfg.use_rmsnorm else "ln"

            # PatchEmbed already has FSDP sharding from vit.py
            self.s_embedder = PatchEmbed(
                PatchEmbedConfig(
                    img_size=cfg.input_size,
                    patch_size=self.s_patch_size,
                    in_channels=cfg.in_channels,
                ),
                cfg.encoder_dim,
                mp,
                rngs=rngs,
            )
            self.x_embedder = PatchEmbed(
                PatchEmbedConfig(
                    img_size=cfg.input_size,
                    patch_size=self.x_patch_size,
                    in_channels=cfg.in_channels,
                ),
                cfg.decoder_dim,
                mp,
                rngs=rngs,
            )
        else:
            self.s_patch_size = 1
            self.x_patch_size = 1
            self.x_channel_per_token = cfg.in_channels
            norm_type = "rms" if cfg.use_rmsnorm else "ln"
            # FSDP: entry embedders are column-parallel
            self.s_embedder = nnx.Linear(
                cfg.in_channels,
                cfg.encoder_dim,
                kernel_init=col_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                param_dtype=mp.param_dtype,
                rngs=rngs,
            )
            self.x_embedder = nnx.Linear(
                cfg.in_channels,
                cfg.decoder_dim,
                kernel_init=col_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                param_dtype=mp.param_dtype,
                rngs=rngs,
            )

        if cfg.encoder_dim != cfg.decoder_dim:
            # FSDP: projector between encoder and decoder is column-parallel
            self.s_projector = nnx.Linear(
                cfg.encoder_dim,
                cfg.decoder_dim,
                use_bias=True,
                kernel_init=col_xavier_init,
                bias_init=nnx.initializers.zeros_init(),
                param_dtype=mp.param_dtype,
                rngs=rngs,
            )
        else:
            self.s_projector = lambda x: x

        # GaussianFourierEmbedding and LabelEmbedder already have FSDP sharding
        self.t_embedder = GaussianFourierEmbedding(cfg.encoder_dim, embedding_size=256, rngs=rngs)
        self.y_embedder = LabelEmbedder(
            cfg.num_classes, cfg.encoder_dim, cfg.class_dropout_prob, rngs=rngs
        )

        if cfg.in_cls_dim is not None:
            # FSDP: in_cls_proj is column-parallel (entry), out_cls_proj is row-parallel (exit)
            self.in_cls_proj = nnx.Linear(
                cfg.in_cls_dim,
                cfg.encoder_dim,
                kernel_init=col_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                param_dtype=mp.param_dtype,
                rngs=rngs,
            )
            self.out_cls_proj = nnx.Linear(
                cfg.decoder_dim,
                cfg.in_cls_dim,
                kernel_init=row_kernel_init,
                bias_init=nnx.initializers.zeros_init(),
                param_dtype=mp.param_dtype,
                rngs=rngs,
            )

        if cfg.use_pos_embed:
            if cfg.patch_size is not None:
                num_patches = int(self.s_embedder.num_patches)
                grid = int(num_patches**0.5)
                self.pos_embed = nnx.Variable(
                    jnp.asarray(get_2d_sincos_pos_embed(cfg.encoder_dim, grid))
                )
            else:
                # FSDP: learnable pos_embed sharded on embedding dimension
                kinit_base = nnx.initializers.truncated_normal(0.02)
                pos_embed_shape = (1, cfg.input_size, cfg.encoder_dim)
                self.pos_embed = nnx.Param(
                    kinit_base(rngs(), pos_embed_shape, dtype=mp.param_dtype),
                    sharding=(None, None, "model"),
                )
        else:
            self.pos_embed = None
        self.x_pos_embed = None

        if cfg.use_rope:
            if cfg.patch_size is not None:
                enc_half_head_dim = cfg.encoder_dim // cfg.encoder_heads // 2
                enc_seq_len = cfg.input_size // self.s_patch_size
                self.enc_feat_rope = VisionRotaryEmbeddingFast(
                    dim=enc_half_head_dim, pt_seq_len=enc_seq_len
                )

                dec_half_head_dim = cfg.decoder_dim // cfg.decoder_heads // 2
                dec_seq_len = cfg.input_size // self.x_patch_size
                self.dec_feat_rope = VisionRotaryEmbeddingFast(
                    dim=dec_half_head_dim, pt_seq_len=dec_seq_len
                )
            else:
                enc_head_dim = cfg.encoder_dim // cfg.encoder_heads
                dec_head_dim = cfg.decoder_dim // cfg.decoder_heads
                self.enc_feat_rope = FlatRotaryEmbedding(enc_head_dim, cfg.input_size)
                self.dec_feat_rope = FlatRotaryEmbedding(dec_head_dim, cfg.input_size)
        else:
            self.enc_feat_rope = None
            self.dec_feat_rope = None

        self.num_reg = cfg.num_registers
        if self.num_reg > 0:
            # FSDP: register tokens sharded on embedding dimension
            kinit_base = nnx.initializers.truncated_normal()
            enc_shape = (1, self.num_reg, cfg.encoder_dim)
            dec_shape = (1, self.num_reg, cfg.decoder_dim)
            self.reg_enc_tokens = nnx.Param(
                kinit_base(rngs(), enc_shape, dtype=mp.param_dtype),
                sharding=(None, None, "model"),
            )
            self.reg_dec_tokens = nnx.Param(
                kinit_base(rngs(), dec_shape, dtype=mp.param_dtype),
                sharding=(None, None, "model"),
            )

        self.encoder_blocks = nnx.List(
            [
                LightningDDTBlock(
                    cfg.encoder_dim,
                    cfg.encoder_heads,
                    cfg.mlp_ratio,
                    cfg.use_qknorm,
                    cfg.use_swiglu,
                    norm_type,
                    cfg.wo_shift,
                    mp,
                    rngs=rngs,
                )
                for _ in range(cfg.encoder_depth)
            ]
        )
        self.decoder_blocks = nnx.List(
            [
                LightningDDTBlock(
                    cfg.decoder_dim,
                    cfg.decoder_heads,
                    cfg.mlp_ratio,
                    cfg.use_qknorm,
                    cfg.use_swiglu,
                    norm_type,
                    cfg.wo_shift,
                    mp,
                    rngs=rngs,
                )
                for _ in range(cfg.decoder_depth)
            ]
        )
        self.final_layer = DDTFinalLayer(
            cfg.decoder_dim,
            self.x_patch_size,
            self.x_channel_per_token,
            norm_type,
            mp,
            rngs=rngs,
        )

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
        if self.cfg.patch_size is None:
            return x
        p = self.x_patch_size
        h = w = int(x.shape[1] ** 0.5)
        return rearrange(
            x,
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            h=h,
            w=w,
            p1=p,
            p2=p,
            c=self.x_channel_per_token,
        )

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        y: jax.Array,
        *,
        train: bool = True,
        cls_: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        t_embed = self.t_embedder(t)
        y_embed = self.y_embedder(y, train=train)
        c = jax.nn.silu(t_embed + y_embed)

        cls_enc = None
        cls_dec = None
        if cls_ is not None:
            assert self.cfg.in_cls_dim is not None
            cls_enc = self.in_cls_proj(cls_)
            cls_dec = self.s_projector(cls_enc[:, None, :])

        s = self.s_embedder(x)
        if self.pos_embed is not None:
            pos_tokens = self.pos_embed.shape[-2]
            if s.shape[1] != pos_tokens:
                raise ValueError(
                    f"Positional embedding length ({pos_tokens}) "
                    f"does not match token length ({s.shape[1]})."
                )
            s = s + self.pos_embed

        if self.num_reg > 0:
            b, r, d = s.shape[0], self.num_reg, s.shape[2]
            regs = jnp.broadcast_to(self.reg_enc_tokens.value, (b, r, d))
            s = jnp.concatenate((regs, s), axis=1)

        if cls_enc is not None:
            s = jnp.concatenate((cls_enc[:, None, :], s), axis=1)

        enc_rope = self.enc_feat_rope
        if cls_enc is not None and self.enc_feat_rope is not None:

            def rope_skip_cls(tokens: jax.Array) -> jax.Array:
                cls_tok = tokens[:, :1]
                rest = tokens[:, 1:]
                if rest.shape[1] == 0:
                    return tokens
                rest = self.enc_feat_rope(rest)
                return jnp.concatenate((cls_tok, rest), axis=1)

            enc_rope = rope_skip_cls

        for block in self.encoder_blocks:
            s = block(s, c, rope=enc_rope)

        t_tokens = jnp.broadcast_to(t_embed[:, None, :], s.shape)
        s = jax.nn.silu(t_tokens + s)

        s = self.s_projector(s)

        x_tokens = self.x_embedder(x)
        if self.x_pos_embed is not None:
            x_tokens = x_tokens + self.x_pos_embed

        if self.num_reg > 0:
            b, r, d = x_tokens.shape[0], self.num_reg, self.reg_dec_tokens.shape[-1]
            regs = jnp.broadcast_to(self.reg_dec_tokens.value, (b, r, d))
            x_tokens = jnp.concatenate((regs, x_tokens), axis=1)

        if cls_dec is not None:
            x_tokens = jnp.concatenate((cls_dec, x_tokens), axis=1)

        dec_rope = self.dec_feat_rope
        if cls_dec is not None and self.dec_feat_rope is not None:

            def rope_skip_cls(tokens: jax.Array) -> jax.Array:
                cls_tok = tokens[:, :1]
                rest = tokens[:, 1:]
                if rest.shape[1] == 0:
                    return tokens
                rest = self.dec_feat_rope(rest)
                return jnp.concatenate((cls_tok, rest), axis=1)

            dec_rope = rope_skip_cls

        for block in self.decoder_blocks:
            x_tokens = block(x_tokens, s, rope=dec_rope)

        cls_out = None
        start = 0
        if cls_dec is not None:
            cls_out = self.out_cls_proj(x_tokens[:, 0])
            start += 1
        if self.num_reg > 0:
            start += self.num_reg

        if start > 0:
            s = s[:, start:]
            x_tokens = x_tokens[:, start:]

        x_tokens = self.final_layer(x_tokens, s)
        x_tokens = self.unpatchify(x_tokens)

        if cls_out is not None:
            return {"cls": cls_out, "x": x_tokens}
        return {"x": x_tokens}
