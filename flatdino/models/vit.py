from typing import Literal
from dataclasses import dataclass, field
import math

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import jmp
import numpy as np
from einops import rearrange, repeat

from flatdino.models.transformer import TransformerConfig, TransformerDecoderLayer, LayerNorm
from flatdino.distributed import Restorable, jax_unstack


@dataclass
class PatchEmbedConfig:
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    init_std: float = 0.02
    kernel_init: Literal["xavier_uniform", "trunc_norm"] = "trunc_norm"


@dataclass
class ViTConfig:
    patch: PatchEmbedConfig | None = field(default_factory=lambda: PatchEmbedConfig())
    num_patches: int | None = None
    """This is used to create the positional embeddings."""

    init_std: float = 0.02

    use_pos_embeds: bool = True
    pos_embed_type: Literal["learned", "sincos"] = "learned"
    """Type of positional embeddings: 'learned' for trainable, 'sincos' for fixed sin/cos."""
    use_cls: bool = False
    num_registers: int = 0
    input_dim: int | None = None
    """Dimensionality of the input. If not specified, it is assumed to be
    defined by the residual_dim of the transformer. If it is specified and it
    does not match residual_dim, then a learnable linear projection will be added to project
    the input to the correct residual_dim. input_dim can only be specified if patches is None.
    """
    output_dim: int | None = None
    """Dimensionality of the output. If not specified, it is assumed to be the residual_dim of the
    transformer. IF it is specified and it does not match residual_dim, then a learnable linear projection
    will be added at the output."""

    transformer: TransformerConfig = field(default_factory=lambda: TransformerConfig())

    def __post_init__(self):
        match (self.patch is None, self.num_patches, self.input_dim):
            case (True, None, _):
                raise ValueError("num_patches required when patch is None")
            case (False, n, i) if n is not None:
                raise ValueError("num_patches must be None when patch is set.")
            case (False, _, i) if i is not None:
                raise ValueError("input_dim must be None when patch is set.")


VIT_CONFIGS = {
    "vit-t": {"embed_dim": 192, "num_layers": 12, "mlp_hidden_dim": 768, "num_heads": 3},
    "vit-s": {"embed_dim": 384, "num_layers": 12, "mlp_hidden_dim": 1536, "num_heads": 6},
    "vit-b": {"embed_dim": 768, "num_layers": 12, "mlp_hidden_dim": 3072, "num_heads": 12},
    "vit-l": {"embed_dim": 1024, "num_layers": 24, "mlp_hidden_dim": 4096, "num_heads": 16},
    "vit-xl": {"embed_dim": 1280, "num_layers": 32, "mlp_hidden_dim": 5120, "num_heads": 16},
}


def swiglu_vit_config(name: str) -> dict:
    """Get VIT_CONFIG with mlp_hidden_dim adjusted for parameter-matched SwiGLU.

    SwiGLU uses 3 weight matrices (up, gate, down) vs GELU's 2 (up, down).
    To match parameter count, scale hidden_dim by 2/3.
    """
    cfg = VIT_CONFIGS[name].copy()
    cfg["mlp_hidden_dim"] = round(cfg["mlp_hidden_dim"] * 2 / 3)
    return cfg


class PatchEmbed(nnx.Module):
    def __init__(
        self,
        cfg: PatchEmbedConfig,
        embed_dim: int,
        mp_policy: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        image_hw = [cfg.img_size] * 2 if isinstance(cfg.img_size, int) else cfg.img_size
        patch_hw = [cfg.patch_size] * 2 if isinstance(cfg.patch_size, int) else cfg.patch_size
        grid_size = [image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1]]

        match cfg.kernel_init:
            case "xavier":
                kernel_init_base = nnx.initializers.xavier_uniform()
            case "trunc_norm":
                kernel_init_base = nnx.initializers.truncated_normal(cfg.init_std)
            case _:
                raise ValueError("invalid kernel_init")

        # FSDP: shard along embed_dim (output dimension)
        kernel_init = nnx.with_partitioning(kernel_init_base, (None, "model"))

        self.patch_size = cfg.patch_size
        self.num_patches = grid_size[0] * grid_size[1]
        self.proj = nnx.Linear(
            cfg.in_channels * patch_hw[0] * patch_hw[1],
            embed_dim,
            use_bias=True,
            kernel_init=kernel_init,
            bias_init=nnx.initializers.zeros_init(),
            param_dtype=mp_policy.param_dtype,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        _, H, W, _ = x.shape
        ph, pw = self.patch_size, self.patch_size

        x = rearrange(x, "b (h ph) (w pw) c -> b (h w) (c ph pw)", ph=ph, pw=pw)
        x = self.proj(x)
        return x


def broadcat(tensors: list[jax.Array], axis: int = -1) -> jax.Array:
    """
    Broadcastable concatenation along `axis`, similar to PyTorch's
    .expand + torch.cat, but implemented in JAX.

    All tensors must:
      - Have the same rank.
      - Be broadcastable along all non-concatenation dimensions.
    """
    if not tensors:
        raise ValueError("tensors must be a non-empty list")

    num_tensors = len(tensors)
    ranks = [t.ndim for t in tensors]
    if len(set(ranks)) != 1:
        raise ValueError("all tensors must share the same rank")
    rank = ranks[0]

    dim = axis + rank if axis < 0 else axis

    shapes = [t.shape for t in tensors]
    dims = list(zip(*shapes))

    expandable_dims = [(i, sizes) for i, sizes in enumerate(dims) if i != dim]

    def _broadcastable(sizes_for_dim):
        """True if this dimension is broadcastable across tensors."""
        uniq = set(sizes_for_dim)
        if len(uniq) == 1:
            # All same -> ok
            return True
        # Allow only {1, max} pattern, like standard broadcasting
        if len(uniq) == 2 and 1 in uniq:
            return True
        return False

    cond = all(_broadcastable(sizes) for _, sizes in expandable_dims)
    if not cond:
        raise ValueError("invalid dimensions for broadcastable concatenation")

    max_dims = [(i, max(sizes)) for i, sizes in expandable_dims]

    expanded_dims = [(i, (size,) * num_tensors) for i, size in max_dims]

    expanded_dims.insert(dim, (dim, dims[dim]))

    per_axis_sizes = [sizes for _, sizes in expanded_dims]
    expandable_shapes = list(zip(*per_axis_sizes))

    broadcasted = [jnp.broadcast_to(t, shape) for t, shape in zip(tensors, expandable_shapes)]

    return jnp.concatenate(broadcasted, axis=dim)


def rotate_half(x: jax.Array):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = jax_unstack(x, axis=-1)
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(
    embed_dim: int, seq_len: int, cls_token: bool = False, extra_tokens: int = 0
):
    """
    Create 1D sinusoidal positional embeddings.

    Args:
      embed_dim (int): embedding dimension (must be even).
      seq_len (int): number of positions to encode.
      cls_token (bool): prepend zero embeddings if True and extra_tokens > 0.
      extra_tokens (int): how many zero tokens to prepend (e.g., class or special tokens).

    Returns:
      np.ndarray of shape (seq_len + extra_tokens, embed_dim)
    """
    grid = np.arange(seq_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim], dtype=np.float32), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


class VisionRotaryEmbeddingFast(nnx.Module):
    def __init__(
        self,
        dim,
        pt_seq_len=16,
        ft_seq_len=None,
        custom_freqs=None,
        freqs_for="lang",
        theta=10000,
        max_freq=10,
        num_freqs=1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == "lang":
            freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim))
        elif freqs_for == "pixel":
            freqs = jnp.linspace(1.0, max_freq / 2, dim // 2) * jnp.pi
        elif freqs_for == "constant":
            freqs = jnp.ones((num_freqs,))
        else:
            raise ValueError(f"unknown modality {freqs_for}")
        freqs = jnp.astype(freqs, jnp.float32)

        if ft_seq_len is None:
            ft_seq_len = pt_seq_len
        t = jnp.arange(ft_seq_len, dtype=jnp.float32) / ft_seq_len * pt_seq_len

        freqs = jnp.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        freqs = broadcat([freqs[:, None, :], freqs[None, :, :]], axis=-1)

        self.freqs_cos = nnx.Variable(rearrange(jnp.cos(freqs), "... d -> (...) d"))
        self.freqs_sin = nnx.Variable(rearrange(jnp.sin(freqs), "... d -> (...) d"))

    def __call__(self, t: jax.Array) -> jax.Array:
        _, T, _, _ = t.shape  # B, L, num_heads, dim
        base = self.freqs_cos
        if T % base.shape[0] != 0:
            raise ValueError("sequence length not compatible with precomputed rope table")
        repeat = T // base.shape[0]
        cos = base if repeat == 1 else jnp.repeat(base, repeat, axis=0)
        sin = self.freqs_sin if repeat == 1 else jnp.repeat(self.freqs_sin, repeat, axis=0)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        return t * cos + rotate_half(t) * sin

        # L, _ = self.freqs_cos.shape  # L, dim
        # repeat_times = T // L
        # freqs_cos, freqs_sin = self.freqs_cos, self.freqs_sin
        # if repeat_times != 1:
        #     freqs_cos = jnp.repeat(freqs_cos, repeat_times, axis=0)
        #     freqs_sin = jnp.repeat(freqs_sin, repeat_times, axis=0)
        # return t * freqs_cos + rotate_half(t) * freqs_sin


class ViTEncoder(nnx.Module, Restorable):
    """
    Pure visual transformer encoder.
    Accepts (B, H, W, C) tensor and returns token embeddings.
    """

    def __init__(self, cfg: ViTConfig, mp: jmp.Policy, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.mp = mp
        if cfg.patch is not None:
            self.patch_embed = PatchEmbed(cfg.patch, cfg.transformer.residual_dim, mp, rngs=rngs)
            num_patches = self.patch_embed.num_patches
        else:
            self.patch_embed = None
            num_patches = cfg.num_patches

        self.use_cls = cfg.use_cls
        self.num_reg = cfg.num_registers

        kinit_base = nnx.initializers.truncated_normal(cfg.init_std)
        # FSDP: shard along embed dimension (output for projections, last dim for embeddings)
        kinit = nnx.with_partitioning(kinit_base, (None, "model"))

        if cfg.input_dim is not None and cfg.input_dim != cfg.transformer.residual_dim:
            self.input_proj = nnx.Linear(
                cfg.input_dim,
                cfg.transformer.residual_dim,
                use_bias=False,
                dtype=mp.compute_dtype,
                param_dtype=mp.param_dtype,
                kernel_init=kinit,
                rngs=rngs,
            )
        else:
            self.input_proj = None

        if cfg.output_dim is not None and cfg.output_dim != cfg.transformer.residual_dim:
            self.output_proj = nnx.Linear(
                cfg.transformer.residual_dim,
                cfg.output_dim,
                use_bias=False,
                dtype=mp.compute_dtype,
                param_dtype=mp.param_dtype,
                kernel_init=kinit,
                rngs=rngs,
            )
        else:
            self.output_proj = None

        if cfg.use_pos_embeds:
            embed_dim = cfg.transformer.residual_dim
            if cfg.pos_embed_type == "sincos":
                # Fixed sin/cos positional embeddings (not trainable)
                # Extra tokens: CLS (if used) + registers
                grid_size = int(num_patches**0.5)
                extra_tokens = (1 if cfg.use_cls else 0) + cfg.num_registers
                sincos_embed = get_2d_sincos_pos_embed(
                    embed_dim, grid_size, cls_token=(extra_tokens > 0), extra_tokens=extra_tokens
                )
                self.pos_embed = nnx.Variable(
                    jnp.asarray(sincos_embed, dtype=mp.param_dtype)[None, ...],
                )
            else:
                # Learned positional embeddings
                pos_embed_shape = (1, num_patches, embed_dim)
                self.pos_embed = nnx.Param(
                    kinit_base(rngs(), pos_embed_shape, dtype=mp.param_dtype),
                    sharding=(None, None, "model"),
                )
        else:
            self.pos_embed = None

        if self.use_cls:
            self.cls_token = nnx.Param(
                kinit_base(rngs(), (1, 1, cfg.transformer.residual_dim), dtype=mp.param_dtype),
                sharding=(None, None, "model"),
            )
        if self.num_reg > 0:
            self.reg_tokens = nnx.Param(
                kinit_base(rngs(), (1, self.num_reg, cfg.transformer.residual_dim), dtype=mp.param_dtype),
                sharding=(None, None, "model"),
            )

        depth = cfg.transformer.num_layers
        drp = [i * cfg.transformer.drop_rate / min(1, depth - 1) for i in range(depth)]
        self.blocks = nnx.List(
            [
                TransformerDecoderLayer(cfg.transformer, mp, drop_p=drp[i], rngs=rngs)
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm(cfg.transformer.residual_dim, param_dtype=mp.param_dtype, rngs=rngs)

    def interpolate(self, squared_dim: int, emb: jax.Array) -> jax.Array:
        embed_dim = emb.shape[-1]
        if squared_dim == emb.shape[1]:
            return emb

        hw = int(math.sqrt(squared_dim))
        assert hw**2 == squared_dim, f"hw: {hw}, squared_dim: {squared_dim}"

        hwp = int(math.sqrt(emb.shape[1]))
        emb_2d = rearrange(emb, "1 (h w) d -> 1 h w d", h=hwp)
        resized = jax.image.resize(emb_2d, (1, hw, hw, embed_dim), "bilinear")
        return rearrange(resized, "1 h w d -> 1 (h w) d")

    def interpolate_pos_encoding(self, tokens: jax.Array) -> jax.Array:
        """Interpolate positional embeddings to match token sequence length.

        Args:
            tokens: Token sequence of shape (B, T, D). Expected order: [CLS, REGS, PATCHES]
                where CLS and REGS are optional based on config.

        Returns:
            Positional embeddings matching token sequence length.
        """
        pos_embed = self.pos_embed.value
        # Number of prefix tokens (CLS + registers) that have zero positional embeddings
        num_prefix = (1 if self.use_cls else 0) + self.num_reg

        if num_prefix > 0 and self.cfg.pos_embed_type == "sincos":
            # Sin/cos embeddings: pos_embed has shape (1, num_prefix + num_patches, D)
            # where indices 0:num_prefix are for CLS/registers (zeros) and rest are for patches
            prefix_pos = pos_embed[:, :num_prefix, :]  # CLS + register positions (zeros for sincos)
            patch_pos = pos_embed[:, num_prefix:, :]  # Patch positions
            # Interpolate patch positions if needed (tokens.shape[1] - num_prefix = num_patches)
            num_patches = tokens.shape[1] - num_prefix
            patch_pos = self.interpolate(num_patches, patch_pos)
            return jnp.concatenate([prefix_pos, patch_pos], axis=1)
        else:
            return self.interpolate(tokens.shape[1], pos_embed)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool | None = None,
        registers: jax.Array | None = None,
        mask: jax.Array | None = None,
        return_attention_weights: bool = False,
    ) -> jax.Array | tuple[jax.Array, list[jax.Array]]:
        """Forward pass of the ViT.

        Args:
          - x (jax.Array): array of shape BHWC or BTD.
            For shapes BHWC: patch_embed must exist, and if using pos_embeds they will be
              resized if necessary to fit to the 2D image.
            For shapes BTD: patch_embed must be unset, and if pos_embeds exist, they must
              have the same shape TD.
          - deterministic (bool | None): nnx eval interface.
          - registers (jax.Array | None): if specified, use those registers instead of the model's registers.
          - mask (jax.Array | None): attention mask.
          - return_attention_weights (bool): if True, return attention weights from all layers.

        Returns:
          If return_attention_weights is False: output tokens of shape (B, T, D)
          If return_attention_weights is True: (tokens, attention_weights) where
            attention_weights is a list of arrays, one per layer, each of shape
            (B, num_heads, T, T).
        """
        tokens = self.mp.cast_to_compute(x)

        if self.input_proj is not None:
            tokens = self.input_proj(tokens)

        b = x.shape[0]

        if self.patch_embed is not None:
            tokens = self.patch_embed(x)
        # Note: tokens now has shape (B, num_patches, D)

        # Positional embedding logic depends on embed type:
        # - "learned": apply pos_embed to patches FIRST, then add registers/CLS (backward compatible)
        # - "sincos": add CLS/registers FIRST, then apply pos_embed (matches RAE decoder)
        use_sincos = self.cfg.pos_embed_type == "sincos"

        # For learned embeddings: add pos_embed to patches before registers/CLS
        if not use_sincos and self.patch_embed is not None and self.pos_embed is not None:
            tokens = tokens + self.interpolate_pos_encoding(tokens)
        elif not use_sincos and self.pos_embed is not None:
            tokens = tokens + self.pos_embed[:, : tokens.shape[1]]

        # Add registers
        if registers is not None:
            regs = jnp.expand_dims(registers, axis=0) if len(registers.shape) == 2 else registers
            regs = jnp.repeat(regs, b, axis=0)
            tokens = jnp.concatenate([regs, tokens], axis=1)
        elif self.num_reg > 0:
            regs = jnp.broadcast_to(
                self.reg_tokens.value, (b, self.num_reg, self.reg_tokens.shape[-1])
            )
            tokens = jnp.concatenate([regs, tokens], axis=1)

        # Add CLS token
        if self.use_cls:
            cls = jnp.broadcast_to(self.cls_token.value, (b, 1, self.cls_token.shape[-1]))
            tokens = jnp.concatenate([cls, tokens], axis=1)

        # For sincos embeddings: add pos_embed after CLS/registers are prepended
        # This matches the RAE decoder where pos_embed includes CLS position
        if use_sincos and self.pos_embed is not None:
            tokens = tokens + self.interpolate_pos_encoding(tokens)

        all_attention_weights = [] if return_attention_weights else None
        for block in self.blocks:
            block_out = block(
                tokens,
                attention_mask=mask,
                deterministic=deterministic,
                return_attention_weights=return_attention_weights,
            )
            if return_attention_weights:
                tokens, attn_weights = block_out
                all_attention_weights.append(attn_weights)
            else:
                tokens = block_out

        tokens = self.norm(tokens)

        if self.output_proj is not None:
            tokens = self.output_proj(tokens)

        tokens = self.mp.cast_to_output(tokens)

        if return_attention_weights:
            return tokens, all_attention_weights
        return tokens
