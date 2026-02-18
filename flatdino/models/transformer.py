from typing import Literal, Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from flax.typing import Dtype
import jmp
from einops import rearrange


@dataclass
class TransformerConfig:
    embed_dim: int | None = None
    """The embedding dimension. If set, `head_dim` and `input_dim` must be unset. This behaves as
    if `head_dim` = `embed_dim` // `num_heads`.
    """

    head_dim: int | None = None
    """Optionally specifies the head dimension. If set, embed_dim must be unset and input_dim
    must be specified."""
    input_dim: int | None = None
    """Only can be set when head_dim is set. It indicates the dimensionality of the input vector,
    which will coincide with the dimesionality of the residual stream.
    """

    num_layers: int = 12
    mlp_hidden_dim: int = 768
    num_heads: int = 3
    qkv_bias: bool = True
    mlp_bias: bool = True
    mlp_type: Literal["gelu", "swiglu"] = "gelu"
    gelu_approximate: bool = True
    """Whether to use approximate GELU (tanh). Default True for backward compatibility."""
    norm_type: Literal["rms", "ln"] = "ln"
    qk_norm: bool = True

    linear_kernel_init: Literal["xavier_uniform", "trunc_normal"] = "trunc_normal"
    linear_init_std: float = 0.02

    drop_mlp: float = 0.0
    drop_proj: float = 0.0
    drop_rate: float = 0.0

    layer_norm_eps: float = 1e-6
    """Epsilon value for layer normalization."""

    implementation: Literal["cudnn", "xla"] = "xla"
    causal: bool = False
    selective: bool = False
    """Whether to checkpoint the attention computation for the backward pass."""

    def __post_init__(self):
        match (self.embed_dim is None, self.head_dim is None, self.input_dim is None):
            case (True, True, _):
                raise ValueError("at least one of embed_dim or embed_dim_per_head must not be None")
            case (False, False, _):
                raise ValueError("can't set both embed_dim and embed_dim_per_head at the same time")
            case (True, False, True):
                raise ValueError("input_dim must be specified if head_dim is specified")
            case (False, _, _):
                if self.embed_dim % self.num_heads != 0:
                    raise ValueError("embed_dim must be divisible by num_heads")

    @property
    def inner_dim(self) -> int:
        return self.embed_dim if self.embed_dim is not None else self.head_dim * self.num_heads

    @property
    def residual_dim(self) -> int:
        return self.embed_dim if self.embed_dim is not None else self.input_dim

    @property
    def h_dim(self) -> int:
        if self.head_dim is None:
            return self.inner_dim // self.num_heads
        else:
            return self.head_dim


class LayerNorm(nnx.Module):
    def __init__(
        self,
        embed_dim: int,
        type: Literal["rms", "ln"] = "ln",
        *,
        param_dtype: Dtype,
        epsilon: float = 1e-6,
        use_scale: bool = True,
        use_bias: bool = True,
        bias_init: nnx.Initializer = nnx.initializers.zeros_init(),
        scale_init: nnx.Initializer = nnx.initializers.ones_init(),
        rngs: nnx.Rngs,
    ):
        kwargs = {
            "epsilon": epsilon,
            "use_scale": use_scale,
            "scale_init": scale_init,
            "param_dtype": param_dtype,
        }
        match type:
            case "ln":
                self.norm = nnx.LayerNorm(
                    embed_dim, use_bias=use_bias, bias_init=bias_init, rngs=rngs, **kwargs
                )
            case "rms":
                self.norm = nnx.RMSNorm(embed_dim, rngs=rngs, **kwargs)
            case _:
                raise ValueError("invalid norm type")

    def __call__(self, x: jax.Array) -> jax.Array:
        orig_dtype = x.dtype
        x = x.astype(jnp.float32)
        return self.norm(x).astype(orig_dtype)


class DropPath(nnx.Module):
    def __init__(
        self,
        rate: float,
        deterministic: bool = False,
        rng_collection: str = "dropout",
        *,
        rngs: nnx.Rngs | nnx.RngStream | None = None,
    ):
        self.rate = rate
        self.deterministic = deterministic
        self.rng_collection = rng_collection

        if isinstance(rngs, nnx.Rngs):
            self.rngs = rngs[self.rng_collection].fork()
        elif isinstance(rngs, nnx.RngStream):
            self.rngs = rngs.fork()
        elif rngs is None:
            self.rngs = None
        else:
            raise TypeError(f"rngs must be a Rngs, RngStream or None, but got {type(rngs)}")

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool | None = None,
        rngs: nnx.Rngs | nnx.RngStream | jax.Array | None = None,
    ):
        det = deterministic if deterministic is not None else self.deterministic
        if self.rate == 0.0 or det:
            return x

        if self.rate == 1.0:
            return jnp.zeros_like(x)

        if rngs is None and self.rngs is None:
            raise ValueError(
                "`deterministic` is False, but no `rngs` argument was provided to DropPath"
            )
        rngs = rngs if rngs is not None else self.rngs

        if isinstance(rngs, nnx.Rngs):
            key = rngs[self.rng_collection]()
        elif isinstance(rngs, nnx.RngStream):
            key = rngs()
        elif isinstance(rngs, jax.Array):
            key = rngs
        else:
            raise TypeError(f"rngs must be a Rngs, RngStream or jax.Array, but got {type(rngs)}")

        keep_prob = 1 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = jax.random.bernoulli(key, p=keep_prob, shape=shape)
        mask = jnp.broadcast_to(mask, x.shape)
        return jax.lax.select(mask, x / keep_prob, jnp.zeros_like(x))


class MLP(nnx.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        mp_policy: jmp.Policy,
        *,
        init_std: float = 0.02,
        i: int | None = None,
        rngs: nnx.Rngs,
    ):
        match cfg.linear_kernel_init:
            case "xavier_uniform":
                kernel_init_base = nnx.initializers.xavier_uniform()
            case "trunc_normal":
                # init_kernel_scale = init_std if i is None else init_std / (2 * (i + 1))
                kernel_init_base = nnx.initializers.truncated_normal(cfg.linear_init_std)
            case _:
                raise ValueError("invalid kernel_init")

        # FSDP Megatron pattern:
        # - up/gate: column-parallel, shard output (None, 'model')
        # - down: row-parallel, shard input ('model', None) -> requires all-reduce after
        col_kernel_init = nnx.with_partitioning(kernel_init_base, (None, "model"))
        row_kernel_init = nnx.with_partitioning(kernel_init_base, ("model", None))

        linear_kwargs = {
            "use_bias": cfg.mlp_bias,
            "dtype": mp_policy.compute_dtype,
            "param_dtype": mp_policy.param_dtype,
            "bias_init": nnx.initializers.zeros_init(),
        }
        input_dim: int = cfg.embed_dim if cfg.embed_dim is not None else cfg.input_dim  # ty: ignore

        self.up_proj = nnx.Linear(
            input_dim,
            cfg.mlp_hidden_dim,
            kernel_init=col_kernel_init,
            rngs=rngs,
            **linear_kwargs,  # ty: ignore
        )
        self.down_proj = nnx.Linear(
            cfg.mlp_hidden_dim,
            input_dim,
            kernel_init=row_kernel_init,
            rngs=rngs,
            **linear_kwargs,  # ty: ignore
        )
        match cfg.mlp_type:
            case "gelu":
                self.gate_proj = None
            case "swiglu":
                self.gate_proj = nnx.Linear(
                    input_dim,
                    cfg.mlp_hidden_dim,
                    kernel_init=col_kernel_init,
                    rngs=rngs,
                    **linear_kwargs,
                )
            case _:
                raise ValueError("invalid mlp type")

        self.drop = nnx.Dropout(cfg.drop_mlp, rngs=rngs)
        self.gelu_approximate = cfg.gelu_approximate

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.gate_proj is None:
            x = self.drop(nnx.gelu(self.up_proj(x), approximate=self.gelu_approximate))
        else:
            x = self.drop(nnx.silu(self.gate_proj(x)) * self.up_proj(x))
        x = self.drop(self.down_proj(x))
        return x


class Attention(nnx.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        mp_policy: jmp.Policy,
        *,
        init_std: float = 0.02,
        i: int | None = None,
        rngs: nnx.Rngs,
    ):
        match cfg.linear_kernel_init:
            case "xavier_uniform":
                kernel_init_base = nnx.initializers.xavier_uniform()
            case "trunc_normal":
                # init_kernel_scale = init_std if i is None else init_std / (2 * (i + 1))
                kernel_init_base = nnx.initializers.truncated_normal(cfg.linear_init_std)
            case _:
                raise ValueError("invalid kernel_init")

        # FSDP Megatron pattern:
        # - qkv_proj: column-parallel, shard output (None, 'model')
        # - o_proj: row-parallel, shard input ('model', None) -> requires all-reduce after
        col_kernel_init = nnx.with_partitioning(kernel_init_base, (None, "model"))
        row_kernel_init = nnx.with_partitioning(kernel_init_base, ("model", None))

        self.num_heads = cfg.num_heads
        self.is_causal = cfg.causal
        self.qk_norm = cfg.qk_norm
        self.selective: bool = cfg.selective  # Mutable at runtime
        self.implementation: str = cfg.implementation  # Mutable at runtime

        linear_kwargs = {
            "use_bias": cfg.qkv_bias,
            "dtype": mp_policy.compute_dtype,
            "param_dtype": mp_policy.param_dtype,
            "bias_init": nnx.initializers.zeros_init(),
        }
        head_dim = (
            cfg.embed_dim // cfg.num_heads if cfg.embed_dim is not None else cfg.head_dim
        )  # ty: ignore
        self.qkv_proj = nnx.Linear(
            cfg.residual_dim, 3 * cfg.inner_dim, kernel_init=col_kernel_init, rngs=rngs, **linear_kwargs
        )
        self.o_proj = nnx.Linear(
            cfg.inner_dim, cfg.residual_dim, kernel_init=row_kernel_init, rngs=rngs, **linear_kwargs
        )
        self.drop_proj = nnx.Dropout(cfg.drop_proj, rngs=rngs)
        if self.qk_norm:
            self.q_norm = LayerNorm(
                head_dim,  # ty: ignore
                type=cfg.norm_type,
                use_bias=False,
                param_dtype=mp_policy.param_dtype,
                bias_init=nnx.initializers.zeros_init(),
                scale_init=nnx.initializers.ones_init(),
                rngs=rngs,
            )
            self.k_norm = LayerNorm(
                head_dim,  # ty: ignore
                type=cfg.norm_type,
                use_bias=False,
                param_dtype=mp_policy.param_dtype,
                bias_init=nnx.initializers.zeros_init(),
                scale_init=nnx.initializers.ones_init(),
                rngs=rngs,
            )

    def _get_attn_fn(self):
        """Get attention function with current implementation setting."""
        attn = partial(jax.nn.dot_product_attention, implementation=self.implementation)
        if self.is_causal:
            attn = partial(attn, is_causal=True)
        if self.selective:
            attn = jax.checkpoint(attn)
        return attn

    def _compute_attention_with_weights(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Compute attention manually to extract weights.

        Args:
            q, k, v: Query, key, value tensors of shape (B, T, num_heads, head_dim)
            mask: Optional attention mask

        Returns:
            output: Attention output of shape (B, T, num_heads, head_dim)
            weights: Attention weights of shape (B, num_heads, T, T)
        """
        head_dim = q.shape[-1]
        scale = head_dim**-0.5

        # (B, T, H, D) @ (B, T, H, D).T -> (B, H, T, T)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) * scale

        if self.is_causal:
            t_q, t_k = q.shape[1], k.shape[1]
            causal_mask = jnp.tril(jnp.ones((t_q, t_k), dtype=jnp.bool_))
            scores = jnp.where(causal_mask, scores, jnp.finfo(scores.dtype).min)

        if mask is not None:
            scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)

        weights = jax.nn.softmax(scores, axis=-1)
        # (B, H, T, T) @ (B, T, H, D) -> (B, T, H, D)
        output = jnp.einsum("bhts,bshd->bthd", weights, v)
        return output, weights

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        rope: Callable | None = None,
        return_attention_weights: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Forward pass of the attention layer.

        Args:
            x: Input tensor of shape (B, T, D)
            attention_mask: Optional attention mask
            rope: Optional rotary position embedding function
            return_attention_weights: If True, return (output, weights) tuple

        Returns:
            If return_attention_weights is False: output tensor of shape (B, T, D)
            If return_attention_weights is True: (output, weights) where weights
                has shape (B, num_heads, T, T)
        """
        qkv = self.qkv_proj(x)
        q, k, v = rearrange(qkv, "b t (three n c) -> three b t n c", three=3, n=self.num_heads)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q).astype(v.dtype)
            k = rope(k).astype(v.dtype)

        if return_attention_weights:
            # Manual computation to get weights (can't use cuDNN here)
            att, weights = self._compute_attention_with_weights(q, k, v, mask=attention_mask)
        else:
            attn_fn = self._get_attn_fn()
            att = attn_fn(q, k, v, mask=attention_mask)
            weights = None

        att = rearrange(att, "b t n c -> b t (n c)")
        att = self.o_proj(att)
        att = self.drop_proj(att)

        if return_attention_weights:
            return att, weights
        return att


class CrossAttention(Attention):
    def __init__(
        self,
        cfg: TransformerConfig,
        mp_policy: jmp.Policy,
        *,
        init_std: float = 0.02,
        i: int | None = None,
        rngs: nnx.Rngs,
    ):
        super(CrossAttention, self).__init__(cfg, mp_policy, init_std=init_std, i=i, rngs=rngs)
        del self.qkv_proj

        match cfg.linear_kernel_init:
            case "xavier_uniform":
                kernel_init_base = nnx.initializers.xavier_uniform()
            case "trunc_normal":
                # init_kernel_scale = init_std if i is None else init_std / (2 * (i + 1))
                kernel_init_base = nnx.initializers.truncated_normal(cfg.linear_init_std)
            case _:
                raise ValueError("invalid kernel_init")

        # FSDP Megatron pattern: q/k/v column-parallel (o_proj already row-parallel from parent)
        col_kernel_init = nnx.with_partitioning(kernel_init_base, (None, "model"))

        linear_kwargs = {
            "use_bias": cfg.qkv_bias,
            "dtype": mp_policy.compute_dtype,
            "param_dtype": mp_policy.param_dtype,
            "bias_init": nnx.initializers.zeros_init(),
            "kernel_init": col_kernel_init,
        }
        self.q_proj = nnx.Linear(cfg.residual_dim, cfg.inner_dim, rngs=rngs, **linear_kwargs)
        self.k_proj = nnx.Linear(cfg.residual_dim, cfg.inner_dim, rngs=rngs, **linear_kwargs)
        self.v_proj = nnx.Linear(cfg.residual_dim, cfg.inner_dim, rngs=rngs, **linear_kwargs)

    def __call__(
        self,
        latent: jax.Array,
        byte: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        rope: Callable | None = None,
    ) -> jax.Array:
        q, k, v = self.q_proj(latent), self.k_proj(byte), self.v_proj(byte)
        q = rearrange(q, "b t (n c) -> b t n c", n=self.num_heads)
        k = rearrange(k, "b t (n c) -> b t n c", n=self.num_heads)
        v = rearrange(v, "b t (n c) -> b t n c", n=self.num_heads)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            q = rope(q).astype(v.dtype)
            k = rope(k).astype(v.dtype)

        attn_fn = self._get_attn_fn()
        att = attn_fn(q, k, v, mask=attention_mask)
        att = rearrange(att, "b t n c -> b t (n c)")
        att = self.o_proj(att)
        att = self.drop_proj(att)
        return att


class TransformerDecoderLayer(nnx.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        mp_policy: jmp.Policy,
        *,
        i: int | None = None,
        drop_p: float = 0.0,
        rngs: nnx.Rngs,
    ):
        self.attention = Attention(cfg, mp_policy, i=i, rngs=rngs)
        self.mlp = MLP(cfg, mp_policy, i=i, rngs=rngs)
        self.drop_path = DropPath(drop_p, rngs=rngs)
        self.att_norm = LayerNorm(
            cfg.residual_dim,
            type=cfg.norm_type,
            param_dtype=mp_policy.param_dtype,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            epsilon=cfg.layer_norm_eps,
            rngs=rngs,
        )
        self.mlp_norm = LayerNorm(
            cfg.residual_dim,
            type=cfg.norm_type,
            param_dtype=mp_policy.param_dtype,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            epsilon=cfg.layer_norm_eps,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        rope: Callable | None = None,
        deterministic: bool | None = None,
        return_attention_weights: bool = False,
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Forward pass of the transformer decoder layer.

        Args:
            x: Input tensor of shape (B, T, D)
            attention_mask: Optional attention mask
            rope: Optional rotary position embedding function
            deterministic: If True, disable dropout
            return_attention_weights: If True, return (output, weights) tuple

        Returns:
            If return_attention_weights is False: output tensor of shape (B, T, D)
            If return_attention_weights is True: (output, weights) where weights
                has shape (B, num_heads, T, T)
        """
        attn_out = self.attention(
            self.att_norm(x),
            attention_mask,
            rope=rope,
            return_attention_weights=return_attention_weights,
        )
        if return_attention_weights:
            attn, weights = attn_out
        else:
            attn = attn_out
            weights = None

        x = x + self.drop_path(attn, deterministic=deterministic)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)), deterministic=deterministic)

        if return_attention_weights:
            return x, weights
        return x


class TransformerCrossAttention(nnx.Module):
    def __init__(
        self,
        cfg: TransformerConfig,
        mp_policy: jmp.Policy,
        *,
        i: int | None = None,
        drop_p: float = 0.0,
        rngs: nnx.Rngs,
    ):
        self.attention = CrossAttention(cfg, mp_policy, i=i, rngs=rngs)
        self.mlp = MLP(cfg, mp_policy, i=i, rngs=rngs)
        self.drop_path = DropPath(drop_p, rngs=rngs)
        self.att_norm = LayerNorm(
            cfg.residual_dim,
            type=cfg.norm_type,
            param_dtype=mp_policy.param_dtype,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            epsilon=cfg.layer_norm_eps,
            rngs=rngs,
        )
        self.mlp_norm = LayerNorm(
            cfg.residual_dim,
            type=cfg.norm_type,
            param_dtype=mp_policy.param_dtype,
            scale_init=nnx.initializers.ones_init(),
            bias_init=nnx.initializers.zeros_init(),
            epsilon=cfg.layer_norm_eps,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        attention_mask: jax.Array | None = None,
        *,
        rope: Callable | None = None,
        deterministic: bool | None = None,
    ):
        attn = self.attention(self.att_norm(x), attention_mask, rope=rope)
        x = x + self.drop_path(attn, deterministic=deterministic)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)), deterministic=deterministic)
        return x


def set_attn_implementation(module: nnx.Module, implementation: Literal["cudnn", "xla"]) -> None:
    """Set attention implementation on all attention modules in a model.

    This allows switching between cuDNN Flash Attention and XLA attention
    at runtime without modifying the model config or reloading checkpoints.

    Works with any module that has an `implementation` attribute (duck typing),
    including Attention, CrossAttention, and CrossAttentionWithRoPE.

    Args:
        module: The model or module to update
        implementation: "cudnn" for Flash Attention, "xla" for default XLA
    """
    for _, submodule in nnx.iter_modules(module):
        if hasattr(submodule, "implementation") and hasattr(submodule, "_get_attn_fn"):
            submodule.implementation = implementation


def set_gradient_checkpointing(module: nnx.Module, selective: bool) -> None:
    """Set gradient checkpointing on all attention modules in a model.

    This allows enabling/disabling gradient checkpointing (memory vs compute tradeoff)
    at runtime without modifying the model config or reloading checkpoints.

    Works with any module that has a `selective` attribute (duck typing),
    including Attention, CrossAttention, and CrossAttentionWithRoPE.

    Args:
        module: The model or module to update
        selective: True to enable gradient checkpointing, False to disable
    """
    for _, submodule in nnx.iter_modules(module):
        if hasattr(submodule, "selective") and hasattr(submodule, "_get_attn_fn"):
            submodule.selective = selective
