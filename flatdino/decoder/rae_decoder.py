import jax
import jax.numpy as jnp
import flax.nnx as nnx
import jmp
from einops import rearrange

from flatdino.data.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flatdino.models.vit import ViTEncoder, ViTConfig


class RAEDecoder(ViTEncoder):
    def __init__(
        self, cfg: ViTConfig, patch_size: int, num_channels: int, mp: jmp.Policy, *, rngs: nnx.Rngs
    ):
        super(RAEDecoder, self).__init__(cfg, mp, rngs=rngs)
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.decoder_proj = nnx.Linear(
            cfg.transformer.residual_dim,
            patch_size**2 * num_channels,
            use_bias=True,
            param_dtype=mp.param_dtype,
            bias_init=nnx.initializers.zeros_init(),
            kernel_init=nnx.initializers.truncated_normal(0.02),
            rngs=rngs,
        )

        self.img_mean = nnx.Variable(rearrange(jnp.asarray(IMAGENET_DEFAULT_MEAN), "c -> 1 1 1 c"))
        self.img_std = nnx.Variable(rearrange(jnp.asarray(IMAGENET_DEFAULT_STD), "c -> 1 1 1 c"))

    def __call__(
        self,
        x: jax.Array | list[jax.Array],
        *,
        deterministic: bool | None = None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        x = super(RAEDecoder, self).__call__(x, deterministic=deterministic, mask=mask)
        x = self.mp.cast_to_compute(x)
        x = self.decoder_proj(x)
        x = x[:, 1:] if self.use_cls else x
        x = self.mp.cast_to_output(x)
        return x

    # TODO: can I just do this in the __call__?
    def unpatchify(self, x: jax.Array, denorm_output: bool = True) -> jax.Array:
        b, t, _ = x.shape
        h = w = int(t**0.5)
        if h * w != t:
            raise ValueError(f"Token count {t} is not a perfect square.")
        recon = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b (h p1) (w p2) c",
            h=h,
            w=w,
            p1=self.patch_size,
            p2=self.patch_size,
            c=self.num_channels,
        )

        if denorm_output:
            recon = recon * self.img_std.value + self.img_mean.value

        return recon
