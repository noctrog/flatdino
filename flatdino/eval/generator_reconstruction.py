from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
import tyro
from einops import rearrange

from flatdino.eval import restore_encoder
from flatdino.decoder.sampling import (
    _infer_time_dist_shift,
    _restore_generator,
    _sample_latents,
)
from flatdino.pretrained.rae_decoder import make_rae_decoder


@dataclass
class Config:
    generator_path: Path
    """Directory containing checkpoints produced by train_generator.py."""

    flatdino_path: Path
    """Directory containing the FlatDINO autoencoder checkpoint."""

    rows: int = 4
    cols: int = 4
    steps: int = 50
    output: Path = Path("/tmp/flatdino_generator_grid.png")
    seed: int = 0
    class_id: int | None = None
    use_ema: bool = True
    cfg_scale: float | None = None
    """Classifier-free guidance scale. None disables CFG."""


def _prepare_labels(
    *,
    num_samples: int,
    num_classes: int,
    class_id: int | None,
    key: jax.Array,
) -> jax.Array:
    if num_classes <= 0:
        raise ValueError("The generator configuration reports zero classes.")
    if class_id is not None:
        if class_id < 0 or class_id >= num_classes:
            raise ValueError(f"class_id {class_id} is outside [0, {num_classes}).")
        labels = jnp.full((num_samples,), class_id, dtype=jnp.int32)
    else:
        labels = jax.random.randint(key, (num_samples,), 0, num_classes, dtype=jnp.int32)
    return labels


def _save_grid(images: np.ndarray, rows: int, cols: int, path: Path) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.imshow(images[idx])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def main(cfg: Config):
    if cfg.rows <= 0 or cfg.cols <= 0:
        raise ValueError("rows and cols must be positive.")
    if cfg.steps <= 0:
        raise ValueError("steps must be positive.")

    num_samples = cfg.rows * cfg.cols

    device_count = jax.device_count()
    if device_count != 1:
        raise RuntimeError("This script currently expects a single accelerator.")

    mesh = jax.make_mesh((device_count, 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    restored = restore_encoder(cfg.flatdino_path, mesh=mesh, mp=mp, encoder=True, decoder=True)
    if restored.encoder is None:
        raise RuntimeError("FlatDINO checkpoint does not include encoder weights.")
    if restored.decoder is None:
        raise RuntimeError("FlatDINO checkpoint does not include decoder weights.")
    if restored.data_cfg is None or restored.aug_cfg is None:
        raise RuntimeError("FlatDINO checkpoint is missing data or augmentation configuration.")

    latent_tokens = restored.encoder.num_reg
    if latent_tokens is None or latent_tokens <= 0:
        raise ValueError("Encoder configuration does not define a positive number of registers.")

    patch_tokens = restored.decoder.num_reg
    if patch_tokens is None or patch_tokens <= 0:
        raise ValueError("Decoder configuration does not define a positive number of patch tokens.")

    image_h, image_w = restored.aug_cfg.image_size
    mean_vals = restored.data_cfg.normalization_mean
    std_vals = restored.data_cfg.normalization_std

    generator, generator_cfg = _restore_generator(
        cfg.generator_path,
        mesh=mesh,
        mp=mp,
        latent_tokens=latent_tokens,
        use_ema=cfg.use_ema,
    )

    generator.eval()
    restored.decoder.eval()

    latent_dim = generator.cfg.in_channels
    latent_shape = (latent_tokens, latent_dim)

    if image_h != image_w:
        raise ValueError("Only square image sizes are supported.")

    rae_decoder = make_rae_decoder(
        num_patches=patch_tokens,
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=cfg.seed,
    )

    key = jax.random.PRNGKey(cfg.seed)
    key, label_key, sample_key = jax.random.split(key, 3)

    labels = _prepare_labels(
        num_samples=num_samples,
        num_classes=generator.cfg.num_classes,
        class_id=cfg.class_id,
        key=label_key,
    )

    time_dist_shift = _infer_time_dist_shift(
        generator_cfg,
        latent_tokens=latent_tokens,
        latent_dim=generator.cfg.in_channels,
    )

    latents = _sample_latents(
        generator,
        labels=labels,
        steps=cfg.steps,
        key=sample_key,
        latent_shape=latent_shape,
        cfg_scale=cfg.cfg_scale,
        time_dist_shift=time_dist_shift,
        pred_type=generator_cfg["pred_type"] if "pred_type" in generator_cfg else "v",
    )
    if isinstance(latents, dict):
        latents = latents["tokens"]

    decoder_input = latents if latents.ndim == 3 else rearrange(latents, "b h w c -> b (h w) c")
    decoder_tokens = restored.decoder(decoder_input, deterministic=True)[:, :patch_tokens]

    decoder_out = rae_decoder(decoder_tokens)
    recon = rae_decoder.unpatchify(
        decoder_out.logits,
        original_image_size=(image_h, image_w),
    )
    recon = jnp.transpose(recon, (0, 2, 3, 1))

    mean = jnp.array(mean_vals, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(std_vals, dtype=jnp.float32)[None, None, None, :]
    recon = jnp.clip(recon * std + mean, 0.0, 1.0)

    recon_np = np.asarray(jax.device_get(recon))
    _save_grid(recon_np, cfg.rows, cfg.cols, cfg.output)
    print(f"Saved generator samples to {cfg.output}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
