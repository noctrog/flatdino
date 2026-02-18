from dataclasses import dataclass, field
import math
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import matplotlib.pyplot as plt
import numpy as np
import tyro
from tqdm import tqdm

from flatdino.data import DataConfig, create_dataloaders
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.eval import extract_mu_logvar, restore_encoder
from flatdino.distributed import prefetch_to_mesh


@dataclass
class Config:
    path: Path
    gpu_batch_size: int = 128
    bins: int = 50
    max_batches: int | None = None
    output: Path | None = None
    data: DataConfig = field(default_factory=lambda: DataConfig())


def main(cfg: Config):
    if cfg.gpu_batch_size <= 0:
        raise ValueError("gpu_batch_size must be positive.")
    if cfg.bins <= 0:
        raise ValueError("bins must be positive.")

    device_count = jax.device_count()
    if device_count == 0:
        raise RuntimeError("No visible JAX devices.")

    mesh = jax.make_mesh((device_count, 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    restored = restore_encoder(cfg.path, mesh=mesh, mp=mp, encoder=True)
    assert restored.encoder is not None
    assert restored.aug_cfg is not None
    assert restored.data_cfg is not None
    restored.dino.eval(), restored.encoder.eval()

    batch_size = cfg.gpu_batch_size * device_count
    data = create_dataloaders(
        cfg.data,
        batch_size=batch_size,
        val_epochs=1,
        val_aug=FlatDinoValAugmentations(restored.aug_cfg, restored.data_cfg),
        drop_remainder_val=False,
    )
    total_batches = math.ceil(data.val_ds_size / batch_size)
    if cfg.max_batches is not None:
        total_batches = min(total_batches, cfg.max_batches)

    @nnx.jit
    def _mu_std_norm(imgs: jax.Array) -> tuple[jax.Array, jax.Array]:
        patches = restored.dino(imgs)[:, 5:]
        enc_out = restored.encoder(patches, deterministic=True)
        mu, logvar = extract_mu_logvar(
            enc_out, restored.num_flat_tokens, restored.encoder_disposable_registers
        )
        std = jnp.exp(0.5 * logvar)
        mu_norm = jnp.linalg.norm(mu, axis=-1)
        std_norm = jnp.linalg.norm(std, axis=-1)
        return mu_norm, std_norm

    mu_values: list[np.ndarray] = []
    std_values: list[np.ndarray] = []
    processed = 0
    for batch in tqdm(
        prefetch_to_mesh(iter(data.val_loader), 1, mesh, trim=True),
        total=total_batches,
        desc="val mu/std",
        leave=False,
    ):
        if cfg.max_batches is not None and processed >= cfg.max_batches:
            break
        imgs = batch["image"]
        mu_norm, std_norm = jax.device_get(_mu_std_norm(imgs))
        mu_values.append(np.asarray(mu_norm, dtype=np.float32))
        std_values.append(np.asarray(std_norm, dtype=np.float32))
        processed += 1

    if not mu_values or not std_values:
        raise RuntimeError("Validation loader produced no batches.")

    mu_flat = np.concatenate(mu_values, axis=0).reshape(-1)
    std_flat = np.concatenate(std_values, axis=0).reshape(-1)

    output_path = cfg.output or (cfg.path / "mu_std_hist.png")
    png_path = output_path if output_path.suffix else output_path.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = png_path.with_suffix(".pdf")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    axes[0].hist(mu_flat, bins=cfg.bins, color="#1f77b4", alpha=0.85)
    axes[0].set_title("FlatDINO encoder mu magnitude (ImageNet val)")
    axes[0].set_xlabel("||mu||_2")
    axes[0].set_ylabel("Count")

    axes[1].hist(std_flat, bins=cfg.bins, color="#ff7f0e", alpha=0.85)
    axes[1].set_title("FlatDINO encoder std magnitude (ImageNet val)")
    axes[1].set_xlabel("||std||_2")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Saved histograms to {png_path} and {pdf_path}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
