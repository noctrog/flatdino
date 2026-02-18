"""kNN evaluation for FlatDINO encoder.

Evaluates the quality of FlatDINO latent representations using k-nearest neighbors
classification on various datasets.

Example usage:
    # ImageNet (default)
    python -m flatdino.eval.knn --path output/flatdino/vae/med-32-bl-128

    # CIFAR-10
    python -m flatdino.eval.knn --path output/flatdino/vae/med-32-bl-128 --dataset cifar10

    # Multiple datasets
    for ds in cifar10 cifar100 flowers102 pets food101 dtd; do
        python -m flatdino.eval.knn --path output/flatdino/vae/med-32-bl-128 --dataset $ds
    done

Available datasets: imagenet, cifar10, cifar100, caltech101, flowers102, pets, food101, dtd
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import jmp
import tyro

from flatdino.metrics import KNNConfig, knn
from flatdino.data import create_dataloaders, DataConfig
from flatdino.eval import restore_encoder, save_eval_results
from flatdino.augmentations import FlatDinoValAugmentations

DatasetName = Literal["imagenet", "cifar10", "cifar100", "caltech101", "flowers102", "pets", "food101", "dtd"]


@dataclass
class Config:
    path: Path
    """Path to FlatDINO checkpoint."""
    dataset: DatasetName = "imagenet"
    """Dataset to evaluate on."""
    knn: KNNConfig = field(default_factory=lambda: KNNConfig())
    """kNN evaluation configuration."""
    gpu_collect_batch_size: int = 128
    """Batch size per GPU for feature collection."""
    num_workers: int = 32
    """Number of data loading workers."""


def main(cfg: Config):
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    r = restore_encoder(cfg.path, mesh=mesh, mp=mp, encoder=True, decoder=False)
    assert r.encoder is not None
    assert r.aug_cfg is not None
    assert r.data_cfg is not None
    r.dino.eval()
    r.encoder.eval()

    # Create dataset config from preset
    data_cfg = DataConfig.from_preset(cfg.dataset, num_workers=cfg.num_workers)
    print(f"Evaluating on {cfg.dataset} ({data_cfg.dataset}, {data_cfg.num_classes} classes)")

    num_flat_tokens = r.num_flat_tokens

    @nnx.jit
    def feat_fn(img: jax.Array):
        # Get DINO patches (remove CLS + 4 register tokens)
        patches = r.dino(img)[:, 5:]

        # Encode with FlatDINO
        encoded = r.encoder(patches, deterministic=True)

        # Extract mu (first half of last dim) from register tokens
        latent_dim = encoded.shape[-1] // 2
        tokens = encoded[:, :num_flat_tokens, :latent_dim]  # (B, num_tokens, latent_dim)

        # L2 normalize, mean pool, L2 normalize
        tokens = tokens / (1e-5 + jnp.linalg.norm(tokens, axis=-1, keepdims=True))
        tokens = jnp.mean(tokens, axis=1, keepdims=False)  # (B, latent_dim)
        tokens = tokens / (1e-5 + jnp.linalg.norm(tokens, axis=-1, keepdims=True))
        return tokens

    collect_batch_size = cfg.gpu_collect_batch_size * jax.device_count()
    data = create_dataloaders(
        data_cfg,
        collect_batch_size,
        train_epochs=1,
        val_epochs=1,
        train_aug=FlatDinoValAugmentations(r.aug_cfg, r.data_cfg),
        val_aug=FlatDinoValAugmentations(r.aug_cfg, r.data_cfg),
        drop_remainder_train=False,
        drop_remainder_val=False,
    )

    df = knn(cfg.knn, collect_batch_size, feat_fn, data, mesh=mesh, num_classes=data_cfg.num_classes)

    # Save results to JSON
    results = {f"k_{row['k']}": {"top_1": row["top_1"], "top_5": row["top_5"]} for _, row in df.iterrows()}
    results["best_top_1"] = df["top_1"].max()
    save_eval_results(cfg.path, f"knn_{cfg.dataset}", results)
    print(f"Saved kNN results to {cfg.path}/eval_results.json")


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
