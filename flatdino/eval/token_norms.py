from dataclasses import dataclass, field
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
import jmp
import tyro
from tqdm import tqdm
import matplotlib.pyplot as plt

from flatdino.data import DataConfig, create_dataloaders
from flatdino.distributed import prefetch_to_mesh
from flatdino.eval import extract_mu, restore_encoder
from flatdino.augmentations import FlatDinoValAugmentations


@dataclass
class Config:
    path: Path
    gpu_batch_size: int = 128
    output_stem: str = "flatdino_token_norms"
    data: DataConfig = field(default_factory=lambda: DataConfig())


def main(cfg: Config):
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    r = restore_encoder(cfg.path, mesh=mesh, mp=mp, encoder=True)
    r.dino.eval(), r.encoder.eval()

    batch_size = cfg.gpu_batch_size * jax.device_count()
    data = create_dataloaders(
        cfg.data,
        batch_size=batch_size,
        val_epochs=1,
        val_aug=FlatDinoValAugmentations(r.aug_cfg, r.data_cfg),
    )
    val_iters = data.val_ds_size // batch_size

    @nnx.jit
    def iter_step(imgs: jax.Array) -> jax.Array:
        patches = r.dino(imgs)[:, 5:]
        enc_out = r.encoder(patches, deterministic=True)
        mu = extract_mu(enc_out, r.num_flat_tokens, r.encoder_disposable_registers)
        return jnp.linalg.norm(mu, axis=-1, keepdims=False)

    all_norms: list[np.ndarray] = []
    for batch in tqdm(
        prefetch_to_mesh(iter(data.val_loader), 1, mesh, trim=True),
        desc="val",
        total=val_iters,
        leave=False,
    ):
        imgs = batch["image"]
        norms = jax.device_get(iter_step(imgs))
        norms = np.asarray(norms)
        all_norms.append(norms)

    if not all_norms:
        raise RuntimeError("Validation loader produced no batches.")

    token_norms = np.concatenate(all_norms, axis=0)
    token_ids = np.arange(token_norms.shape[1])

    token_mean = np.mean(token_norms, axis=0)
    token_var = np.var(token_norms, axis=0)
    token_std = np.std(token_norms, axis=0)

    output_dir = cfg.path if cfg.path.is_dir() else cfg.path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{cfg.output_stem}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["token_id", "mu", "var", "std"])
        for tid, mu, var, std in zip(token_ids, token_mean, token_var, token_std):
            writer.writerow([int(tid), float(mu), float(var), float(std)])

    fig, ax = plt.subplots(figsize=(12, 6))
    parts = ax.violinplot(token_norms, positions=token_ids, showmedians=True)
    for body in parts["bodies"]:
        body.set_alpha(0.4)

    ax.set_xticks(token_ids)
    ax.set_xticklabels([f"T{i}" for i in token_ids])
    ax.set_xlabel("Token index")
    ax.set_ylabel("Embedding L2 norm")
    ax.set_title("Validation token norm distribution")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{cfg.output_stem}.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
