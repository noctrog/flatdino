from typing import Literal
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import tyro
import matplotlib.pyplot as plt

from flatdino.data import create_dataloaders, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
try:
    from ssljax.data.pusht import PushTDataset, TrajSlicerDataset
except ImportError:
    PushTDataset = TrajSlicerDataset = None
from flatdino.metrics.inversion import TokenInvConfig, invert_token
from flatdino.eval import restore_encoder
from flatdino.augmentations import FlatDinoValAugmentations


@dataclass
class Config:
    path: Path
    output: Path = Path("/tmp/inverted_flatdino")
    save_inverted_image_only: bool = False
    inversion: TokenInvConfig = field(default_factory=lambda: TokenInvConfig())
    data: Literal["imagenet", "pusht"] = "imagenet"
    invert: Literal["encoder", "decoder"] = "encoder"
    token_id: int | None = None
    """If specified, invert only the specific token id. Only works when `invert` is 'encoder'."""


def main(cfg: Config):
    assert jax.device_count() == 1, "This script should be run on a single GPU."
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    r = restore_encoder(cfg.path, mesh=mesh, mp=mp, encoder=True, decoder=cfg.invert == "decoder")
    r.dino.eval(), r.encoder.eval()
    if cfg.invert == "decoder":
        r.decoder.eval()

    mu = jnp.array(IMAGENET_DEFAULT_MEAN)[None, None, None, :]
    std = jnp.array(IMAGENET_DEFAULT_STD)[None, None, None, :]

    match cfg.data:
        case "imagenet":
            data = create_dataloaders(
                r.data_cfg, batch_size=1, val_aug=FlatDinoValAugmentations(r.aug_cfg, r.data_cfg)
            )
            img = jax.device_put(next(iter(data.val_loader))["image"])
        case "pusht":
            ds = TrajSlicerDataset(PushTDataset("train"), num_frames=4, frameskip=4)
            img = (jax.device_put(ds[0].obs_visual[[0]]) - mu) / std
        case _:
            raise ValueError("invalid dataset")

    match cfg.invert:
        case "encoder":
            target_token = r.encoder(r.dino(img)[:, 5:])[0]

            if cfg.token_id is not None:
                target_token = target_token[:, cfg.token_id]

            def token_fn(img: jax.Array):
                img = (img - mu) / std
                patches = r.dino(img)[:, 5:]
                tokens = r.encoder(patches)[0]
                if cfg.token_id is not None:
                    tokens = tokens[:, cfg.token_id]
                return tokens
        case "decoder":
            if cfg.token_id is not None:
                print(
                    "WARNING: token id specified, but we are inverting the 2D patch grid. Ignoring."
                )
            target_token = r.decoder(r.encoder(r.dino(img)[:, 5:])[0])

            def token_fn(img: jax.Array):
                img = (img - mu) / std
                patches = r.dino(img)[:, 5:]
                embeddings = r.encoder(patches)[0]
                return r.decoder(embeddings)

    inverted_img = invert_token(cfg.inversion, token_fn, target_token, h=224, w=224)

    original_img = jnp.clip(img * std + mu, 0.0, 1.0)
    reconstructed_img = jnp.clip(inverted_img, 0.0, 1.0)

    original_np = jax.device_get(original_img[0])
    reconstructed_np = jax.device_get(reconstructed_img[0])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(original_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(reconstructed_np)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    output_base = f"{cfg.output}_{cfg.data}_{cfg.invert}"
    output_path = Path(f"{output_base}.png")
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Comparison figure saved at {output_path}")

    if cfg.save_inverted_image_only:
        image_output_path = Path(f"{output_base}_inverted.png")
        plt.imsave(image_output_path, reconstructed_np)
        print(f"Reconstructed image saved at {image_output_path}")


if __name__ == "__main__":
    cfg: Config = tyro.cli(Config)
    main(cfg)
