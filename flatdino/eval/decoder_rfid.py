from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true"
)

import jax
import jax.numpy as jnp
import jmp
import orbax.checkpoint as ocp
from dacite import Config as DaciteConfig, from_dict
from absl import logging
import tyro

from flatdino.data import create_dataloaders
from flatdino.eval import restore_encoder
from flatdino.train_decoder import GeneratorConfig, compute_recon_fid
from flatdino.decoder.augmentations import RAEDecoderValAugmentations
from flatdino.decoder.rae_decoder import RAEDecoder

jax.config.update("jax_optimization_level", "O1")
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"
)


@dataclass
class Args:
    decoder_path: Path
    """Directory containing checkpoints produced by train_decoder.py."""

    flatdino_path: Path
    """Directory containing the FlatDINO checkpoint (encoder weights are required)."""

    batch_size: int = 256
    """Batch size for FID statistics and reconstruction."""

    num_eval_images: int | None = None
    """Optional cap on the number of validation images to use."""

    debug: bool = False


def _restore_decoder(
    path: Path,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
) -> tuple[RAEDecoder, GeneratorConfig]:
    item_names = ["config", "decoder"]
    opts = ocp.CheckpointManagerOptions(read_only=True)
    mngr = ocp.CheckpointManager(path.absolute(), options=opts, item_names=item_names)

    step = mngr.best_step()
    if step is None:
        step = mngr.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found in {path}.")
    logging.info(f"Found decoder checkpoint at step {step}. Restoring...")

    dacite_cfg = DaciteConfig(cast=[tuple])
    cfg_blob = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))
    cfg_dict = cfg_blob["config"]
    cfg = from_dict(GeneratorConfig, cfg_dict, config=dacite_cfg)

    decoder = RAEDecoder.restore(
        mngr,
        step,
        "decoder",
        mesh,
        cfg=cfg.vit,
        patch_size=cfg.patch_size,
        num_channels=cfg.out_channels,
        mp=mp,
    )
    mngr.close()
    return decoder, cfg


def _create_val_loader(cfg: GeneratorConfig, batch_size: int):
    loaders = create_dataloaders(
        cfg.data,
        batch_size,
        train_epochs=1,
        val_epochs=1,
        val_aug=RAEDecoderValAugmentations(cfg.aug, cfg.data),
        drop_remainder_val=False,
        val_shuffle=True,
    )
    return loaders.val_loader


def main(args: Args):
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not args.decoder_path.exists():
        raise FileNotFoundError(f"Decoder checkpoint directory not found: {args.decoder_path}")
    if not args.flatdino_path.exists():
        raise FileNotFoundError(f"FlatDINO checkpoint directory not found: {args.flatdino_path}")

    if args.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().use_absl_log_file()

    device_count = jax.device_count()
    if args.batch_size % device_count != 0:
        raise ValueError("batch_size must be divisible by the number of devices.")
    mesh = jax.make_mesh((device_count, 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.float32, output_dtype=jnp.float32)

    decoder, cfg = _restore_decoder(args.decoder_path, mesh=mesh, mp=mp)
    decoder.eval()

    restored = restore_encoder(args.flatdino_path, mesh=mesh, mp=mp, encoder=True, decoder=False)
    if restored.encoder is None:
        raise RuntimeError("FlatDINO encoder weights are required for rFID computation.")
    restored.dino.eval()
    restored.encoder.eval()

    val_loader = _create_val_loader(cfg, args.batch_size)
    fid_value = compute_recon_fid(
        decoder,
        restored.dino,
        restored.encoder,
        cfg,
        mesh,
        val_loader=val_loader,
        batch_size=args.batch_size,
        num_eval_images=args.num_eval_images,
        verbose=args.debug,
        noise_tau=0.0,
    )
    logging.info("Reconstruction FID (Inception V3): %.4f", float(fid_value))


if __name__ == "__main__":
    main(tyro.cli(Args))
