from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
import math
import os
from typing import Literal

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true"
)

import jax
import jax.numpy as jnp
import chex
import numpy as np
import jmp
import flax.nnx as nnx
import grain.python as grain
import optax
import orbax.checkpoint as ocp
import tyro
import wandb
from tqdm import tqdm
from einops import rearrange
from absl import logging
from lpips_nnx import LPIPS
from dacite import from_dict, Config as DaciteConfig

from flatdino.metrics import fid as fid_eval
from flatdino.distributed import (
    prefetch_to_mesh,
    is_primary_host,
    init_distributed,
    determine_save_path,
    open_restore_manager,
    restore_data_loader,
    restore_model_state,
    restore_optimizer_state,
    TrainingProfiler,
)
from flatdino.data import DataConfig, DataLoaders, create_dataloaders
from flatdino.data.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flatdino.models.transformer import TransformerConfig, set_attn_implementation
from flatdino.models.vit import ViTConfig, VIT_CONFIGS
from flatdino.decoder.augmentations import (
    RAEDecoderAugConfig,
    RAEDecoderTrainAugmentations,
    RAEDecoderValAugmentations,
)
from flatdino.eval import restore_encoder
from flatdino.utils import build_lr_schedule, extract_mu_logvar
from flatdino.decoder.rae_decoder import RAEDecoder
from flatdino.decoder.diffaug import DiffAug, DiffAugConfig
from flatdino.decoder.discriminator import DinoDisc

jax.config.update("jax_optimization_level", "O1")

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

ITEM_NAMES = ["decoder", "decoder_ema", "discriminator", "optim_dec", "optim_disc", "loader", "config"]


def save_checkpoint(
    manager: ocp.CheckpointManager | None,
    step: int,
    decoder: RAEDecoder,
    decoder_ema: RAEDecoder,
    discriminator: DinoDisc,
    optim_dec: nnx.Optimizer,
    optim_disc: nnx.Optimizer,
    data_iter,
    cfg: "GeneratorConfig",
) -> bool:
    """Save checkpoint if the manager decides it should be saved.

    Returns True if a checkpoint was actually saved.
    """
    if manager is None:
        return False
    saved = manager.save(
        step,
        args=ocp.args.Composite(
            decoder=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(decoder))),
            decoder_ema=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(decoder_ema))),
            discriminator=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(discriminator))),
            optim_dec=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim_dec))),
            optim_disc=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim_disc))),
            loader=grain.PyGrainCheckpointSave(data_iter),
            config=ocp.args.JsonSave(asdict(cfg)),
        ),
    )
    return saved


@dataclass
class Args:
    flatdino_path: Path
    """Path to the FlatDINO VAE checkpoint. When gcs_bucket is set, this path is
    automatically prefixed with gs://{gcs_bucket}/."""
    seed: int = 42
    restore: Path | Literal["default"] | None = None
    """Path to restore checkpoint from. Use 'default' to restore from the default
    checkpoint path (output/flatdino/decoder/{experiment} or GCS equivalent if gcs_bucket is set)."""
    maybe_restore: Path | Literal["default"] | None = None
    """Like --restore, but if no checkpoint exists at the path, logs a warning and trains
    from scratch instead of crashing. Useful for resumable training scripts."""
    checkpoint: bool = True
    checkpoint_dir: Path | None = None
    gcs_bucket: str | None = None
    """GCS bucket name for checkpoints and flatdino_path. When set, checkpoints are saved
    to gs://{gcs_bucket}/output/... and flatdino_path is loaded from gs://{gcs_bucket}/...
    using orbax's native GCS support. Requires running on a GCP instance."""
    data_in_bucket: bool = False
    """When True and gcs_bucket is set, load datasets from gs://{gcs_bucket}/{dataset_name}
    via tfds instead of from local storage."""
    num_data_workers: int | None = None
    """Override the number of data loading workers. If None, uses the value from DataConfig."""
    keep_checkpoints_without_metrics: bool = True
    """If True, preemption checkpoints (which have no validation metrics) are preserved
    indefinitely. If False, they are cleaned up once we have enough checkpoints with metrics."""
    use_wandb: bool = False
    wandb_log_every: int = 20
    project_name: str = "flatdino-decoder"
    gpu_batch_size: int = 64
    profile_mode: Literal["disabled", "always", "window"] = "disabled"
    """Profiler mode: 'disabled' (no profiling), 'always' (start immediately and run forever),
    or 'window' (run between profiler_start_step and profiler_stop_step)."""
    profiler_port: int = 7777
    profiler_start_step: int = 10
    """Step to start profiling (only used in 'window' mode)."""
    profiler_stop_step: int = 2000
    """Step to stop profiling (only used in 'window' mode)."""
    val_end: bool = True
    experiment: str = "vit-s"
    debug: bool = False
    rfid_batch_size: int | None = None
    implementation: Literal["cudnn", "xla"] = "xla"
    """Attention implementation: 'cudnn' for Flash Attention, 'xla' for default."""
    fsdp: int = 1
    """FSDP (Fully Sharded Data Parallelism) axis size. When fsdp=1, only data parallelism
    is used. When fsdp>1, model parameters are sharded across fsdp devices using the
    Megatron-LM pattern. The data axis size is num_devices // fsdp."""
    distributed: bool = False
    """Enable distributed training for multi-host setups (e.g., TPU pods). Must be set
    before any JAX operations occur."""


@dataclass
class OptimConfig:
    epochs: int = 16
    batch_size: int = 512

    adam_b1: float = 0.9
    adam_b2: float = 0.95
    lr_start: float = 0.0
    lr_peak: float = 2e-4
    lr_final: float = 2e-5
    warmup_epochs: int = 1
    linear_end_epochs: int = 0
    weight_decay: float = 0.0

    ema_decay: float = 0.9978

    lpips_start: int = 0
    disc_start: int = 6
    disc_gan_start: int = 8
    max_d_weight: float = 1e4

    lpips_weight: float = 1.0
    gan_weight: float = 0.75
    lr_schedule: Literal["warmup_cosine"] = "warmup_cosine"


@dataclass
class GeneratorConfig:
    train: OptimConfig = field(default_factory=lambda: OptimConfig())
    vit: ViTConfig = field(
        default_factory=lambda: ViTConfig(
            patch=None,
            num_patches=32,
            num_registers=256,
            transformer=TransformerConfig(**VIT_CONFIGS["vit-s"]),
        )
    )
    data: DataConfig = field(default_factory=lambda: DataConfig())
    aug: RAEDecoderAugConfig = field(default_factory=lambda: RAEDecoderAugConfig())
    diffaug: DiffAugConfig = field(default_factory=lambda: DiffAugConfig(prob=1.0, cutout=0.0))
    patch_size: int = 16
    out_channels: int = 3
    noise_tau: float = 0.2


def _decoder_proj_kernel(tree: RAEDecoder) -> jax.Array:
    params = nnx.state(tree, nnx.Param)
    return params["decoder_proj"]["kernel"]


def generator_loss_fn(
    key: jax.Array,
    decoder: RAEDecoder,
    imgs: jax.Array,
    latents: jax.Array,
    use_lpips: bool,
    lpips: LPIPS,
    use_gan: bool,
    diffaug: DiffAug,
    discriminator: DinoDisc,
    *,
    lpips_weight: float,
    gan_weight: float,
    max_d_weight: float,
) -> tuple[jax.Array, RAEDecoder, dict[str, jax.Array]]:
    """Reconstruction + GAN loss with adaptive weighting on the last decoder layer."""
    img_mean = rearrange(jnp.asarray(IMAGENET_DEFAULT_MEAN, dtype=imgs.dtype), "c -> 1 1 1 c")
    img_std = rearrange(jnp.asarray(IMAGENET_DEFAULT_STD, dtype=imgs.dtype), "c -> 1 1 1 c")

    def _forward(model: RAEDecoder, disc: DinoDisc):
        recon_tokens = model(latents)
        recon_tokens = recon_tokens[:, : model.num_reg]
        recon = model.unpatchify(recon_tokens, denorm_output=False)
        recon_pix = recon * img_std + img_mean
        recon_normed = jnp.clip(2 * recon_pix - 1, -1.0, 1.0)

        imgs_pix = imgs * img_std + img_mean
        rec_loss = jnp.mean(jnp.abs(recon_pix - imgs_pix))

        if use_lpips:
            lpips_imgs = jnp.clip(imgs_pix, 0.0, 1.0)
            lpips_imgs = 2 * lpips_imgs - 1
            lpips_loss = jnp.mean(lpips(recon_normed, lpips_imgs))
        else:
            lpips_loss = jnp.asarray(0.0, dtype=recon.dtype)

        if use_gan:
            fake_key, _ = jax.random.split(key)
            fake_aug = diffaug(fake_key, recon_normed)
            logits_fake = disc(fake_aug)
            gan_loss = -jnp.mean(logits_fake)
        else:
            gan_loss = jnp.asarray(0.0, dtype=recon.dtype)

        recon_total = rec_loss + lpips_weight * lpips_loss
        return recon_total, gan_loss, rec_loss, lpips_loss

    (recon_total, gan_loss, rec_loss, lpips_loss), pullback = jax.vjp(
        _forward, decoder, discriminator
    )

    (recon_grads, _) = pullback((1.0, 0.0, 0.0, 0.0))
    if use_gan:
        (gan_grads, _) = pullback((0.0, 1.0, 0.0, 0.0))

        recon_norm = jnp.linalg.norm(_decoder_proj_kernel(recon_grads))
        gan_norm = jnp.linalg.norm(_decoder_proj_kernel(gan_grads))
        adaptive_w = jnp.clip(recon_norm / (gan_norm + 1e-6), 0.0, max_d_weight)
    else:
        gan_grads = jax.tree.map(jnp.zeros_like, recon_grads)
        adaptive_w = jnp.asarray(0.0, dtype=recon_total.dtype)

    adaptive_w = jax.lax.stop_gradient(adaptive_w)
    total_loss = recon_total + gan_weight * adaptive_w * gan_loss

    combined_grads = jax.tree.map(
        lambda r, g: r + gan_weight * adaptive_w * g,
        nnx.state(recon_grads, nnx.Param),
        nnx.state(gan_grads, nnx.Param),
    )

    metrics = {
        "recon_loss": rec_loss,
        "lpips_loss": lpips_loss,
        "gan_loss": gan_loss,
        "gan_weight": adaptive_w,
    }
    return total_loss, combined_grads, metrics


def discriminator_loss_fn(
    discriminator: DinoDisc,
    real_input: jax.Array,
    fake_input: jax.Array,
):
    def hinge_d_loss(logits_real: jax.Array, logits_fake: jax.Array) -> jax.Array:
        loss_real = jnp.mean(jax.nn.relu(1.0 - logits_real))
        loss_fake = jnp.mean(jax.nn.relu(1.0 + logits_fake))
        return 0.5 * (loss_real + loss_fake)

    logits_real = discriminator(real_input)
    logits_fake = discriminator(fake_input)

    return hinge_d_loss(logits_real, logits_fake)


def encode_flat_tokens(
    key: jax.Array,
    images: jax.Array,
    dino,
    encoder,
    noise_tau: float,
    num_latents: int,
    encoder_disposable_registers: int = 0,
) -> jax.Array:
    """Encode images into flat tokens (mu) with optional Gaussian noise."""
    b, _, _, c = images.shape
    resized = jax.image.resize(images, (b, 224, 224, c), method="bilinear")
    dino_tokens = dino(resized)[:, 5:]

    enc_out = encoder(dino_tokens, deterministic=True)
    mu, logvar = extract_mu_logvar(enc_out, num_latents, encoder_disposable_registers)
    if noise_tau > 0:
        eps = jax.random.normal(key, mu.shape, dtype=mu.dtype)
        flat_tokens = mu + noise_tau * eps * jnp.exp(0.5 * logvar)
    else:
        flat_tokens = mu
    return flat_tokens


def _infer_flatdino_latent_dim(enc_cfg: ViTConfig) -> int:
    """Infer latent dimensionality (mu/logvar) from the FlatDINO encoder config."""
    if enc_cfg.output_dim is not None:
        if enc_cfg.output_dim % 2 != 0:
            raise ValueError("FlatDINO encoder output_dim must be even (mu/logvar split).")
        return enc_cfg.output_dim // 2

    latent_dim = enc_cfg.transformer.residual_dim
    if latent_dim is None:
        raise ValueError("FlatDINO encoder config missing output_dim and embed_dim.")
    return latent_dim


@nnx.jit(
    static_argnames=(
        "should_update_ema",
        "ema_momentum",
        "use_gan",
        "use_lpips",
        "lpips_weight",
        "noise_tau",
        "num_latents",
        "encoder_disposable_registers",
    ),
    donate_argnames=("decoder", "decoder_ema", "optim_dec", "discriminator"),
)
def train_generator_step(
    key: jax.Array,
    decoder: RAEDecoder,
    decoder_ema: RAEDecoder,
    optim_dec: nnx.Optimizer,
    discriminator: DinoDisc,
    dino,
    flat_enc,
    imgs: jax.Array,
    use_gan: bool,
    gan_weight: float,
    diffaug: DiffAug,
    max_d_weight: float,
    use_lpips: bool,
    lpips: LPIPS,
    lpips_weight: float,
    ema_momentum: float,
    noise_tau: float,
    num_latents: int,
    encoder_disposable_registers: int = 0,
    should_update_ema: bool = True,
) -> tuple[jax.Array, dict[str, ...]]:
    """
    Returns:
     - latents
     - metrics (dict)
    """
    sample_key, loss_key = jax.random.split(key)
    chex.assert_rank(imgs, 4)
    latents = encode_flat_tokens(
        sample_key, imgs, dino, flat_enc, noise_tau, num_latents, encoder_disposable_registers
    )

    gen_loss, gen_grads, metrics = generator_loss_fn(
        loss_key,
        decoder,
        imgs,
        latents,
        use_lpips,
        lpips,
        use_gan,
        diffaug,
        discriminator,
        lpips_weight=lpips_weight,
        gan_weight=gan_weight,
        max_d_weight=max_d_weight,
    )

    optim_dec.update(decoder, gen_grads)

    if should_update_ema:
        new_ema_state = jax.tree.map(
            lambda t, s: t * ema_momentum + s * (1 - ema_momentum),
            nnx.state(decoder_ema, nnx.Param),
            nnx.state(decoder, nnx.Param),
        )
        nnx.update(decoder_ema, new_ema_state)

    return latents, metrics


@nnx.jit(donate_argnames=("discriminator", "optim_disc"))
def train_discriminator_step(
    key: jax.Array,
    discriminator: DinoDisc,
    optim_disc: nnx.Optimizer,
    decoder: RAEDecoder,
    imgs: jax.Array,
    latents: jax.Array,
    diffaug: DiffAug,
) -> dict[str, ...]:
    img_mean = rearrange(jnp.asarray(IMAGENET_DEFAULT_MEAN, dtype=imgs.dtype), "c -> 1 1 1 c")
    img_std = rearrange(jnp.asarray(IMAGENET_DEFAULT_STD, dtype=imgs.dtype), "c -> 1 1 1 c")

    fake_imgs = decoder(latents)
    fake_imgs = fake_imgs[:, : decoder.num_reg]
    fake_imgs = decoder.unpatchify(fake_imgs, denorm_output=False)
    fake_imgs = 2 * (fake_imgs * img_std + img_mean) - 1
    real_imgs = 2 * (imgs * img_std + img_mean) - 1
    fake_imgs = jnp.clip(fake_imgs, -1.0, 1.0)
    real_imgs = jnp.clip(real_imgs, -1.0, 1.0)

    # Discretize fake images to simulate uint8 quantization at inference time
    fake_imgs = jnp.round((fake_imgs + 1.0) * 127.5) / 127.5 - 1.0

    fake_key, real_key = jax.random.split(key)
    fake_imgs = diffaug(fake_key, fake_imgs)
    real_imgs = diffaug(real_key, real_imgs)

    disc_loss, disc_grads = nnx.value_and_grad(discriminator_loss_fn)(
        discriminator, real_imgs, fake_imgs
    )

    optim_disc.update(discriminator, disc_grads)

    return {"loss_disc": disc_loss}


def compute_recon_fid(
    decoder: RAEDecoder,
    dino,
    flat_enc,
    cfg: GeneratorConfig,
    mesh: jax.sharding.Mesh,
    *,
    val_loader: grain.DataLoader,
    batch_size: int,
    num_latents: int,
    encoder_disposable_registers: int = 0,
    num_eval_images: int | None = None,
    verbose: bool = False,
    noise_tau: float = 0.0,
) -> float:
    """Compute reconstruction FID on ImageNet validation."""
    inception_forward = fid_eval.create_inception_forward()
    fid_key = jax.random.PRNGKey(0)

    @nnx.jit
    def inception_feats(batch) -> jax.Array:
        return inception_forward(batch["image"])

    @nnx.jit
    def recon_feats(dec: RAEDecoder, imgs: jax.Array) -> jax.Array:
        b, _, _, c = imgs.shape
        latents = encode_flat_tokens(
            fid_key, imgs, dino, flat_enc, 0.0, num_latents, encoder_disposable_registers
        )
        tokens = dec(latents, deterministic=True)
        tokens = tokens[:, : dec.num_reg]
        recon = dec.unpatchify(tokens, denorm_output=False)
        recon = recon * dec.img_std.value + dec.img_mean.value
        recon = jnp.clip(recon, 0.0, 1.0)
        recon = jax.image.resize(recon, (b, 299, 299, c), method="bilinear")
        recon = (recon - 0.5) / 0.5
        return inception_forward(recon)

    def recon_feat_fn(batch) -> jax.Array:
        return recon_feats(decoder, batch["image"])

    real_stats = fid_eval.fid_dataset(
        cfg.data,
        batch_size,
        inception_feats,
        num_eval_images=num_eval_images,
        verbose=verbose,
        mesh=mesh,
    )

    val_iter = prefetch_to_mesh(iter(val_loader), 1, mesh=mesh, pad_to=mesh.size, trim=True)
    recon_stats = fid_eval.fid_for_iterable(
        val_iter, recon_feat_fn, num_eval_images=num_eval_images, verbose=verbose
    )

    return fid_eval.frechet_distance(real_stats, recon_stats)


def main(args: Args, cfg: GeneratorConfig):
    # Initialize distributed JAX for multi-host setups (e.g., TPU pods)
    # Must be called before any JAX operations
    if args.distributed:
        init_distributed()

    # Silence non-primary hosts to avoid duplicate log messages
    if not is_primary_host():
        logging.set_verbosity(logging.WARNING)

    target_cfg = cfg
    # Use different RNG seeds per host for data augmentation diversity
    rngs = nnx.Rngs(args.seed + jax.process_index())

    # Create 2D mesh for FSDP: (data_parallel, model_parallel)
    # When fsdp=1, model axis has size 1 (weights replicated, pure data parallelism)
    # When fsdp>1, model parameters are sharded across fsdp devices
    num_devices = jax.device_count()
    if num_devices % args.fsdp != 0:
        raise ValueError(
            f"Number of devices ({num_devices}) must be divisible by fsdp ({args.fsdp})"
        )
    data_parallel_size = num_devices // args.fsdp
    mesh = jax.make_mesh((data_parallel_size, args.fsdp), ("data", "model"))
    jax.set_mesh(mesh)  # Set globally for FSDP sharding annotations
    logging.info(f"Mesh shape: data={data_parallel_size}, model={args.fsdp}")

    # Open restore manager using the shared utility
    default_restore_path = f"output/flatdino/decoder/{args.experiment}"
    restore_mngr, ckpt_step = open_restore_manager(
        args.restore, args.maybe_restore, default_restore_path, args.gcs_bucket, ITEM_NAMES
    )

    if restore_mngr is not None and ckpt_step > 0:

        cfg_d = restore_mngr.restore(
            ckpt_step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
        )["config"]
        ckpt_cfg = from_dict(GeneratorConfig, cfg_d, DaciteConfig(cast=[tuple], strict=False))
        if ckpt_cfg.train.batch_size != target_cfg.train.batch_size:
            raise ValueError(
                "Restoring with a different train.batch_size is not supported; "
                "checkpointed optimizer/loader state assumes the saved global batch size."
            )
        cfg = replace(ckpt_cfg, train=target_cfg.train)

    # With FSDP, only data_parallel_size contributes to batch scaling
    # (model parallel devices share the same batch)
    assert cfg.train.batch_size % (args.gpu_batch_size * data_parallel_size) == 0
    grad_acc_steps = cfg.train.batch_size // (args.gpu_batch_size * data_parallel_size)
    micro_bs = args.gpu_batch_size * data_parallel_size
    rfid_batch_size = args.rfid_batch_size or micro_bs
    if rfid_batch_size % data_parallel_size != 0:
        raise ValueError("rFID batch size must be divisible by data_parallel_size.")

    data_cfg = cfg.data
    if args.num_data_workers is not None:
        data_cfg = replace(data_cfg, num_workers=args.num_data_workers)

    data: DataLoaders = create_dataloaders(
        data_cfg,
        micro_bs,
        train_aug=RAEDecoderTrainAugmentations(cfg.aug, cfg.data),
        val_aug=RAEDecoderValAugmentations(cfg.aug, cfg.data),
        val_epochs=1,
        drop_remainder_train=True,
        drop_remainder_val=False,
        val_shuffle=True,
        gcs_bucket=args.gcs_bucket if args.data_in_bucket else None,
    )
    train_loader = data.train_loader
    total_updates = (data.train_ds_size * cfg.train.epochs) // cfg.train.batch_size
    data_iter = iter(train_loader)
    steps_per_epoch = total_updates // cfg.train.epochs

    logging.info(f"Gradient accumulation steps: {grad_acc_steps}")
    logging.info(f"Micro batch size: {micro_bs}")
    logging.info(f"rFID batch size: {rfid_batch_size}")
    logging.info(f"train updates: {total_updates}")
    logging.info(f"updates per epoch: {steps_per_epoch}")

    lr_sched_dec = build_lr_schedule(cfg.train, total_updates)
    lr_sched_disc = build_lr_schedule(
        cfg.train,
        total_updates,
        warmup_steps_override=steps_per_epoch,
        decay_steps_override=(cfg.train.epochs - cfg.train.disc_start) * steps_per_epoch,
    )

    # Compute flatdino path - use GCS path if gcs_bucket is set
    flatdino_ckpt_path: Path | str = (
        f"gs://{args.gcs_bucket}/{args.flatdino_path}"
        if args.gcs_bucket is not None
        else args.flatdino_path
    )
    restored = restore_encoder(flatdino_ckpt_path, mesh=mesh, mp=mp, encoder=True, decoder=False)
    if restored.encoder is None:
        raise ValueError("FlatDINO encoder must be restored for tokenization.")

    dino = restored.dino
    dino.eval()
    restored.encoder.eval()

    latent_dim = _infer_flatdino_latent_dim(restored.encoder.cfg)
    num_flat_tokens = restored.num_flat_tokens
    decoder_vit = replace(
        cfg.vit,
        patch=None,
        num_patches=num_flat_tokens,
        num_registers=cfg.vit.num_registers,
    )
    decoder_input_dim = decoder_vit.input_dim or decoder_vit.transformer.residual_dim
    if decoder_input_dim != latent_dim:
        logging.warning(
            "Adjusting decoder input_dim from %s to match FlatDINO latent dim %s.",
            decoder_input_dim,
            latent_dim,
        )
        decoder_vit = replace(decoder_vit, input_dim=latent_dim)
    cfg = replace(cfg, vit=decoder_vit)
    grid = int(math.sqrt(decoder_vit.num_registers))
    if grid * grid != decoder_vit.num_registers:
        raise ValueError(
            f"Decoder registers ({decoder_vit.num_registers}) do not form a square grid."
        )
    if grid * cfg.patch_size != cfg.aug.crop_size[0]:
        raise ValueError("Patch size and grid size do not match the crop size.")

    # Only primary host logs to wandb
    if args.use_wandb and is_primary_host():
        name = args.experiment
        wandb.init(project=args.project_name, name=name, config=asdict(cfg))

    save_path = determine_save_path(
        checkpoint_enabled=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        default_path=f"output/flatdino/decoder/{args.experiment}",
        gcs_bucket=args.gcs_bucket,
    )

    decoder = RAEDecoder(decoder_vit, cfg.patch_size, cfg.out_channels, mp, rngs=rngs)
    decoder_ema = RAEDecoder(decoder_vit, cfg.patch_size, cfg.out_channels, mp, rngs=rngs)
    nnx.update(decoder_ema, jax.tree.map(lambda x: jnp.copy(x), nnx.state(decoder, nnx.Param)))
    set_attn_implementation(decoder, args.implementation)
    set_attn_implementation(decoder_ema, args.implementation)
    logging.info(f"Attention implementation set to: {args.implementation}")
    discriminator = DinoDisc(ks=9, mp=mp, rngs=rngs)
    diffaug = DiffAug(cfg.diffaug)

    # Restore model states if resuming from checkpoint
    if restore_mngr is not None:
        restore_model_state(restore_mngr, ckpt_step, decoder, mesh, "decoder")
        restore_model_state(restore_mngr, ckpt_step, decoder_ema, mesh, "decoder_ema")
        restore_model_state(restore_mngr, ckpt_step, discriminator, mesh, "discriminator")
        logging.info("Restored model states from checkpoint")

    if save_path is not None:
        # Convert to string path (handles both local Path and GCS string paths)
        if isinstance(save_path, Path):
            path_str = str(save_path.absolute())
        else:
            path_str = save_path

        # Always save at the final step to ensure we don't lose training progress
        save_on_steps = frozenset([total_updates])

        opts = ocp.CheckpointManagerOptions(
            save_interval_steps=steps_per_epoch,
            save_on_steps=save_on_steps,
            max_to_keep=2,
            create=True,
            read_only=False,
            keep_checkpoints_without_metrics=args.keep_checkpoints_without_metrics,
        )
        mngr = ocp.CheckpointManager(path_str, options=opts, item_names=ITEM_NAMES)
    else:
        mngr = None

    chain_dec = optax.MultiSteps(
        optax.adamw(
            lr_sched_dec, cfg.train.adam_b1, cfg.train.adam_b2, weight_decay=cfg.train.weight_decay
        ),
        grad_acc_steps,
    )
    chain_disc = optax.MultiSteps(
        optax.adamw(
            lr_sched_disc, cfg.train.adam_b1, cfg.train.adam_b2, weight_decay=cfg.train.weight_decay
        ),
        grad_acc_steps,
    )
    optim_dec = nnx.Optimizer(decoder, chain_dec, wrt=nnx.Param)  # ty: ignore
    optim_disc = nnx.Optimizer(discriminator, chain_disc, wrt=nnx.Param)  # ty: ignore

    # Restore optimizer and data loader states if resuming from checkpoint
    if restore_mngr is not None:
        restore_optimizer_state(restore_mngr, ckpt_step, optim_dec, mesh, "optim_dec")
        restore_optimizer_state(restore_mngr, ckpt_step, optim_disc, mesh, "optim_disc")
        data_iter = restore_data_loader(restore_mngr, ckpt_step, data_iter)
        logging.info("Restored optimizer and data loader states from checkpoint")

    decoder_params = jax.tree.leaves(nnx.state(decoder, nnx.Param))
    decoder_param_count = sum(jax.tree.map(lambda x: jnp.size(x), decoder_params))
    discriminator_params = jax.tree.leaves(nnx.state(discriminator, nnx.Param))
    discriminator_param_count = sum(jax.tree.map(lambda x: jnp.size(x), discriminator_params))
    logging.info(f"decoder params: {decoder_param_count / 1_000_000:.2f}M params")
    logging.info(f"discriminator params: {discriminator_param_count / 1_000_000:.2f}M params")

    train_stream = prefetch_to_mesh(data_iter, 1, mesh)

    # Only primary host logs to wandb
    use_wandb = args.use_wandb and is_primary_host()

    if use_wandb:
        log_grid_side = 8
        imgs = jnp.asarray(next(iter(data.val_loader))["image"][:64])
        log_latents = encode_flat_tokens(
            rngs(), imgs, dino, restored.encoder, 0.0,
            num_flat_tokens, restored.encoder_disposable_registers,
        )

    lpips = LPIPS(rngs=rngs)

    discriminator.eval()
    updates_completed = int(optax.tree_utils.tree_get(optim_dec.opt_state, "gradient_step"))
    pbar = tqdm(
        desc="Update",
        initial=updates_completed,
        total=total_updates,
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        disable=not is_primary_host(),
    )
    profiler = TrainingProfiler(
        mode=args.profile_mode,
        port=args.profiler_port,
        start_step=args.profiler_start_step,
        stop_step=args.profiler_stop_step,
    )

    for samples in train_stream:

        use_lpips = updates_completed >= steps_per_epoch * cfg.train.lpips_start
        use_gan = updates_completed >= steps_per_epoch * cfg.train.disc_gan_start
        train_disc = updates_completed >= steps_per_epoch * cfg.train.disc_start

        prev_updates = updates_completed
        # EMA should track optimizer updates; trigger on the last micro-step in each accumulation window.
        mini_step = int(optax.tree_utils.tree_get(optim_dec.opt_state, "mini_step"))
        should_update_ema = (mini_step + 1) % grad_acc_steps == 0

        # Set discriminator to eval mode during generator training (no SpectralNorm stats updates)
        discriminator.eval()
        latents, gen_metrics = train_generator_step(
            rngs(),
            decoder,
            decoder_ema,
            optim_dec,
            discriminator,
            dino,
            restored.encoder,
            samples["image"],
            use_gan=use_gan,
            gan_weight=cfg.train.gan_weight,
            diffaug=diffaug,
            max_d_weight=cfg.train.max_d_weight,
            use_lpips=use_lpips,
            lpips=lpips,
            lpips_weight=cfg.train.lpips_weight,
            ema_momentum=cfg.train.ema_decay,
            noise_tau=cfg.noise_tau,
            num_latents=num_flat_tokens,
            encoder_disposable_registers=restored.encoder_disposable_registers,
            should_update_ema=should_update_ema,
        )

        if train_disc:
            # Set discriminator to train mode for discriminator training (update SpectralNorm stats)
            discriminator.train()
            disc_metrics = train_discriminator_step(
                rngs(), discriminator, optim_disc, decoder, samples["image"], latents, diffaug
            )
        metrics = {}

        updates_completed = int(optax.tree_utils.tree_get(optim_dec.opt_state, "gradient_step"))
        ran_update = updates_completed > prev_updates

        if ran_update:
            pbar.update(updates_completed - prev_updates)
            profiler.step(updates_completed)

            # Check preemption FIRST - if preempted, save immediately without validation
            is_preemption = mngr.reached_preemption(updates_completed) if mngr else False

            if is_preemption:
                # Preemption detected - save checkpoint immediately
                logging.info(
                    "Preemption detected at step %d. Saving checkpoint immediately...",
                    updates_completed,
                )
                save_checkpoint(
                    mngr, updates_completed, decoder, decoder_ema, discriminator,
                    optim_dec, optim_disc, data_iter, cfg
                )
                mngr.wait_until_finished()
                logging.info("Checkpoint saved. Exiting due to preemption.")
                break  # Exit the training loop

            # Check if we should save a checkpoint at this step (regular interval)
            should_save_now = mngr.should_save(updates_completed) if mngr else False

            if should_save_now:
                save_checkpoint(
                    mngr, updates_completed, decoder, decoder_ema, discriminator,
                    optim_dec, optim_disc, data_iter, cfg
                )

        if use_wandb:
            if ran_update and updates_completed % args.wandb_log_every == 0:
                count_dec = optax.tree_utils.tree_get(optim_dec.opt_state, "gradient_step")
                count_disc = optax.tree_utils.tree_get(optim_disc.opt_state, "gradient_step")
                metrics["train/lr_dec"] = lr_sched_dec(int(count_dec))
                metrics["train/lr_disc"] = lr_sched_disc(int(count_disc))
                metrics = metrics | {name: float(value) for name, value in gen_metrics.items()}
                if train_disc:
                    metrics = metrics | {name: float(value) for name, value in disc_metrics.items()}

            if ran_update and updates_completed % steps_per_epoch == 0:
                assert log_latents is not None
                tokens = decoder_ema(log_latents, deterministic=True)
                tokens = tokens[:, : decoder_ema.num_reg]
                recon = decoder_ema.unpatchify(tokens, denorm_output=True)
                recon = np.clip(np.asarray(recon), 0.0, 1.0)
                side = min(log_grid_side, int(math.sqrt(recon.shape[0])))
                grid_image = rearrange(
                    recon[: side * side],
                    "(r c) h w d -> (r h) (c w) d",
                    r=side,
                    c=side,
                )
                metrics["val/recon_grid"] = wandb.Image(
                    grid_image, caption=f"epoch {updates_completed // steps_per_epoch}"
                )

        if use_wandb and metrics:
            metrics["step"] = updates_completed
            wandb.log(metrics)

        if updates_completed >= total_updates:
            break

    pbar.close()
    if mngr is not None:
        mngr.close()

    if args.val_end:
        fid_bs = args.rfid_batch_size or micro_bs
        fid_value = compute_recon_fid(
            decoder_ema,
            dino,
            restored.encoder,
            cfg,
            mesh,
            val_loader=data.val_loader,
            batch_size=fid_bs,
            num_latents=num_flat_tokens,
            encoder_disposable_registers=restored.encoder_disposable_registers,
            num_eval_images=None,
            verbose=args.debug,
            noise_tau=0.0,
        )
        logging.info(f"Reconstruction FID (Inception V3): {fid_value:.4f}")
        if use_wandb:
            wandb.log({"fid/reconstruction": fid_value, "step": updates_completed})

    if use_wandb:
        wandb.finish()


baseline = GeneratorConfig()


def _swiglu_config(vit_name: str) -> dict:
    """Create parameter-matched SwiGLU config from VIT_CONFIGS.

    SwiGLU has 3 weight matrices (up, gate, down) vs GELU's 2 (up, down).
    To match parameters: new_hidden_dim = (2/3) * original_hidden_dim.
    """
    cfg = VIT_CONFIGS[vit_name].copy()
    cfg["mlp_hidden_dim"] = cfg["embed_dim"] * 8 // 3
    cfg["mlp_type"] = "swiglu"
    return cfg


experiments = {
    "vit-s": replace(
        baseline,
        vit=replace(
            baseline.vit, transformer=replace(baseline.vit.transformer, **_swiglu_config("vit-s"))
        ),
    ),
    "vit-b": replace(
        baseline,
        vit=replace(
            baseline.vit, transformer=replace(baseline.vit.transformer, **_swiglu_config("vit-b"))
        ),
    ),
    "vit-l": replace(
        baseline,
        vit=replace(
            baseline.vit, transformer=replace(baseline.vit.transformer, **_swiglu_config("vit-l"))
        ),
    ),
    "vit-xl": replace(
        baseline,
        vit=replace(
            baseline.vit, transformer=replace(baseline.vit.transformer, **_swiglu_config("vit-xl"))
        ),
    ),
}


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    assert args.experiment in experiments.keys(), (
        f"{args.experiment} is not a valid experiment name"
    )

    if args.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().use_absl_log_file()

    config = experiments[args.experiment]
    main(args, config)
