from typing import Literal, Any, NamedTuple
from dataclasses import dataclass, asdict, field, replace
from pathlib import Path
import json
import re

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import matplotlib.pyplot as plt
import grain.python as grain
import chex
import jmp
import flax.nnx as nnx
import numpy as np
import optax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.checkpoint_managers import preservation_policy as pp
import tyro
import wandb
from tqdm import tqdm
from dacite import from_dict, Config as DaciteConfig
from einops import rearrange
from absl import logging

from flatdino.data import DataConfig, DataLoaders, create_dataloaders
from flatdino.models.vit import ViTEncoder
from flatdino.models.transformer import set_attn_implementation
from flatdino.models.dit import LightningDiT, DiTConfig, LIGHTNING_DIT_CONFIGS
from flatdino.decoder.dit_dh import DiTDH, DiTDHConfig
from flatdino.decoder.sampling import xpred_one_minus_t
from flatdino.augmentations import (
    FlatDinoGeneratorTrainAugmentations,
    FlatDinoValAugmentations,
    FlatDinoAugConfig,
)
from flatdino.eval import restore_encoder
from flatdino.utils import (
    build_lr_schedule,
    extract_decoder_patches,
    extract_mu_logvar,
)
from flatdino.eval.gfid import compute_gfid
from flatdino.pretrained.rae_decoder import make_rae_decoder
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.distributed import (
    prefetch_to_mesh,
    TrainingProfiler,
    is_primary_host,
    init_distributed,
    determine_save_path,
    open_restore_manager,
    restore_data_loader,
    restore_optimizer_state,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


METRICS_FILENAME = "metrics.json"


def load_metrics(checkpoint_path: Path) -> dict[str, Any]:
    """Load training metrics from JSON file in checkpoint folder."""
    metrics_path = checkpoint_path / METRICS_FILENAME
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {"validation": []}


def save_metrics(checkpoint_path: Path, step: int, metrics: dict[str, float]) -> None:
    """Append validation metrics to metrics.json in checkpoint folder.

    Saves metrics with their training step for later plotting.
    """
    existing = load_metrics(checkpoint_path)
    record = {"step": step, **metrics}
    existing["validation"].append(record)
    metrics_path = checkpoint_path / METRICS_FILENAME
    with open(metrics_path, "w") as f:
        json.dump(existing, f, indent=2)


@dataclass
class Args:
    flatdino_path: Path
    """Path to the FlatDINO VAE checkpoint. When gcs_bucket is set, this path is
    automatically prefixed with gs://{gcs_bucket}/."""
    seed: int = 42
    restore: Path | Literal["default"] | None = None
    """Path to restore checkpoint from. Use 'default' to restore from the default
    checkpoint path (output/flatdino-generator/{flatdino_name}/{experiment} or GCS equivalent)."""
    maybe_restore: Path | Literal["default"] | None = None
    """Like --restore, but if no checkpoint exists at the path, logs a warning and trains
    from scratch instead of crashing. Useful for resumable training scripts."""
    checkpoint: bool = True
    checkpoint_dir: Path | None = None
    keep_all_checkpoints: bool = False
    keep_checkpoints_without_metrics: bool = False
    """If True, preemption checkpoints (which have no validation metrics) are preserved
    indefinitely. If False, they are cleaned up once we have enough checkpoints with metrics."""
    gcs_bucket: str | None = None
    """GCS bucket name for checkpoints and flatdino_path. When set, checkpoints are saved
    to gs://{gcs_bucket}/output/... and flatdino_path is loaded from gs://{gcs_bucket}/...
    using orbax's native GCS support. Requires running on a GCP instance."""
    data_in_bucket: bool = False
    """When True and gcs_bucket is set, load datasets from gs://{gcs_bucket}/{dataset_name}
    via tfds instead of from local storage."""
    num_data_workers: int | None = None
    """Override the number of data loading workers. If None, uses the value from DataConfig."""
    use_wandb: bool = False
    wandb_log_every: int = 20
    project_name: str = "flatdino-generator"
    gpu_batch_size: int = 128
    profile_mode: Literal["disabled", "always", "window"] = "disabled"
    """Profiler mode: 'disabled' (no profiling), 'always' (start immediately and run forever),
    or 'window' (run between profiler_start_step and profiler_stop_step)."""
    profiler_port: int = 7777
    profiler_start_step: int = 10
    """Step to start profiling (only used in 'window' mode)."""
    profiler_stop_step: int = 2000
    """Step to stop profiling (only used in 'window' mode)."""
    val_epochs_freq: int = 5
    skip_val_gfid: bool = False
    """Skip gFID computation during validation. Checkpoints are still saved at the same
    intervals, but without validation metrics."""
    experiment: str = "ditb"
    overfit_one_image: bool = False
    gfid_steps: int = 50
    gfid_per_class: int = 50
    gfid_batch_size: int | None = None
    gfid_cfg_scale: float | None = 1.0
    gfid_grid_size: int = 8
    gfid_verbose: bool = False
    debug: bool = False
    implementation: Literal["cudnn", "xla"] = "xla"
    """Attention implementation: 'cudnn' for Flash Attention, 'xla' for default."""
    fsdp: int = 1
    """FSDP (Fully Sharded Data Parallelism) axis size. When fsdp=1, only data parallelism
    is used. When fsdp>1, model parameters are sharded across fsdp devices using the
    Megatron-LM pattern. The data axis size is num_devices // fsdp."""
    distributed: bool = False
    """Enable distributed training for multi-host setups (e.g., TPU pods). Must be set
    before any JAX operations occur."""
    checkpoints_to_keep: tuple[int, ...] | None = None
    """List of epochs whose checkpoints should be kept forever (e.g., --checkpoints-to-keep 80 400 800).
    These are converted to steps internally and passed to orbax's should_keep_fn."""
    stats_path: Path | None = None
    """Override the default stats path ({flatdino_path}/stats.npz) for latent normalization.
    Only used when experiment name includes '-norm'."""


@dataclass
class OptimConfig:
    epochs: int = 800
    batch_size: int = 1024

    adam_b1: float = 0.9
    adam_b2: float = 0.95
    lr_start: float = 2e-4
    lr_peak: float = 2e-4
    lr_final: float = 2e-5
    warmup_epochs: int = 40

    ema: float = 0.9995

    lr_schedule: Literal["constant", "warmup_cosine", "wsd", "linear_decay"] = "constant"


@dataclass
class GeneratorConfig:
    model_type: Literal["dit", "dit_dh"] = "dit"
    train: OptimConfig = field(default_factory=lambda: OptimConfig())
    dit: DiTConfig = field(default_factory=lambda: DiTConfig(patch_embed=None, num_patches=32))
    dit_dh: DiTDHConfig = field(
        default_factory=lambda: DiTDHConfig(input_size=32, patch_size=None, in_channels=768)
    )
    data: DataConfig = field(default_factory=lambda: DataConfig())
    aug: FlatDinoAugConfig = field(default_factory=lambda: FlatDinoAugConfig())
    kappa: float | None = None
    sample_latents: bool = False
    time_dist_shift_base: int | None = 4096
    pred_type: Literal["v", "x"] = "v"
    use_latent_norm: bool = False
    """Whether to normalize latents to zero mean/unit variance before diffusion."""


def _save_overfit_comparison(
    batch: dict[str, jax.Array],
    restored,
    output_dir: Path,
    seed: int,
    mp: jmp.Policy,
):
    """Save original and reconstructed versions of the overfit image for debugging."""
    if restored.encoder is None or restored.decoder is None:
        logging.info("FlatDINO decoder not available; skipping overfit comparison save.")
        return
    if restored.data_cfg is None or restored.aug_cfg is None:
        logging.info("Missing data/augmentation config; skipping overfit comparison save.")
        return

    image_h, image_w = restored.aug_cfg.image_size
    if image_h != image_w:
        logging.warning("Only square images are supported for overfit comparison saving.")
        return

    # Use a single sample to keep the visualization lightweight.
    images = jax.lax.index_in_dim(batch["image"], 0, axis=0, keepdims=True)
    resized = jax.image.resize(images, (images.shape[0], 224, 224, images.shape[3]), "bilinear")
    patches = restored.dino(resized)[:, 5:]

    rae_decoder = make_rae_decoder(
        num_patches=patches.shape[1],
        image_size=image_h,
        dtype=mp.param_dtype,
        seed=seed,
    )

    rae_out = rae_decoder(patches)
    rae_recon = rae_decoder.unpatchify(rae_out.logits)
    rae_recon = jnp.transpose(rae_recon, (0, 2, 3, 1))

    mu_latent, _ = _encode_mu_logvar(
        restored.encoder,
        patches,
        deterministic=True,
        num_latents=restored.num_flat_tokens,
        encoder_disposable_registers=restored.encoder_disposable_registers,
    )
    if restored.tanh:
        mu_latent = jnp.tanh(mu_latent)
    num_output_patches = restored.decoder.num_reg - restored.decoder_disposable_registers
    decoded_tokens = extract_decoder_patches(
        restored.decoder(mu_latent, deterministic=True),
        num_output_patches,
        restored.decoder_disposable_registers,
    )
    decoded_out = rae_decoder(decoded_tokens)
    encoded_recon = rae_decoder.unpatchify(decoded_out.logits)
    encoded_recon = jnp.transpose(encoded_recon, (0, 2, 3, 1))

    mean = jnp.array(restored.data_cfg.normalization_mean, dtype=jnp.float32)[None, None, None, :]
    std = jnp.array(restored.data_cfg.normalization_std, dtype=jnp.float32)[None, None, None, :]

    def _denorm(x):
        return jnp.clip(x * std + mean, 0.0, 1.0)

    viz_images = {
        "Original": _denorm(images),
        "Encoded+Decoded": _denorm(encoded_recon),
        "RAE Decode": _denorm(rae_recon),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "overfit_one_image.png"

    np_images = {k: np.asarray(jax.device_get(v[0])) for k, v in viz_images.items()}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, img) in zip(axes, np_images.items()):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    logging.info(f"Saved overfit comparison to {output_path}")


def make_generator(
    cfg: GeneratorConfig,
    mp: jmp.Policy,
    rngs: nnx.Rngs,
    *,
    num_tokens: int | None,
    latent_mean: jax.Array | None = None,
    latent_std: jax.Array | None = None,
):
    match cfg.model_type:
        case "dit":
            return LightningDiT(
                cfg.dit,
                mp=mp,
                rngs=rngs,
                num_tokens=num_tokens,
                latent_mean=latent_mean,
                latent_std=latent_std,
            )
        case "dit_dh":
            dit_cfg = cfg.dit_dh
            if dit_cfg.patch_size is None and num_tokens is not None:
                dit_cfg = replace(dit_cfg, input_size=num_tokens)
            return DiTDH(
                dit_cfg,
                mp=mp,
                rngs=rngs,
                latent_mean=latent_mean,
                latent_std=latent_std,
            )
        case _:
            raise ValueError(f"Unsupported model_type: {cfg.model_type}")


def _infer_flatdino_latent_dim(encoder: ViTEncoder) -> int:
    """Infer latent dimensionality (mu/logvar) from the FlatDINO encoder config."""
    cfg = encoder.cfg
    if cfg.output_dim is not None:
        if cfg.output_dim % 2 != 0:
            raise ValueError("FlatDINO encoder output_dim must be even (mu/logvar split).")
        return cfg.output_dim // 2

    latent_dim = cfg.transformer.embed_dim
    if latent_dim is None:
        raise ValueError("FlatDINO encoder config missing embed_dim.")
    return latent_dim


def _encode_mu_logvar(
    encoder: ViTEncoder,
    tokens: jax.Array,
    *,
    deterministic: bool,
    num_latents: int,
    encoder_disposable_registers: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Run the encoder and split its output into (mu, logvar) for the latent tokens."""
    enc_out = encoder(tokens, deterministic=deterministic)
    return extract_mu_logvar(enc_out, num_latents, encoder_disposable_registers)


@nnx.jit(
    static_argnames=(
        "should_update_ema",
        "ema_momentum",
        "kappa",
        "model_type",
        "pred_type",
        "sample_latents",
        "xpred_steps",
        "num_latents",
        "encoder_disposable_registers",
        "tanh",
    ),
    donate_argnames=("generator", "generator_ema", "optim"),
)
def train_step(
    generator,
    generator_ema,
    dino: DinoWithRegisters,
    flat_enc: ViTEncoder,
    optim: nnx.Optimizer,
    imgs: jax.Array,
    lbls: jax.Array,
    rng: jax.Array,
    ema_momentum: float,
    should_update_ema: bool = True,
    kappa: float | None = None,
    model_type: str = "dit",
    pred_type: str = "v",
    sample_latents: bool = False,
    xpred_steps: int = 50,
    num_latents: int = 0,
    encoder_disposable_registers: int = 0,
    tanh: bool = False,
) -> jax.Array:
    """Single flow-matching training step.

    This samples Gaussian noise x0 and target latents x1 (coming from the frozen
    FlatDiNO encoder), interpolates them at time t, and trains the DiT to predict
    the velocity v = x1 - x0 following the standard linear flow-matching loss.
    """
    chex.assert_rank(imgs, 4)

    b, _, _, d = imgs.shape
    dino_imgs = jax.image.resize(imgs, (b, 224, 224, d), method="bicubic")
    dino_outputs = dino(dino_imgs)
    features = dino_outputs[:, 5:]

    mu, logvar = _encode_mu_logvar(
        flat_enc,
        features,
        deterministic=True,
        num_latents=num_latents,
        encoder_disposable_registers=encoder_disposable_registers,
    )
    if sample_latents:
        rng_, rng = jax.random.split(rng)
        eps = jax.random.normal(rng_, logvar.shape, dtype=logvar.dtype)
        latents = mu + eps * jnp.exp(0.5 * logvar)
    else:
        latents = mu
    if tanh:
        latents = jnp.tanh(latents)

    # Normalize latents if stats are provided (identity if not)
    latents = generator.normalize(latents)

    if generator.cfg.in_cls_dim is not None:
        dino_cls = dino_outputs[:, 0]
    else:
        dino_cls = None

    rng_noise, rng_time = jax.random.split(rng)
    if dino_cls is not None:
        rng_noise, rng_cls = jax.random.split(rng_noise)
    x0 = jax.random.normal(rng_noise, latents.shape, dtype=latents.dtype)
    if dino_cls is not None:
        x0_cls = jax.random.normal(rng_cls, dino_cls.shape, dtype=dino_cls.dtype)
    t = jax.random.uniform(rng_time, (latents.shape[0],), dtype=latents.dtype)
    if kappa is not None:
        t = t / (kappa - (kappa - 1.0) * t)
    t_broadcast = t.reshape((t.shape[0],) + (1,) * (latents.ndim - 1))
    xt = (1.0 - t_broadcast) * x0 + t_broadcast * latents
    if dino_cls is not None:
        t_cls = t.reshape((t.shape[0],) + (1,) * (dino_cls.ndim - 1))
        xt_cls = (1.0 - t_cls) * x0_cls + t_cls * dino_cls
        target_velocity_cls = dino_cls - x0_cls
    else:
        xt_cls = None
        target_velocity_cls = None

    if model_type == "dit_dh" and getattr(generator, "cfg").patch_size is not None:
        tokens = latents.shape[1]
        grid = int(jnp.sqrt(tokens))
        if grid * grid != tokens:
            raise ValueError(f"Latent tokens ({tokens}) are not a perfect square for DiTDH.")
        latents = jnp.reshape(latents, (latents.shape[0], grid, grid, latents.shape[-1]))
        xt = jnp.reshape(xt, (xt.shape[0], grid, grid, xt.shape[-1]))
        x0 = jnp.reshape(x0, (x0.shape[0], grid, grid, x0.shape[-1]))

    target_velocity = latents - x0

    def flow_matching_v_pred_v_loss(model) -> jax.Array:
        pred_velocity = model(xt, t, lbls, train=True, cls_=xt_cls)
        if dino_cls is not None:
            token_err = jnp.square(pred_velocity["x"] - target_velocity)
            cls_err = jnp.square(pred_velocity["cls"] - target_velocity_cls)
            return (jnp.sum(token_err) + jnp.sum(cls_err)) / (token_err.size + cls_err.size)
        else:
            return jnp.mean((pred_velocity["x"] - target_velocity) ** 2)

    def flow_matching_x_pred_v_loss(model) -> jax.Array:
        pred_x = model(xt, t, lbls, train=True, cls_=xt_cls)
        denom = xpred_one_minus_t(t, steps=xpred_steps, target_ndim=xt.ndim, time_dist_shift=kappa)
        if dino_cls is not None:
            pred_v = (pred_x["x"] - xt) / denom
            denom_cls = xpred_one_minus_t(
                t, steps=xpred_steps, target_ndim=xt_cls.ndim, time_dist_shift=kappa
            )
            pred_v_cls = (pred_x["cls"] - xt_cls) / denom_cls
            token_err = jnp.square(pred_v - target_velocity)
            cls_err = jnp.square(pred_v_cls - target_velocity_cls)
            return (jnp.sum(token_err) + jnp.sum(cls_err)) / (token_err.size + cls_err.size)
        else:
            pred_v = (pred_x["x"] - xt) / denom
            return jnp.mean((pred_v - target_velocity) ** 2)

    match pred_type:
        case "v":
            loss_fn = flow_matching_v_pred_v_loss
        case "x":
            loss_fn = flow_matching_x_pred_v_loss
        case _:
            raise ValueError("Invalid pred_type")

    loss, grads = nnx.value_and_grad(loss_fn)(generator)
    optim.update(generator, grads)

    if should_update_ema:
        new_ema_state = jax.tree.map(
            lambda t, s: t * ema_momentum + s * (1 - ema_momentum),
            nnx.state(generator_ema, nnx.Param),
            nnx.state(generator, nnx.Param),
        )
        nnx.update(generator_ema, new_ema_state)

    return loss


def run_validation(
    generator_ema,
    cfg: GeneratorConfig,
    args: Args,
    mesh,
    mp: jmp.Policy,
    r,
    kappa: float | None,
    gfid_batch_size: int,
    save_path: Path | str | None,
    updates_completed: int,
    use_wandb: bool,
) -> dict[str, float]:
    """Run validation (gFID computation) and return metrics.

    Returns:
        Dictionary with "val/gfid" and optionally saves grid images.
    """
    fid_value, sample_images = compute_gfid(
        generator_ema,
        cfg,
        mesh=mesh,
        mp=mp,
        steps=args.gfid_steps,
        per_class=args.gfid_per_class,
        batch_size=gfid_batch_size,
        seed=args.seed,
        cfg_scale=args.gfid_cfg_scale,
        verbose=args.gfid_verbose,
        restored=r,
        num_grid_images=args.gfid_grid_size * args.gfid_grid_size,
        time_dist_shift=kappa,
    )

    logging.info(f"gFID: {fid_value}")
    val_metrics: dict[str, Any] = {"val/gfid": float(fid_value)}

    # Save metrics to metrics.json for later plotting (skip for GCS paths)
    is_gcs_path = isinstance(save_path, str) and save_path.startswith("gs://")
    if save_path is not None and is_primary_host() and not is_gcs_path:
        save_metrics(save_path, updates_completed, {"gfid": float(fid_value)})

    grid_image = None
    if sample_images is not None and sample_images.size:
        grid_image = rearrange(
            sample_images[: args.gfid_grid_size * args.gfid_grid_size],
            "(r c) h w d -> (r h) (c w) d",
            r=args.gfid_grid_size,
            c=args.gfid_grid_size,
        )
        grid_image = np.clip(grid_image, 0.0, 1.0)

    # Save grid images locally (skip for GCS paths)
    if grid_image is not None and save_path is not None and is_primary_host() and not is_gcs_path:
        grid_dir = Path(save_path) / "eval_grids"
        grid_dir.mkdir(parents=True, exist_ok=True)
        grid_path = grid_dir / f"update_{updates_completed:06d}.png"
        plt.imsave(grid_path, grid_image)
        if use_wandb:
            val_metrics["val/grid"] = wandb.Image(grid_image, caption=f"update {updates_completed}")

    return val_metrics


def save_checkpoint(
    mngr: ocp.CheckpointManager | None,
    step: int,
    generator,
    generator_ema,
    optim: nnx.Optimizer,
    data_iter,
    cfg: GeneratorConfig,
    metrics: dict | None = None,
) -> bool:
    """Save checkpoint if the manager decides it should be saved.

    Returns True if a checkpoint was actually saved.
    """
    if mngr is None:
        return False
    saved = mngr.save(
        step,
        metrics=metrics,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(generator))),
            model_ema=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(generator_ema))),
            optim=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim))),
            loader=grain.PyGrainCheckpointSave(data_iter),
            config=ocp.args.JsonSave(asdict(cfg)),
        ),
    )
    return saved


def main(args: Args, cfg: GeneratorConfig):
    # Initialize distributed JAX for multi-host setups (e.g., TPU pods)
    # MUST be called before any JAX operations (including jnp array creation)
    if args.distributed:
        init_distributed()

    # Silence non-primary hosts to avoid duplicate log messages
    if not is_primary_host():
        logging.set_verbosity(logging.WARNING)

    # Create mixed precision policy AFTER distributed init
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    # Only use wandb on primary host
    use_wandb = args.use_wandb and is_primary_host()
    if use_wandb:
        wandb.init(project=args.project_name, name=args.experiment, config=asdict(cfg))

    # Create 2D mesh for FSDP: (data_parallel, model_parallel)
    # When fsdp=1, model axis has size 1 (weights replicated, pure data parallelism)
    # When fsdp>1, model parameters are sharded across fsdp devices
    # Note: jax.device_count() returns total devices across ALL hosts
    num_devices = jax.device_count()
    if num_devices % args.fsdp != 0:
        raise ValueError(
            f"Number of devices ({num_devices}) must be divisible by fsdp ({args.fsdp})"
        )
    data_parallel_size = num_devices // args.fsdp
    # Use create_device_mesh for proper TPU topology handling across all hosts
    devices = mesh_utils.create_device_mesh((data_parallel_size, args.fsdp))
    mesh = Mesh(devices, ("data", "model"))
    jax.set_mesh(mesh)  # Set globally for FSDP sharding annotations
    logging.info(f"Process count: {jax.process_count()}")
    logging.info(f"Mesh shape: data={data_parallel_size}, model={args.fsdp}")
    logging.info(f"Mesh devices: {mesh.devices}")

    # With FSDP, only data_parallel_size contributes to batch scaling
    # (model parallel devices share the same batch)
    assert cfg.train.batch_size % (args.gpu_batch_size * data_parallel_size) == 0
    grad_acc_steps = cfg.train.batch_size // (args.gpu_batch_size * data_parallel_size)
    micro_bs = args.gpu_batch_size * data_parallel_size
    gfid_batch_size = args.gfid_batch_size or micro_bs
    assert gfid_batch_size % data_parallel_size == 0, (
        "gFID bs must be divisible by data_parallel_size."
    )
    logging.info(f"Gradient accumulation steps: {grad_acc_steps}")
    logging.info(f"Micro batch size: {micro_bs}")
    logging.info(f"gFID batch size: {gfid_batch_size}")

    data_cfg = cfg.data
    if args.num_data_workers is not None:
        data_cfg = replace(data_cfg, num_workers=args.num_data_workers)

    data: DataLoaders = create_dataloaders(
        data_cfg,
        micro_bs,
        train_aug=FlatDinoGeneratorTrainAugmentations(cfg.aug, cfg.data),
        val_aug=FlatDinoValAugmentations(cfg.aug, cfg.data),
        val_epochs=1,
        drop_remainder_train=True,
        drop_remainder_val=True,
        gcs_bucket=args.gcs_bucket if args.data_in_bucket else None,
    )
    train_loader = data.train_loader
    total_updates = (data.train_ds_size * cfg.train.epochs) // cfg.train.batch_size
    (data.val_ds_size + micro_bs - 1) // micro_bs
    data_iter = iter(train_loader)

    lr_sched = build_lr_schedule(cfg.train, total_updates)

    flatdino_name = args.flatdino_path.name
    save_path = determine_save_path(
        checkpoint_enabled=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        default_path=f"output/flatdino-generator/{flatdino_name}/{args.experiment}",
        gcs_bucket=args.gcs_bucket,
    )

    # Calculate validation/checkpoint interval based on val_epochs_freq
    # Validation and checkpointing happen at the same steps so checkpoints have metrics
    steps_per_epoch = total_updates // cfg.train.epochs if cfg.train.epochs > 0 else total_updates
    if args.val_epochs_freq > 0:
        save_interval_steps = steps_per_epoch * args.val_epochs_freq
    else:
        # Default to saving once per epoch if val_epochs_freq is 0
        save_interval_steps = steps_per_epoch

    # Use different RNG seeds per host for data augmentation diversity
    rngs = nnx.Rngs(args.seed + jax.process_index())

    # Compute flatdino path - use GCS path if gcs_bucket is set
    flatdino_ckpt_path: Path | str = (
        f"gs://{args.gcs_bucket}/{args.flatdino_path}"
        if args.gcs_bucket is not None
        else args.flatdino_path
    )
    # Only restore the decoder when needed (gFID validation or overfit mode) for memory efficiency,
    # since training only requires the encoder to produce target latents.
    need_decoder = ((args.val_epochs_freq > 0) and not args.skip_val_gfid) or args.overfit_one_image
    r = restore_encoder(
        flatdino_ckpt_path,
        mesh=mesh,
        mp=mp,
        encoder=True,
        decoder=need_decoder,
    )

    assert r.num_flat_tokens == cfg.dit.num_patches
    logging.info(f"FlatDINO num tokens: {r.num_flat_tokens}")

    # Load latent normalization stats if enabled in config
    latent_mean: jax.Array | None = None
    latent_std: jax.Array | None = None
    if cfg.use_latent_norm:
        if args.stats_path is not None:
            stats_path = args.stats_path
        else:
            # Default: {flatdino_path}/stats.npz
            stats_path = args.flatdino_path / "stats.npz"
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Latent normalization enabled (-norm in experiment name) but stats file not found: {stats_path}. "
                f"Run compute_stats.py first or specify --stats-path."
            )
        logging.info(f"Loading latent stats from {stats_path}")
        stats = np.load(stats_path)
        latent_mean = jnp.asarray(stats["mean"], dtype=jnp.float32)
        latent_std = jnp.sqrt(jnp.asarray(stats["var"], dtype=jnp.float32))
        logging.info(f"  Mean shape: {latent_mean.shape}, range: [{latent_mean.min():.4f}, {latent_mean.max():.4f}]")
        logging.info(f"  Std shape: {latent_std.shape}, range: [{latent_std.min():.4f}, {latent_std.max():.4f}]")

    item_names = ["model", "model_ema", "optim", "loader", "config"]
    latent_dim = _infer_flatdino_latent_dim(r.encoder) if r.encoder is not None else None

    # Open restore manager using the shared utility
    default_restore_path = f"output/flatdino-generator/{flatdino_name}/{args.experiment}"
    restore_mngr, step = open_restore_manager(
        args.restore, args.maybe_restore, default_restore_path, args.gcs_bucket, item_names
    )

    if restore_mngr is not None and step > 0:
        # Restore the config. The actual model restore happens in its constructor
        cfg_d = restore_mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))[
            "config"
        ]
        cfg = from_dict(GeneratorConfig, cfg_d, DaciteConfig(cast=[tuple]))

        num_tokens = r.num_flat_tokens

        def restore_model(name: str):
            match cfg.model_type:
                case "dit":
                    return LightningDiT.restore(
                        restore_mngr,
                        step,
                        name,
                        mesh,
                        cfg.dit,
                        mp,
                        num_tokens=num_tokens,
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                    )
                case "dit_dh":
                    return DiTDH.restore(
                        restore_mngr,
                        step,
                        name,
                        mesh,
                        cfg.dit_dh,
                        mp,
                        latent_mean=latent_mean,
                        latent_std=latent_std,
                    )
                case _:
                    raise ValueError(f"Unsupported model_type: {cfg.model_type}")

        generator = restore_model("model")
        generator_ema = restore_model("model_ema")

        # Warn if restored model already has stats but experiment name includes -norm
        if cfg.use_latent_norm and hasattr(generator, "latent_mean"):
            logging.warning(
                "Restored model already has latent normalization stats. "
                "The -norm flag in experiment name will override them with the newly loaded stats."
            )
    else:
        # Set generator input dimensionality from the FlatDINO encoder latents so the
        # DiT/DiT-DH sees the correct feature size (and persists this in the config).
        if latent_dim is not None:
            logging.warning(
                "The VAE latent dim does not match with the generator config. "
                "Overwriting the config to match the encoder latent dim."
            )
            cfg = replace(
                cfg,
                dit=replace(cfg.dit, in_channels=latent_dim),
                dit_dh=replace(cfg.dit_dh, in_channels=latent_dim),
            )
        num_tokens = r.num_flat_tokens
        if cfg.model_type == "dit_dh":
            dit_dh_cfg = cfg.dit_dh
            if dit_dh_cfg.patch_size is None:
                dit_dh_cfg = replace(dit_dh_cfg, input_size=num_tokens)
            cfg = replace(cfg, dit_dh=dit_dh_cfg)
        generator = make_generator(
            cfg, mp, rngs, num_tokens=num_tokens, latent_mean=latent_mean, latent_std=latent_std
        )
        generator_ema = make_generator(
            cfg, mp, rngs, num_tokens=num_tokens, latent_mean=latent_mean, latent_std=latent_std
        )
        nnx.update(generator_ema, jax.tree.map(lambda x: jnp.copy(x), nnx.state(generator)))

    # If a checkpoint manager is provided, it will restore the model weights from the checkpoint
    assert r.aug_cfg.image_size[0] == r.aug_cfg.image_size[1]

    # Set attention implementation
    set_attn_implementation(generator, args.implementation)
    set_attn_implementation(generator_ema, args.implementation)
    logging.info(f"Attention implementation set to: {args.implementation}")

    chain = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(lr_sched, cfg.train.adam_b1, cfg.train.adam_b2)
    )
    chain = optax.MultiSteps(chain, grad_acc_steps)
    optim = nnx.Optimizer(generator, chain, wrt=nnx.Param)  # ty: ignore

    # Retrieve train state from checkpoint (optim and loader)
    if restore_mngr is not None:
        restore_optimizer_state(restore_mngr, step, optim, mesh)
        data_iter = restore_data_loader(restore_mngr, step, data_iter)

    generator_params = jax.tree.leaves(nnx.state(generator, nnx.Param))
    generator_param_count = sum(jax.tree.map(lambda x: jnp.size(x), generator_params))
    logging.info(f"generator params: {generator_param_count / 1_000_000:.2f}M params")

    if save_path is not None:
        # Convert to string path (handles both local Path and GCS string paths)
        if isinstance(save_path, Path):
            path_str = str(save_path.absolute())
        else:
            path_str = save_path

        # Always save at the final step to ensure we don't lose training progress
        save_on_steps = frozenset([total_updates])

        # Convert epochs to steps for checkpoints_to_keep
        # - Add to save_on_steps to ensure they are saved at those steps
        # - Add to preservation_policy to ensure they are not deleted
        keep_steps: set[int] | None = None
        if args.checkpoints_to_keep is not None:
            keep_steps = set(epoch * steps_per_epoch for epoch in args.checkpoints_to_keep)
            save_on_steps = save_on_steps | keep_steps
            logging.info(f"Checkpoints to keep (epochs): {args.checkpoints_to_keep}")
            logging.info(f"Checkpoints to keep (steps): {sorted(keep_steps)}")

        # Build preservation policy using the modern API
        # Combine multiple policies: keep latest N, keep best by gFID, and keep custom steps
        policies: list[pp.PreservationPolicy] = []
        if not args.keep_all_checkpoints:
            policies.append(pp.LatestN(1))
        else:
            policies.append(pp.PreserveAll())
        policies.append(
            pp.BestN(
                n=1,
                get_metric_fn=lambda metrics: metrics["val/gfid"],
                reverse=False,  # Lower gFID is better
                keep_checkpoints_without_metrics=args.keep_checkpoints_without_metrics,
            )
        )
        if keep_steps is not None:
            policies.append(pp.CustomSteps(keep_steps))

        preservation_policy = pp.AnyPreservationPolicy(policies)

        opts = ocp.CheckpointManagerOptions(
            save_interval_steps=save_interval_steps,
            save_on_steps=save_on_steps,
            create=True,
            read_only=False,
            preservation_policy=preservation_policy,
        )
        mngr = ocp.CheckpointManager(path_str, options=opts, item_names=item_names)
    else:
        mngr = None

    if cfg.time_dist_shift_base is not None:
        latent_tokens = r.num_flat_tokens
        latent_dim = generator.cfg.in_channels
        time_dist_shift_dim = latent_tokens * latent_dim
        if cfg.kappa is None:
            kappa = (time_dist_shift_dim / cfg.time_dist_shift_base) ** 0.5
        else:
            kappa = cfg.kappa
        logging.info(f"time_dist_shift_dim: {time_dist_shift_dim}")
        logging.info(f"kappa: {kappa}")
    else:
        kappa = None

    train_stream = prefetch_to_mesh(data_iter, 1, mesh)

    if args.overfit_one_image:
        try:
            first_batch = next(train_stream)
        except StopIteration as exc:
            raise ValueError("Training loader is empty; cannot overfit to a single image.") from exc

        def _repeat_first_entry(x):
            if not hasattr(x, "ndim") or x.ndim == 0:
                return x
            first = jax.lax.index_in_dim(x, 0, axis=0, keepdims=True)
            return jnp.broadcast_to(first, x.shape)

        overfit_batch = jax.tree.map(_repeat_first_entry, first_batch)

        # Save overfit comparison locally (skip for GCS paths)
        is_gcs = isinstance(save_path, str) and save_path.startswith("gs://")
        if not is_gcs:
            comparison_dir = (
                Path(save_path) if save_path else Path("output")
            ) / "overfit_one_image"
            _save_overfit_comparison(first_batch, r, comparison_dir, args.seed, mp)

        def _repeat_forever(batch):
            while True:
                yield batch

        train_stream = _repeat_forever(overfit_batch)

    updates_completed = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
    pbar = tqdm(
        desc="Update",
        initial=updates_completed,
        total=total_updates,
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        disable=not is_primary_host(),
    )

    train_step_cached = nnx.cached_partial(
        train_step, generator, generator_ema, r.dino, r.encoder, optim
    )
    profiler = TrainingProfiler(
        mode=args.profile_mode,
        port=args.profiler_port,
        start_step=args.profiler_start_step,
        stop_step=args.profiler_stop_step,
    )
    for samples in train_stream:
        mini_step = int(optax.tree_utils.tree_get(optim.opt_state, "mini_step"))
        should_update_ema = (mini_step + 1) % grad_acc_steps == 0
        train_loss = train_step_cached(
            samples["image"],
            samples["label"],
            rngs(),
            cfg.train.ema,
            should_update_ema=should_update_ema,
            kappa=kappa,
            model_type=cfg.model_type,
            pred_type=cfg.pred_type,
            sample_latents=cfg.sample_latents,
            xpred_steps=args.gfid_steps,
            num_latents=r.num_flat_tokens,
            encoder_disposable_registers=r.encoder_disposable_registers,
            tanh=r.tanh,
        )
        metrics = {}

        prev_updates = updates_completed
        updates_completed = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
        ran_update = updates_completed > prev_updates
        profiler.step(updates_completed)
        if ran_update:
            pbar.update(updates_completed - prev_updates)

            # Check preemption FIRST - if preempted, save immediately without validation
            is_preemption = mngr.reached_preemption(updates_completed) if mngr else False

            if is_preemption:
                # Preemption detected - save checkpoint immediately without validation
                logging.info(
                    "Preemption detected at step %d. Saving checkpoint immediately...",
                    updates_completed,
                )
                jax.block_until_ready(train_loss)
                save_checkpoint(
                    mngr, updates_completed, generator, generator_ema, optim, data_iter, cfg
                )
                mngr.wait_until_finished()
                logging.info("Checkpoint saved. Exiting due to preemption.")
                break  # Exit the training loop

            # Check if we should save a checkpoint at this step (regular interval)
            should_save_now = mngr.should_save(updates_completed) if mngr else False

            # Run validation only for regular saves (not preemption)
            if should_save_now:
                jax.block_until_ready(train_loss)

                if args.skip_val_gfid:
                    # Save checkpoint without validation metrics
                    save_checkpoint(
                        mngr,
                        updates_completed,
                        generator,
                        generator_ema,
                        optim,
                        data_iter,
                        cfg,
                        None,
                    )
                else:
                    val_metrics = run_validation(
                        generator_ema,
                        cfg,
                        args,
                        mesh,
                        mp,
                        r,
                        kappa,
                        gfid_batch_size,
                        save_path,
                        updates_completed,
                        use_wandb,
                    )
                    metrics.update(val_metrics)

                    # Save checkpoint with validation metrics
                    save_checkpoint(
                        mngr,
                        updates_completed,
                        generator,
                        generator_ema,
                        optim,
                        data_iter,
                        cfg,
                        {"val/gfid": val_metrics["val/gfid"]},
                    )

        if use_wandb:
            if ran_update and updates_completed % args.wandb_log_every == 0:
                optim_step = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
                metrics["train/lr"] = lr_sched(optim_step)
                metrics["train/loss"] = float(train_loss)

            if metrics:
                metrics["step"] = updates_completed
                wandb.log(metrics)

        if updates_completed >= total_updates:
            break

    pbar.close()

    if mngr is not None:
        mngr.close()
    if use_wandb:
        wandb.finish()


class ExperimentSpec(NamedTuple):
    """Parsed experiment specification for generator training."""

    size: str = "b"  # s, b, l, xl
    model_type: str = "dit"  # dit, dit_dh
    kappa: float | None = None
    pred_type: str = "v"  # v, x
    use_latent_norm: bool = False


def parse_experiment(name: str) -> ExperimentSpec:
    """Parse experiment name like 'ditb-dh-kappa2.5-xpred-norm' into spec.

    Format: dit{size}[-dh][-kappa{float}][-xpred][-norm]

    Examples:
        dits                    # small dit
        ditb                    # base dit
        ditl-dh                 # large dit with double head
        ditxl-kappa3            # extra large dit with kappa=3
        ditb-dh-kappa2.5        # base dit with double head and kappa=2.5
        dits-kappa0.5           # small dit with kappa=0.5
        ditb-xpred              # base dit with x-prediction
        ditl-dh-kappa2-xpred    # large dit with double head, kappa=2, x-prediction
        ditb-norm               # base dit with latent normalization
        ditl-dh-kappa2-norm     # large dit with double head, kappa=2, latent normalization
    """
    pattern = r"""
        ^dit(?P<size>s|b|l|xl)
        (?P<dh>-dh)?
        (?:-kappa(?P<kappa>[\d.]+))?
        (?P<xpred>-xpred)?
        (?P<norm>-norm)?
        $
    """
    match = re.match(pattern, name, re.VERBOSE)
    if not match:
        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Expected format: dit{{size}}[-dh][-kappa{{float}}][-xpred][-norm] "
            f"(e.g., 'ditb', 'ditl-dh', 'ditxl-kappa2.5', 'dits-norm')"
        )

    kappa = float(match.group("kappa")) if match.group("kappa") else None
    model_type = "dit_dh" if match.group("dh") else "dit"
    pred_type = "x" if match.group("xpred") else "v"
    use_latent_norm = bool(match.group("norm"))

    return ExperimentSpec(
        size=match.group("size"),
        model_type=model_type,
        kappa=kappa,
        pred_type=pred_type,
        use_latent_norm=use_latent_norm,
    )


def build_config(spec: ExperimentSpec) -> "GeneratorConfig":
    """Build GeneratorConfig from parsed experiment spec."""
    baseline = GeneratorConfig()
    size_key = f"dit-{spec.size}"

    if spec.model_type == "dit":
        cfg = replace(
            baseline,
            model_type="dit",
            kappa=spec.kappa,
            pred_type=spec.pred_type,
            use_latent_norm=spec.use_latent_norm,
            dit=replace(
                baseline.dit,
                transformer=replace(
                    baseline.dit.transformer,
                    **LIGHTNING_DIT_CONFIGS[size_key],
                ),
                in_channels=768,
            ),
        )
    else:  # dit_dh
        cfg = replace(
            baseline,
            model_type="dit_dh",
            kappa=spec.kappa,
            pred_type=spec.pred_type,
            use_latent_norm=spec.use_latent_norm,
            dit_dh=replace(
                baseline.dit_dh,
                in_channels=768,
                encoder_dim=LIGHTNING_DIT_CONFIGS[size_key]["embed_dim"],
                encoder_heads=LIGHTNING_DIT_CONFIGS[size_key]["num_heads"],
                encoder_depth=LIGHTNING_DIT_CONFIGS[size_key]["num_layers"],
                mlp_ratio=LIGHTNING_DIT_CONFIGS[size_key]["mlp_hidden_dim"]
                / LIGHTNING_DIT_CONFIGS[size_key]["embed_dim"],
            ),
        )

    return cfg


def get_experiment(name: str) -> "GeneratorConfig":
    """Get experiment config by name."""
    spec = parse_experiment(name)
    return build_config(spec)


if __name__ == "__main__":
    args: Args = tyro.cli(Args)

    if args.debug:
        logging.set_verbosity(logging.DEBUG)
    else:
        logging.set_verbosity(logging.INFO)
    logging.get_absl_handler().use_absl_log_file()

    config = get_experiment(args.experiment)
    main(args, config)
