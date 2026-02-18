from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Literal, NamedTuple
import re

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
import grain.python as grain
import jmp
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp
import chex
import tyro
import wandb
from tqdm import tqdm
from dacite import from_dict, Config as DaciteConfig
from absl import logging

from flatdino.data import DataLoaders, create_dataloaders
from flatdino.models.vit import TransformerConfig, ViTConfig, VIT_CONFIGS
from flatdino.models.transformer import set_attn_implementation
from flatdino.pretrained import DinoWithRegisters
from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig, OptimConfig
from flatdino.augmentations import FlatDinoTrainAugmentations, FlatDinoValAugmentations
from flatdino.eval import save_eval_results
from flatdino.utils import build_lr_schedule
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

# jax.config.update("jax_debug_nans", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@dataclass
class Args:
    seed: int = 42
    restore: Path | Literal["default"] | None = None
    """Path to restore checkpoint from. Use 'default' to restore from the default
    checkpoint path (output/flatdino/vae/{experiment} or GCS equivalent if gcs_bucket is set)."""
    maybe_restore: Path | Literal["default"] | None = None
    """Like --restore, but if no checkpoint exists at the path, logs a warning and trains
    from scratch instead of crashing. Useful for resumable training scripts."""
    checkpoint: bool = True
    checkpoint_dir: Path | None = None
    keep_checkpoints_without_metrics: bool = True
    """If True, preemption checkpoints (which have no validation metrics) are preserved
    indefinitely. If False, they are cleaned up once we have enough checkpoints with metrics."""
    gcs_bucket: str | None = None
    """GCS bucket name for checkpoints. When set, checkpoints are saved directly to
    gs://{gcs_bucket}/output/... using orbax's native GCS support, avoiding gcsfuse
    consistency issues in multi-host training. Requires running on a GCP instance."""
    data_in_bucket: bool = False
    """When True and gcs_bucket is set, load datasets from gs://{gcs_bucket}/{dataset_name}
    via tfds instead of from local storage."""
    num_data_workers: int | None = None
    """Override the number of data loading workers. If None, uses the value from DataConfig."""
    use_wandb: bool = False
    wandb_log_every: int = 20
    project_name: str = "flatdino"
    wandb_name: str | None = None
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
    experiment: str = "baseline"
    implementation: Literal["cudnn", "xla"] = "xla"
    """Attention implementation: 'cudnn' for Flash Attention, 'xla' for default."""
    fsdp: int = 1
    """FSDP (Fully Sharded Data Parallelism) axis size. When fsdp=1, only data parallelism
    is used. When fsdp>1, model parameters are sharded across fsdp devices using the
    Megatron-LM pattern. The data axis size is num_devices // fsdp."""
    distributed: bool = False
    """Enable distributed training for multi-host setups (e.g., TPU pods). Must be set
    before any JAX operations occur."""


def loss_fn(
    key: jax.Array,
    flatdino: FlatDinoAutoencoder,
    x: jax.Array,
    recon_weight: float,
    kl_weight: float,
    free_bits: float = 0.0,
    kl_var_weight: float = 0.0,
    x_intermediate: jax.Array | None = None,
    intermediate_weight: float = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    z, aux = flatdino.encode(x, key=key)
    mu, logvar = aux["mu"], aux["logvar"]

    # Request intermediate output only when we have intermediate targets
    if x_intermediate is not None:
        decoded = flatdino.decode(z, return_intermediate=True)
        x_hat = decoded["final"]
        x_hat_intermediate = decoded["intermediate"]
    else:
        x_hat = flatdino.decode(z)
        x_hat_intermediate = None

    # Reconstruction loss (L2/Huber)
    recon_loss = jnp.mean(optax.huber_loss(x, x_hat))

    # Intermediate reconstruction loss (if enabled)
    if x_intermediate is not None and x_hat_intermediate is not None:
        intermediate_loss = jnp.mean(optax.huber_loss(x_intermediate, x_hat_intermediate))
    else:
        intermediate_loss = jnp.array(0.0)

    # KL divergence loss
    std = jnp.exp(0.5 * logvar)
    kl_per_dim = 0.5 * (std**2 + mu**2 - 1.0 - logvar)
    kl_nats = jnp.mean(kl_per_dim)
    # Apply free bits: only penalize KL above the threshold per dimension
    kl_per_dim_clipped = jnp.maximum(kl_per_dim, free_bits)
    kl_loss = jnp.mean(jnp.sum(kl_per_dim_clipped, axis=(-1, -2)))

    # Per-token KL: sum over features -> shape (batch, tokens)
    kl_per_token_per_batch = jnp.sum(kl_per_dim_clipped, axis=-1)
    # Mean over batch -> shape (tokens,) for logging
    kl_per_token = jnp.mean(kl_per_token_per_batch, axis=0)

    # KL variance loss: variance across tokens, mean over batch
    # Encourages balanced KL distribution across tokens
    kl_variance = jnp.mean(jnp.var(kl_per_token_per_batch, axis=-1))

    total_recon = recon_loss + intermediate_weight * intermediate_loss
    loss = (
        recon_weight * total_recon + kl_weight * kl_loss + kl_weight * kl_var_weight * kl_variance
    )

    metrics = {
        "recon_loss": recon_loss,
        "intermediate_loss": intermediate_loss,
        "kl_loss": kl_loss,
        "kl_nats": kl_nats,
        "kl_variance": kl_variance,
        # Posterior statistics (detect collapse/explosion)
        "logvar_mean": jnp.mean(logvar),
        "logvar_min": jnp.min(logvar),
        "logvar_max": jnp.max(logvar),
        "std_mean": jnp.mean(std),
        "mu_sq_mean": jnp.mean(mu**2),
        # Latent code statistics
        "z_var": jnp.var(z),
        # Reconstruction error
        "recon_mse": jnp.mean((x - x_hat) ** 2),
        # Per-token KL losses
        "kl_per_token": kl_per_token,
    }
    return loss, metrics


@nnx.jit(
    donate_argnames=("flatdino", "optim"),
    static_argnames=("free_bits", "kl_var_weight", "intermediate_layer", "intermediate_weight"),
)
def train_step(
    flatdino: FlatDinoAutoencoder,
    dino: DinoWithRegisters,
    optim: nnx.Optimizer,
    key: jax.Array,
    x: jax.Array,
    recon_weight: float,
    kl_weight: float,
    free_bits: float = 0.0,
    kl_var_weight: float = 0.0,
    intermediate_layer: int | None = None,
    intermediate_weight: float = 1.0,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    chex.assert_rank(x, 4)
    b, h, w, d = x.shape

    x = jax.image.resize(x, (b, 224, 224, d), method="bilinear")

    # Extract DINO features (and intermediate if configured)
    if intermediate_layer is not None:
        dino_out, intermediates = dino(x, layers=[intermediate_layer])
        dino_patches = dino_out[:, 5:]
        dino_intermediate = intermediates[0][:, 5:]
    else:
        dino_patches = dino(x)[:, 5:]
        dino_intermediate = None

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=1, has_aux=True)(
        key,
        flatdino,
        dino_patches,
        recon_weight,
        kl_weight,
        free_bits,
        kl_var_weight,
        dino_intermediate,
        intermediate_weight,
    )
    optim.update(flatdino.trainable_pytree, grads)
    return loss, metrics


@nnx.jit
def val_step(
    flatdino: FlatDinoAutoencoder,
    dino: DinoWithRegisters,
    x: jax.Array,
) -> jax.Array:
    # We return the sum because we expect the val loop to average over the
    # sum of all batch sizes
    chex.assert_rank(x, 4)
    b, _, _, d = x.shape
    x = jax.image.resize(x, (b, 224, 224, d), method="bilinear")
    dino_patches = dino(x)[:, 5:]
    z, _ = flatdino.encode(dino_patches)  # deterministic (no key)
    x_hat = flatdino.decode(z)  # Always returns final patches

    return jnp.sum(jnp.mean(optax.huber_loss(dino_patches, x_hat), axis=(1, 2)))


def run_validation(
    flatdino: FlatDinoAutoencoder,
    dino: DinoWithRegisters,
    val_loader,
    val_iters: int,
    mesh,
) -> dict[str, float]:
    """Run validation loop and compute metrics.

    Returns:
        Dictionary with "val/loss".
    """
    val_iter = iter(val_loader)
    total_samples = 0
    loss = 0.0
    for val_batch in tqdm(
        prefetch_to_mesh(val_iter, 1, mesh),
        desc="eval",
        total=val_iters,
        leave=False,
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        disable=not is_primary_host(),
    ):
        imgs = val_batch["image"]
        loss += val_step(flatdino, dino, imgs).item()
        total_samples += imgs.shape[0]

    val_metrics = {"val/loss": float(loss / total_samples)}
    return val_metrics


def save_checkpoint(
    manager: ocp.CheckpointManager | None,
    step: int,
    flatdino: FlatDinoAutoencoder,
    optim: nnx.Optimizer,
    data_iter,
    cfg: FlatDinoConfig,
    metrics: dict | None = None,
) -> bool:
    """Save checkpoint if the manager decides it should be saved.

    Returns True if a checkpoint was actually saved.

    Note: This function is preemption-safe. Call it at every gradient step and
    orbax will decide whether to actually save based on save_interval_steps and
    preemption signals.
    """
    if manager is None:
        return False
    state = flatdino.get_state()
    saved = manager.save(
        step,
        metrics=metrics,
        args=ocp.args.Composite(
            optim=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim))),
            loader=grain.PyGrainCheckpointSave(data_iter),
            config=ocp.args.JsonSave(asdict(cfg)),
            **{name: ocp.args.PyTreeSave(tree) for name, tree in state.items()},
        ),
    )
    return saved


def create_checkpoint_managers(
    save_path: Path | str | None,
    item_names: list[str],
    wsd_decay_start: int | None,
    save_interval_steps: int,
    total_steps: int,
    keep_checkpoints_without_metrics: bool = True,
) -> tuple[ocp.CheckpointManager | None, ocp.CheckpointManager | None]:
    """Create checkpoint managers with preemption-safe settings.

    Args:
        save_path: Directory to save checkpoints. Can be a local Path or a GCS path
            string (e.g., "gs://bucket/path").
        item_names: Names of items to checkpoint.
        wsd_decay_start: Step at which WSD decay starts (for pre-decay checkpoint).
        save_interval_steps: Interval between regular checkpoints. Should match
            validation frequency so that checkpoints have metrics for best_fn.
        total_steps: Total number of training steps. Used to ensure final checkpoint
            is saved even if it doesn't fall on a save_interval_steps boundary.
        keep_checkpoints_without_metrics: If True, preemption checkpoints (without
            metrics) are preserved indefinitely. If False, they are cleaned up once
            we have enough checkpoints with metrics.

    The main checkpoint manager uses:
        - save_interval_steps: Saves at regular intervals (with validation metrics)
        - save_on_steps: Includes total_steps to ensure final checkpoint is saved
        - max_to_keep=2: Keeps best checkpoint + one more for preemption safety
        - best_fn: Tracks val/loss to keep the best checkpoint

    Preemption handling:
        When preemption is detected, we save immediately WITHOUT running validation
        (to save as fast as possible). These checkpoints have no metrics.

        The max_to_keep=2 ensures we keep:
        1. The best checkpoint (by val/loss)
        2. One additional checkpoint (for safety during preemption recovery)
    """
    if save_path is None:
        return None, None

    # Convert to string path (handles both local Path and GCS string paths)
    if isinstance(save_path, Path):
        path_str = str(save_path.absolute())
    else:
        path_str = save_path

    # Always save at the final step to ensure we don't lose training progress
    save_on_steps = frozenset([total_steps])

    opts = ocp.CheckpointManagerOptions(
        save_interval_steps=save_interval_steps,
        save_on_steps=save_on_steps,
        max_to_keep=2,
        create=True,
        read_only=False,
        best_fn=lambda metrics: metrics["val/loss"],
        best_mode="min",
        keep_checkpoints_without_metrics=keep_checkpoints_without_metrics,
    )
    mngr = ocp.CheckpointManager(path_str, options=opts, item_names=item_names)

    pre_decay_mngr = None
    if wsd_decay_start is not None:
        # Construct pre-decay path (works for both local and GCS paths)
        if path_str.startswith("gs://"):
            pre_decay_path = path_str.rstrip("/") + "-before-decay"
        else:
            pre_decay_path = str(Path(path_str).with_name(Path(path_str).name + "-before-decay"))
        pre_decay_opts = ocp.CheckpointManagerOptions(
            save_on_steps=frozenset([wsd_decay_start]),
            max_to_keep=1,
            create=True,
            read_only=False,
        )
        pre_decay_mngr = ocp.CheckpointManager(
            pre_decay_path, options=pre_decay_opts, item_names=item_names
        )

    return mngr, pre_decay_mngr


def main(args: Args, cfg: FlatDinoConfig):
    # Initialize distributed JAX for multi-host setups (e.g., TPU pods)
    # Must be called before any JAX operations
    if args.distributed:
        init_distributed()

    # Silence non-primary hosts to avoid duplicate log messages
    if not is_primary_host():
        logging.set_verbosity(logging.WARNING)

    # Create mixed precision policy (must be after distributed init)
    mp_policy = jmp.Policy(
        param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32
    )

    target_cfg = cfg
    # Use different RNG seeds per host for data augmentation diversity
    rngs = nnx.Rngs(args.seed + jax.process_index())

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

    item_names = FlatDinoAutoencoder.get_item_names() + ["optim", "loader", "config"]

    # Open restore manager (handles both --restore and --maybe-restore)
    restore_mngr, ckpt_step = open_restore_manager(
        args.restore,
        args.maybe_restore,
        default_path=f"output/flatdino/vae/{args.experiment}",
        gcs_bucket=args.gcs_bucket,
        item_names=item_names,
    )

    if restore_mngr is not None:
        cfg_d = restore_mngr.restore(
            ckpt_step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
        )["config"]
        ckpt_cfg = from_dict(FlatDinoConfig, cfg_d, DaciteConfig(cast=[tuple], strict=False))
        # Preserve the new experiment's training hyperparams while keeping checkpoint-compatible model/data.
        if ckpt_cfg.train != target_cfg.train:
            logging.warning(
                "Restoring model/data from checkpoint but overriding training config "
                "with current experiment; schedules/epochs/optimizer settings will follow "
                "the new config while architecture/data stay as in the checkpoint."
            )
        if ckpt_cfg.train.batch_size != target_cfg.train.batch_size:
            raise ValueError(
                "Restoring with a different train.batch_size is not supported; "
                "checkpointed optimizer/loader state assumes the saved global batch size."
            )
        cfg = replace(ckpt_cfg, train=target_cfg.train)

    # Only primary host logs to wandb
    use_wandb = args.use_wandb and is_primary_host()
    if use_wandb:
        name = args.wandb_name or args.experiment
        wandb.init(project=args.project_name, name=name, config=asdict(cfg))

    # With FSDP, only data_parallel_size contributes to batch scaling
    # (model parallel devices share the same batch)
    assert cfg.train.batch_size % (args.gpu_batch_size * data_parallel_size) == 0
    grad_acc_steps = cfg.train.batch_size // (args.gpu_batch_size * data_parallel_size)
    micro_bs = args.gpu_batch_size * data_parallel_size
    logging.info(f"Gradient accumulation steps: {grad_acc_steps}")
    logging.info(f"Micro batch size: {micro_bs}")

    data_cfg = cfg.data
    if args.num_data_workers is not None:
        data_cfg = replace(data_cfg, num_workers=args.num_data_workers)

    data: DataLoaders = create_dataloaders(
        data_cfg,
        micro_bs,
        train_aug=FlatDinoTrainAugmentations(cfg.aug, cfg.data),
        val_aug=FlatDinoValAugmentations(cfg.aug, cfg.data),
        val_epochs=1,
        drop_remainder_train=True,
        drop_remainder_val=True,
        gcs_bucket=args.gcs_bucket if args.data_in_bucket else None,
    )
    train_loader = data.train_loader
    val_loader = data.val_loader
    total_updates = (data.train_ds_size * cfg.train.epochs) // cfg.train.batch_size
    wsd_decay_start = None
    if cfg.train.lr_schedule == "wsd":
        steps_per_epoch = total_updates // cfg.train.epochs
        warmup_steps = steps_per_epoch * cfg.train.warmup_epochs
        decay_steps = steps_per_epoch * (
            cfg.train.decay_epochs
            if cfg.train.decay_epochs is not None
            else cfg.train.warmup_epochs
        )
        remaining_steps = max(total_updates - warmup_steps, 0)
        decay_steps = min(decay_steps, remaining_steps)
        wsd_decay_start = total_updates - decay_steps
    else:
        warmup_steps = None
        decay_steps = None

    val_iters = (data.val_ds_size + micro_bs - 1) // micro_bs
    data_iter = iter(train_loader)

    lr_sched = build_lr_schedule(
        cfg.train,
        total_updates,
        warmup_steps_override=warmup_steps,
        decay_steps_override=decay_steps,
    )

    save_path = determine_save_path(
        checkpoint_enabled=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        default_path=f"output/flatdino/vae/{args.experiment}",
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

    # If a checkpoint manager is provided, it will restore the model weights from the checkpoint
    assert cfg.aug.image_size[0] == cfg.aug.image_size[1]
    dino = DinoWithRegisters(cfg.dino_name, resolution=cfg.aug.image_size[0])
    flatdino = FlatDinoAutoencoder(
        cfg, mesh, mngr=restore_mngr, step=ckpt_step, mp=mp_policy, rngs=rngs
    )

    # Set attention implementation
    set_attn_implementation(flatdino, args.implementation)
    logging.info(f"Attention implementation set to: {args.implementation}")

    # Weight decay mask: exclude embeddings and non-2D params (biases, layer norms)
    def wd_mask_fn(path: str, param: nnx.Variable) -> bool:
        # Exclude position embeddings, cls token, register tokens
        if path in ["pos_embed", "cls_token", "reg_tokens"]:
            return False
        # Only apply weight decay to 2D params (weight matrices)
        if param[...].ndim != 2:
            return False
        return True

    wd_mask = nnx.map_state(wd_mask_fn, flatdino.trainable_state)
    chain = optax.chain(
        optax.clip_by_global_norm(10.0),
        optax.adamw(
            lr_sched,
            cfg.train.adam_b1,
            cfg.train.adam_b2,
            weight_decay=cfg.train.weight_decay,
            mask=wd_mask,
        ),
    )
    chain = optax.MultiSteps(chain, grad_acc_steps)
    optim = nnx.Optimizer(flatdino.trainable_pytree, chain, wrt=nnx.Param)  # ty: ignore

    # Retrieve train state from checkpoint (optim and loader)
    # Retrieve train state from checkpoint (optim and loader)
    if restore_mngr is not None:
        restore_optimizer_state(restore_mngr, ckpt_step, optim, mesh)
        data_iter = restore_data_loader(restore_mngr, ckpt_step, data_iter)

    encoder_params = jax.tree.leaves(nnx.state(flatdino.encoder, nnx.Param))
    encoder_param_count = sum(jax.tree.map(lambda x: jnp.size(x), encoder_params))
    decoder_params = jax.tree.leaves(nnx.state(flatdino.decoder, nnx.Param))
    decoder_param_count = sum(jax.tree.map(lambda x: jnp.size(x), decoder_params))

    logging.info(f"num tokens: {flatdino.encoder.num_reg}")
    logging.info(f"Encoder ViT: {encoder_param_count / 1_000_000:.2f}M params")
    logging.info(f"Decoder ViT: {decoder_param_count / 1_000_000:.2f}M params")

    mngr, pre_decay_mngr = create_checkpoint_managers(
        save_path,
        item_names,
        wsd_decay_start,
        save_interval_steps,
        total_steps=total_updates,
        keep_checkpoints_without_metrics=args.keep_checkpoints_without_metrics,
    )

    updates_completed = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
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
    final_val_metrics: dict[str, float] = {}  # Track final validation metrics for JSON

    train_step_cached = nnx.cached_partial(train_step, flatdino, dino, optim)

    # Compute kl_weight if not set (for static experiments like "baseline")
    if cfg.train.kl_weight is not None:
        base_kl_weight = cfg.train.kl_weight
    else:
        tokens = cfg.num_latents  # Use actual latent count (excluding disposable registers)
        features = cfg.encoder.output_dim // 2  # output_dim = 2 * features (mu + logvar)
        base_kl_weight = compute_default_kl_weight(tokens, features)

    def get_kl_weight(step: int) -> float:
        if cfg.train.kl_anneal_steps is None:
            return base_kl_weight
        anneal_factor = min(1.0, step / cfg.train.kl_anneal_steps)
        return base_kl_weight * anneal_factor

    logging.info(f"base_kl_weight: {base_kl_weight}")

    free_bits = cfg.train.free_bits if cfg.train.free_bits is not None else 0.0
    kl_var_weight = cfg.train.kl_var_weight if cfg.train.kl_var_weight is not None else 0.0
    if kl_var_weight > 0:
        logging.info(f"kl_var_weight: {kl_var_weight}")

    intermediate_layer = cfg.intermediate_layer
    intermediate_weight = cfg.intermediate_weight
    if intermediate_layer is not None:
        logging.info(f"intermediate_layer: {intermediate_layer}, weight: {intermediate_weight}")

    for samples in prefetch_to_mesh(data_iter, 1, mesh):
        imgs = samples["image"]

        prev_updates = updates_completed
        kl_weight_effective = get_kl_weight(updates_completed)
        train_loss, train_metrics = train_step_cached(
            rngs(),
            imgs,
            cfg.train.recon_weight,
            kl_weight_effective,
            free_bits,
            kl_var_weight,
            intermediate_layer,
            intermediate_weight,
        )
        metrics = {}

        updates_completed = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
        ran_update = updates_completed > prev_updates

        if ran_update:
            pbar.update(updates_completed - prev_updates)
            profiler.step(updates_completed)

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
                    mngr, updates_completed, flatdino, optim, data_iter, cfg, metrics=None
                )
                mngr.wait_until_finished()
                logging.info("Checkpoint saved. Exiting due to preemption.")
                break  # Exit the training loop

            # Check if we should save a checkpoint at this step (regular interval)
            should_save_now = mngr.should_save(updates_completed) if mngr else False

            # Run validation only for regular saves (not preemption)
            if should_save_now:
                jax.block_until_ready(train_loss)
                val_metrics = run_validation(
                    flatdino,
                    dino,
                    val_loader,
                    val_iters,
                    mesh,
                )

                # Track final validation metrics for saving to JSON at end of training
                final_val_metrics = val_metrics.copy()
                metrics.update(val_metrics)

                # Save checkpoint with validation metrics
                save_checkpoint(
                    mngr, updates_completed, flatdino, optim, data_iter, cfg, val_metrics
                )

                # Also save to pre-decay manager at the WSD decay start step
                if pre_decay_mngr is not None:
                    save_checkpoint(
                        pre_decay_mngr, updates_completed, flatdino, optim, data_iter, cfg
                    )

        if use_wandb:
            if ran_update and updates_completed % args.wandb_log_every == 0:
                metrics["train/lr"] = lr_sched(updates_completed)
                metrics["train/kl_weight"] = kl_weight_effective
                metrics["train/loss"] = float(train_loss)
                for name, value in train_metrics.items():
                    if name == "kl_per_token":
                        # Expand per-token KL array into individual metrics
                        for i, kl_val in enumerate(value):
                            metrics[f"train/kl_per_token/{i}"] = float(kl_val)
                    else:
                        metrics[f"train/{name}"] = float(value)

            if metrics:
                metrics["step"] = updates_completed
                wandb.log(metrics)

        if updates_completed >= total_updates:
            break

    pbar.close()

    # Save final validation metrics to eval_results.json (only on primary host)
    # Skip for GCS paths since save_eval_results uses local file operations
    is_gcs_path = isinstance(save_path, str) and save_path.startswith("gs://")
    if save_path is not None and final_val_metrics and is_primary_host() and not is_gcs_path:
        training_results = {
            "final_step": updates_completed,
            "total_steps": total_updates,
            "epochs": cfg.train.epochs,
        }
        if "val/loss" in final_val_metrics:
            training_results["val_loss"] = final_val_metrics["val/loss"]
        save_eval_results(save_path, "training", training_results)
        logging.info(f"Saved training results to {save_path}/eval_results.json")

    if mngr is not None:
        mngr.close()
    if pre_decay_mngr is not None:
        pre_decay_mngr.close()
    if use_wandb:
        wandb.finish()


baseline = FlatDinoConfig(
    dino_name="facebook/dinov2-with-registers-base",
    train=OptimConfig(batch_size=512, epochs=100),
    encoder=ViTConfig(
        patch=None,
        use_pos_embeds=False,  # TODO: (try with) since dino already encodes this
        num_patches=256,
        input_dim=768,
        num_registers=32,
        transformer=TransformerConfig(
            embed_dim=768, num_layers=6, mlp_hidden_dim=3072, num_heads=12, selective=True
        ),
        output_dim=768 * 2,
    ),
    decoder=ViTConfig(
        patch=None,
        use_pos_embeds=True,
        num_patches=32,  # The register inputs
        output_dim=768,
        num_registers=256,  # These are the masks, which should match the DINO patches
        transformer=TransformerConfig(
            embed_dim=768, num_layers=6, mlp_hidden_dim=3072, num_heads=12, selective=True
        ),
    ),
)


def compute_default_kl_weight(tokens: int, features: int) -> float:
    """Compute default KL weight normalized by latent dimension.

    Uses reference: 32 tokens × 16 features = 512 dim at kl_weight = 1e-6.
    Scales inversely with latent dimension to keep per-dimension KL penalty constant.

    Formula: kl_weight = 1e-6 * (512 / (tokens * features))
    """
    ref_dim = 512  # 32 tokens × 16 features
    ref_kl_weight = 1e-6
    latent_dim = tokens * features
    return ref_kl_weight * (ref_dim / latent_dim)


class ExperimentSpec(NamedTuple):
    """Parsed experiment specification."""

    variant: str = "med"
    toks: int = 32
    enc: str = "s"
    dec: str = "s"
    feat: int = 384
    half_layers: bool = False
    kl_weight: float | None = None  # None means use default (computed from latent dim)
    anneal_steps: int | None = None
    free_bits: float | None = None
    kl_var_weight: float | None = None  # None disables KL variance loss
    encoder_disposable_registers: int = 0
    decoder_disposable_registers: int = 0
    tanh: bool = False
    nokl: bool = False  # If True, disable KL loss entirely (kl_weight=0)
    intermediate_layer: int | None = None  # Intermediate DINO layer to reconstruct
    intermediate_weight: float = 1.0  # Weight for intermediate reconstruction loss


def parse_experiment(name: str) -> ExperimentSpec | None:
    """Parse experiment name like 'fast-32-sb-128-kl3-an5k-fb0.1' into spec.

    Format: variant-toks-enc_dec-feat[-hl][-nokl][-[M]klN][-anNk][-fbN][-klvarN][-erN][-drN][-tanh][-ilN][-iwX]

    Examples:
        fast-32-sb-128              # basic config
        fast-32-sb-128-nokl         # disable KL loss (deterministic autoencoder)
        fast-32-sb-128-kl3          # kl_weight=1e-3
        fast-32-sb-128-25kl7        # kl_weight=2.5e-7 (mantissa=25 → 2.5)
        fast-32-sb-128-375kl6       # kl_weight=3.75e-6 (mantissa=375 → 3.75)
        fast-32-sb-128-kl3-an5k     # kl_weight=1e-3, anneal 5k steps
        fast-32-sb-128-fb0.1        # free_bits=0.1
        fast-32-sb-128-klvar3       # kl_var_weight=1e-3 (KL variance loss)
        fast-32-sb-128-er4          # 4 encoder disposable registers
        fast-32-sb-128-dr4          # 4 decoder disposable registers
        fast-32-sb-128-er4-dr4      # 4 disposable registers for both
        fast-32-sb-128-tanh         # apply tanh to latents
        fast-32-sb-128-il4          # reconstruct intermediate layer 4
        fast-32-sb-128-il4-iw05     # intermediate layer 4 with weight 0.5

    Returns None if name doesn't match the pattern (e.g., 'baseline').
    """
    pattern = r"""
        ^(?P<variant>fast|med|long|ext)
        -(?P<toks>\d+)
        -(?P<enc>[tsbl])(?P<dec>[tsbl])
        -(?P<feat>\d+)
        (?P<hl>-hl)?
        (?P<nokl>-nokl)?
        (?:-(?P<kl_mantissa>\d+)?kl(?P<kl>\d+))?
        (?:-an(?P<an>\d+)k)?
        (?:-fb(?P<fb>[\d.]+))?
        (?:-klvar(?P<klvar>\d+))?
        (?:-er(?P<er>\d+))?
        (?:-dr(?P<dr>\d+))?
        (?P<tanh>-tanh)?
        (?:-il(?P<il>\d+))?
        (?:-iw(?P<iw>[\d.]+))?
        $
    """
    match = re.match(pattern, name, re.VERBOSE)
    if not match:
        return None

    anneal_steps = int(match.group("an")) * 1000 if match.group("an") else None

    # Parse KL weight: -klN means 1e-N, -MklN means M*10^(-len(M)+1) * 1e-N
    # e.g., -kl3 → 1e-3, -25kl7 → 2.5e-7, -375kl6 → 3.75e-6
    kl_weight = None
    if match.group("kl"):
        exp = int(match.group("kl"))
        if match.group("kl_mantissa"):
            mantissa_str = match.group("kl_mantissa")
            mantissa = int(mantissa_str) / (10 ** (len(mantissa_str) - 1))
        else:
            mantissa = 1.0
        kl_weight = mantissa * (10**-exp)

    # Parse KL variance weight: -klvarN means 1e-N
    kl_var_weight = None
    if match.group("klvar"):
        kl_var_weight = 10 ** -int(match.group("klvar"))

    # Parse intermediate weight: -iwX means weight X (e.g., -iw05 → 0.5, -iw1 → 1.0)
    intermediate_weight = 1.0
    if match.group("iw"):
        iw_str = match.group("iw")
        # If it looks like "05", interpret as 0.5; if "1" interpret as 1.0
        if "." in iw_str:
            intermediate_weight = float(iw_str)
        elif len(iw_str) >= 2 and iw_str.startswith("0"):
            intermediate_weight = float("0." + iw_str[1:])
        else:
            intermediate_weight = float(iw_str)

    return ExperimentSpec(
        variant=match.group("variant"),
        toks=int(match.group("toks")),
        enc=match.group("enc"),
        dec=match.group("dec"),
        feat=int(match.group("feat")),
        half_layers=match.group("hl") is not None,
        kl_weight=kl_weight,
        anneal_steps=anneal_steps,
        free_bits=float(match.group("fb")) if match.group("fb") else None,
        kl_var_weight=kl_var_weight,
        encoder_disposable_registers=int(match.group("er")) if match.group("er") else 0,
        decoder_disposable_registers=int(match.group("dr")) if match.group("dr") else 0,
        tanh=match.group("tanh") is not None,
        nokl=match.group("nokl") is not None,
        intermediate_layer=int(match.group("il")) if match.group("il") else None,
        intermediate_weight=intermediate_weight,
    )


def build_config(spec: ExperimentSpec) -> FlatDinoConfig:
    """Build FlatDinoConfig from parsed experiment spec."""
    vit_map = {"t": "vit-t", "s": "vit-s", "b": "vit-b", "l": "vit-l"}
    epochs_map = {"fast": 50, "med": 100, "long": 150, "ext": 300}
    epochs = epochs_map[spec.variant]

    enc_cfg = VIT_CONFIGS[vit_map[spec.enc]].copy()
    dec_cfg = VIT_CONFIGS[vit_map[spec.dec]].copy()

    if spec.half_layers:
        enc_cfg["num_layers"] = max(1, enc_cfg["num_layers"] // 2)
        dec_cfg["num_layers"] = max(1, dec_cfg["num_layers"] // 2)

    # Compute kl_weight: use explicit value if provided, otherwise auto-compute from latent dim
    match (spec.nokl, spec.kl_weight):
        case (True, _):
            kl_weight = 0.0
        case (False, None):
            kl_weight = compute_default_kl_weight(spec.toks, spec.feat)
        case (False, weight):
            kl_weight = weight

    train_kwargs: dict = {
        "epochs": epochs,
        "warmup_epochs": 5,
        "lr_schedule": "wsd",
        "kl_weight": kl_weight,
    }
    match spec.variant:
        case "ext":
            train_kwargs["decay_epochs"] = int(epochs * 0.25)
        case "long":
            train_kwargs["decay_epochs"] = int(epochs * 0.15)
        case "med":
            train_kwargs["decay_epochs"] = int(epochs * 0.15)
        case "fast":
            train_kwargs["decay_epochs"] = int(epochs * 0.1)

    if spec.anneal_steps is not None:
        train_kwargs["kl_anneal_steps"] = spec.anneal_steps
    if spec.free_bits is not None:
        train_kwargs["free_bits"] = spec.free_bits
    if spec.kl_var_weight is not None:
        train_kwargs["kl_var_weight"] = spec.kl_var_weight

    # Total registers = latent tokens + disposable registers
    enc_total_registers = spec.toks + spec.encoder_disposable_registers
    # Decoder num_registers from baseline (256 output patches) + disposable
    dec_total_registers = baseline.decoder.num_registers + spec.decoder_disposable_registers

    # Decoder output_dim: 768 for final only, 768*2 for final + intermediate
    decoder_output_dim = 768 * 2 if spec.intermediate_layer is not None else 768

    return replace(
        baseline,
        train=replace(baseline.train, **train_kwargs),
        encoder=replace(
            baseline.encoder,
            num_registers=enc_total_registers,
            output_dim=2 * spec.feat,
            transformer=replace(baseline.encoder.transformer, mlp_type="swiglu", **enc_cfg),
        ),
        decoder=replace(
            baseline.decoder,
            input_dim=spec.feat,
            num_patches=spec.toks,
            num_registers=dec_total_registers,
            output_dim=decoder_output_dim,
            transformer=replace(baseline.decoder.transformer, mlp_type="swiglu", **dec_cfg),
        ),
        encoder_disposable_registers=spec.encoder_disposable_registers,
        decoder_disposable_registers=spec.decoder_disposable_registers,
        tanh=spec.tanh,
        intermediate_layer=spec.intermediate_layer,
        intermediate_weight=spec.intermediate_weight,
    )


# Static experiments that don't follow the standard naming pattern
_static_experiments = {
    "baseline": baseline,
    "fast-32": replace(
        baseline, train=replace(baseline.train, epochs=50, warmup_epochs=5, lr_schedule="wsd")
    ),
    "fast-32-64": replace(
        baseline,
        train=replace(baseline.train, epochs=50, warmup_epochs=5, lr_schedule="wsd"),
        encoder=replace(baseline.encoder, output_dim=128),  # mu and sigma
        decoder=replace(baseline.decoder, input_dim=64),
    ),
}


def get_experiment(name: str) -> FlatDinoConfig:
    """Get experiment config by name.

    First checks static experiments, then tries to parse the name dynamically.
    """
    if name in _static_experiments:
        return _static_experiments[name]

    spec = parse_experiment(name)
    if spec is None:
        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Expected format: variant-toks-enc_dec-feat[-hl][-nokl][-[M]klN][-anNk][-fbN][-klvarN][-erN][-drN][-tanh][-ilN][-iwX] "
            f"(e.g., 'fast-32-sb-128-kl3-an5k-fb0.1', 'fast-32-sb-128-il4', 'fast-32-sb-128-il4-iw05') "
            f"or one of: {list(_static_experiments.keys())}"
        )

    return build_config(spec)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    args: Args = tyro.cli(Args)

    config = get_experiment(args.experiment)
    main(args, config)
