from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Literal, NamedTuple
import re

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
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
from flatdino.vq_autoencoder import VQDinoAutoencoder, VQDinoConfig, VQOptimConfig
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

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


@dataclass
class Args:
    seed: int = 42
    restore: Path | Literal["default"] | None = None
    """Path to restore checkpoint from. Use 'default' to restore from the default
    checkpoint path (output/flatdino/vq/{experiment} or GCS equivalent if gcs_bucket is set)."""
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
    gs://{gcs_bucket}/output/... using orbax's native GCS support."""
    data_in_bucket: bool = False
    """When True and gcs_bucket is set, load datasets from gs://{gcs_bucket}/{dataset_name}."""
    num_data_workers: int | None = None
    """Override the number of data loading workers. If None, uses the value from DataConfig."""
    use_wandb: bool = False
    wandb_log_every: int = 20
    project_name: str = "vqdino"
    wandb_name: str | None = None
    gpu_batch_size: int = 128
    profile_mode: Literal["disabled", "always", "window"] = "disabled"
    profiler_port: int = 7777
    profiler_start_step: int = 10
    profiler_stop_step: int = 2000
    val_epochs_freq: int = 5
    experiment: str = "baseline"
    implementation: Literal["cudnn", "xla"] = "xla"
    """Attention implementation: 'cudnn' for Flash Attention, 'xla' for default."""
    fsdp: int = 1
    """FSDP axis size. When fsdp=1, only data parallelism is used."""
    distributed: bool = False
    """Enable distributed training for multi-host setups (e.g., TPU pods)."""


def loss_fn(
    vqdino: VQDinoAutoencoder,
    x: jax.Array,
    recon_weight: float,
    key: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    z, aux = vqdino.encode(x, key=key)
    x_hat = vqdino.decode(z, keep_mask=aux.get("keep_mask"))

    recon_loss = jnp.mean(optax.huber_loss(x, x_hat))
    loss = recon_weight * recon_loss

    # Codebook utilization: fraction of unique indices used in this batch
    indices = aux["indices"]
    used = jnp.zeros(vqdino.fsq.codebook_size, dtype=jnp.bool_)
    used = used.at[indices.reshape(-1)].set(True)
    codebook_usage = jnp.sum(used) / vqdino.fsq.codebook_size

    metrics = {
        "recon_loss": recon_loss,
        "recon_mse": jnp.mean((x - x_hat) ** 2),
        "codebook_usage": codebook_usage,
        "z_pre_var": jnp.var(aux["z_pre"]),
    }
    if aux.get("k_keep") is not None:
        metrics["k_keep"] = aux["k_keep"].astype(jnp.float32)
    return loss, metrics


@nnx.jit(donate_argnames=("vqdino", "optim"))
def train_step(
    vqdino: VQDinoAutoencoder,
    dino: DinoWithRegisters,
    optim: nnx.Optimizer,
    key: jax.Array,
    x: jax.Array,
    recon_weight: float,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    chex.assert_rank(x, 4)
    b, h, w, d = x.shape

    x = jax.image.resize(x, (b, 224, 224, d), method="bilinear")
    dino_patches = dino(x)[:, 5:]

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, argnums=0, has_aux=True)(
        vqdino,
        dino_patches,
        recon_weight,
        key,
    )
    optim.update(vqdino.trainable_pytree, grads)
    return loss, metrics


@nnx.jit
def val_step(
    vqdino: VQDinoAutoencoder,
    dino: DinoWithRegisters,
    x: jax.Array,
) -> jax.Array:
    chex.assert_rank(x, 4)
    b, _, _, d = x.shape
    x = jax.image.resize(x, (b, 224, 224, d), method="bilinear")
    dino_patches = dino(x)[:, 5:]
    z, _ = vqdino.encode(dino_patches)
    x_hat = vqdino.decode(z)

    return jnp.sum(jnp.mean(optax.huber_loss(dino_patches, x_hat), axis=(1, 2)))


@nnx.jit
def val_step_per_k(
    vqdino: VQDinoAutoencoder,
    dino: DinoWithRegisters,
    x: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Validation step returning full loss and per-k losses for nested dropout."""
    chex.assert_rank(x, 4)
    b, _, _, d = x.shape
    x = jax.image.resize(x, (b, 224, 224, d), method="bilinear")
    dino_patches = dino(x)[:, 5:]
    z, _ = vqdino.encode(dino_patches)

    x_hat = vqdino.decode(z)
    total_loss = jnp.sum(jnp.mean(optax.huber_loss(dino_patches, x_hat), axis=(1, 2)))

    num_latents = vqdino.cfg.num_latents
    per_k = []
    k = 1
    while k <= num_latents:
        keep_mask = jnp.arange(num_latents)[None, :, None] < k
        x_hat_k = vqdino.decode(z, keep_mask=keep_mask)
        per_k.append(jnp.sum(jnp.mean(optax.huber_loss(dino_patches, x_hat_k), axis=(1, 2))))
        k *= 2

    return total_loss, jnp.stack(per_k)


def run_validation(
    vqdino: VQDinoAutoencoder,
    dino: DinoWithRegisters,
    val_loader,
    val_iters: int,
    mesh,
) -> dict[str, float]:
    val_iter = iter(val_loader)
    total_samples = 0
    loss = 0.0
    nested_dropout = vqdino.cfg.nested_dropout

    if nested_dropout:
        num_latents = vqdino.cfg.num_latents
        k_values = []
        k = 1
        while k <= num_latents:
            k_values.append(k)
            k *= 2
        per_k_totals = [0.0] * len(k_values)

    for val_batch in tqdm(
        prefetch_to_mesh(val_iter, 1, mesh),
        desc="eval",
        total=val_iters,
        leave=False,
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        disable=not is_primary_host(),
    ):
        imgs = val_batch["image"]
        if nested_dropout:
            batch_loss, batch_per_k = val_step_per_k(vqdino, dino, imgs)
            loss += batch_loss.item()
            for i, v in enumerate(batch_per_k.tolist()):
                per_k_totals[i] += v
        else:
            loss += val_step(vqdino, dino, imgs).item()
        total_samples += imgs.shape[0]

    metrics = {"val/loss": float(loss / total_samples)}
    if nested_dropout:
        for i, k in enumerate(k_values):
            metrics[f"val/loss_k{k}"] = float(per_k_totals[i] / total_samples)
    return metrics


def save_checkpoint(
    manager: ocp.CheckpointManager | None,
    step: int,
    vqdino: VQDinoAutoencoder,
    optim: nnx.Optimizer,
    data_iter,
    cfg: VQDinoConfig,
    metrics: dict | None = None,
) -> bool:
    if manager is None:
        return False
    saved = manager.save(
        step,
        metrics=metrics,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(vqdino))),
            optim=ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim))),
            loader=grain.PyGrainCheckpointSave(data_iter),
            config=ocp.args.JsonSave(asdict(cfg)),
        ),
    )
    return saved


def restore_model_state(
    restore_mngr: ocp.CheckpointManager,
    step: int,
    model: nnx.Module,
    mesh: Mesh,
) -> None:
    """Restore VQDinoAutoencoder state from checkpoint."""
    model_state = nnx.state(model)
    pure_state = nnx.to_pure_dict(model_state)
    restore_args = jax.tree.map(
        lambda x: ocp.ArrayRestoreArgs(sharding=NamedSharding(mesh, P())),
        pure_state,
    )
    pure_state = restore_mngr.restore(
        step,
        args=ocp.args.Composite(
            model=ocp.args.PyTreeRestore(pure_state, restore_args=restore_args)
        ),
    )["model"]
    nnx.replace_by_pure_dict(model_state, pure_state)
    nnx.update(model, model_state)


def create_checkpoint_managers(
    save_path: Path | str | None,
    item_names: list[str],
    wsd_decay_start: int | None,
    save_interval_steps: int,
    total_steps: int,
    keep_checkpoints_without_metrics: bool = True,
) -> tuple[ocp.CheckpointManager | None, ocp.CheckpointManager | None]:
    if save_path is None:
        return None, None

    if isinstance(save_path, Path):
        path_str = str(save_path.absolute())
    else:
        path_str = save_path

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


def main(args: Args, cfg: VQDinoConfig):
    if args.distributed:
        init_distributed()

    if not is_primary_host():
        logging.set_verbosity(logging.WARNING)

    mp_policy = jmp.Policy(
        param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32
    )

    target_cfg = cfg
    rngs = nnx.Rngs(args.seed + jax.process_index())

    num_devices = jax.device_count()
    if num_devices % args.fsdp != 0:
        raise ValueError(
            f"Number of devices ({num_devices}) must be divisible by fsdp ({args.fsdp})"
        )
    data_parallel_size = num_devices // args.fsdp
    devices = mesh_utils.create_device_mesh((data_parallel_size, args.fsdp))
    mesh = Mesh(devices, ("data", "model"))
    jax.set_mesh(mesh)
    logging.info(f"Process count: {jax.process_count()}")
    logging.info(f"Mesh shape: data={data_parallel_size}, model={args.fsdp}")
    logging.info(f"Mesh devices: {mesh.devices}")

    item_names = ["model", "optim", "loader", "config"]

    restore_mngr, ckpt_step = open_restore_manager(
        args.restore,
        args.maybe_restore,
        default_path=f"output/flatdino/vq/{args.experiment}",
        gcs_bucket=args.gcs_bucket,
        item_names=item_names,
    )

    if restore_mngr is not None:
        cfg_d = restore_mngr.restore(
            ckpt_step, args=ocp.args.Composite(config=ocp.args.JsonRestore())
        )["config"]
        ckpt_cfg = from_dict(VQDinoConfig, cfg_d, DaciteConfig(cast=[tuple], strict=False))
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

    use_wandb = args.use_wandb and is_primary_host()
    if use_wandb:
        name = args.wandb_name or args.experiment
        wandb.init(project=args.project_name, name=name, config=asdict(cfg))

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
        default_path=f"output/flatdino/vq/{args.experiment}",
        gcs_bucket=args.gcs_bucket,
    )

    steps_per_epoch = total_updates // cfg.train.epochs if cfg.train.epochs > 0 else total_updates
    if args.val_epochs_freq > 0:
        save_interval_steps = steps_per_epoch * args.val_epochs_freq
    else:
        save_interval_steps = steps_per_epoch

    assert cfg.aug.image_size[0] == cfg.aug.image_size[1]
    dino = DinoWithRegisters(cfg.dino_name, resolution=cfg.aug.image_size[0])
    vqdino = VQDinoAutoencoder(cfg, mp=mp_policy, rngs=rngs)

    if restore_mngr is not None:
        restore_model_state(restore_mngr, ckpt_step, vqdino, mesh)

    set_attn_implementation(vqdino, args.implementation)
    logging.info(f"Attention implementation set to: {args.implementation}")

    def wd_mask_fn(path: str, param: nnx.Variable) -> bool:
        if path in ["pos_embed", "cls_token", "reg_tokens"]:
            return False
        if param[...].ndim != 2:
            return False
        return True

    wd_mask = nnx.map_state(wd_mask_fn, vqdino.trainable_state)
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
    optim = nnx.Optimizer(vqdino.trainable_pytree, chain, wrt=nnx.Param)

    if restore_mngr is not None:
        restore_optimizer_state(restore_mngr, ckpt_step, optim, mesh)
        data_iter = restore_data_loader(restore_mngr, ckpt_step, data_iter)

    encoder_params = jax.tree.leaves(nnx.state(vqdino.encoder, nnx.Param))
    encoder_param_count = sum(jax.tree.map(lambda x: jnp.size(x), encoder_params))
    decoder_params = jax.tree.leaves(nnx.state(vqdino.decoder, nnx.Param))
    decoder_param_count = sum(jax.tree.map(lambda x: jnp.size(x), decoder_params))

    logging.info(f"num tokens: {vqdino.encoder.num_reg}")
    logging.info(f"levels: {cfg.levels}, codebook_size: {cfg.codebook_size}")
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
    final_val_metrics: dict[str, float] = {}

    train_step_cached = nnx.cached_partial(train_step, vqdino, dino, optim)

    for samples in prefetch_to_mesh(data_iter, 1, mesh):
        imgs = samples["image"]

        prev_updates = updates_completed
        train_loss, train_metrics = train_step_cached(
            rngs(),
            imgs,
            cfg.train.recon_weight,
        )
        metrics = {}

        updates_completed = int(optax.tree_utils.tree_get(optim.opt_state, "gradient_step"))
        ran_update = updates_completed > prev_updates

        if ran_update:
            pbar.update(updates_completed - prev_updates)
            profiler.step(updates_completed)

            is_preemption = mngr.reached_preemption(updates_completed) if mngr else False

            if is_preemption:
                logging.info(
                    "Preemption detected at step %d. Saving checkpoint immediately...",
                    updates_completed,
                )
                jax.block_until_ready(train_loss)
                save_checkpoint(
                    mngr, updates_completed, vqdino, optim, data_iter, cfg, metrics=None
                )
                mngr.wait_until_finished()
                logging.info("Checkpoint saved. Exiting due to preemption.")
                break

            should_save_now = mngr.should_save(updates_completed) if mngr else False

            if should_save_now:
                jax.block_until_ready(train_loss)
                val_metrics = run_validation(
                    vqdino,
                    dino,
                    val_loader,
                    val_iters,
                    mesh,
                )

                final_val_metrics = val_metrics.copy()
                metrics.update(val_metrics)

                save_checkpoint(
                    mngr, updates_completed, vqdino, optim, data_iter, cfg, val_metrics
                )

                if pre_decay_mngr is not None:
                    save_checkpoint(
                        pre_decay_mngr, updates_completed, vqdino, optim, data_iter, cfg
                    )

        if use_wandb:
            if ran_update and updates_completed % args.wandb_log_every == 0:
                metrics["train/lr"] = lr_sched(updates_completed)
                metrics["train/loss"] = float(train_loss)
                for name, value in train_metrics.items():
                    metrics[f"train/{name}"] = float(value)

            if metrics:
                metrics["step"] = updates_completed
                wandb.log(metrics)

        if updates_completed >= total_updates:
            break

    pbar.close()

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


# ---------------------------------------------------------------------------
# Baseline config
# ---------------------------------------------------------------------------

baseline = VQDinoConfig(
    dino_name="facebook/dinov2-with-registers-base",
    train=VQOptimConfig(batch_size=512, epochs=100),
    levels=[8, 5, 5, 5],
    encoder=ViTConfig(
        patch=None,
        use_pos_embeds=False,
        num_patches=256,
        input_dim=768,
        num_registers=32,
        transformer=TransformerConfig(
            embed_dim=768, num_layers=6, mlp_hidden_dim=3072, num_heads=12, selective=True
        ),
        output_dim=4,  # len(levels) = 4
    ),
    decoder=ViTConfig(
        patch=None,
        use_pos_embeds=True,
        num_patches=32,
        output_dim=768,
        num_registers=256,
        transformer=TransformerConfig(
            embed_dim=768, num_layers=6, mlp_hidden_dim=3072, num_heads=12, selective=True
        ),
        input_dim=4,  # len(levels) = 4
    ),
)


# ---------------------------------------------------------------------------
# Experiment parsing
# ---------------------------------------------------------------------------


class VQExperimentSpec(NamedTuple):
    """Parsed VQ experiment specification."""

    variant: str = "med"
    toks: int = 32
    enc: str = "s"
    dec: str = "s"
    levels: list[int] = [8, 5, 5, 5]
    half_layers: bool = False
    encoder_disposable_registers: int = 0
    decoder_disposable_registers: int = 0
    nested_dropout: bool = False


def parse_vq_experiment(name: str) -> VQExperimentSpec | None:
    """Parse VQ experiment name like 'fast-32-sb-L8555' into spec.

    Format: variant-toks-enc_dec-L<levels>[-hl][-nd][-erN][-drN]

    Each digit after L is one FSQ level (single-digit 3-9 per FSQ paper).

    Examples:
        fast-32-sb-L8555              # levels=[8,5,5,5], codebook=1000
        med-32-tb-L75555              # levels=[7,5,5,5,5], codebook=4375
        fast-64-sb-L88865             # levels=[8,8,8,6,5], codebook=15360
        fast-32-sb-L8555-hl           # half layers
        fast-32-sb-L8555-nd           # nested dropout
        fast-32-sb-L8555-er4          # 4 encoder disposable registers

    Returns None if name doesn't match the pattern.
    """
    pattern = r"""
        ^(?P<variant>fast|med|long|ext)
        -(?P<toks>\d+)
        -(?P<enc>[tsbl])(?P<dec>[tsbl])
        -L(?P<levels>\d+)
        (?P<hl>-hl)?
        (?P<nd>-nd)?
        (?:-er(?P<er>\d+))?
        (?:-dr(?P<dr>\d+))?
        $
    """
    match = re.match(pattern, name, re.VERBOSE)
    if not match:
        return None

    levels = [int(d) for d in match.group("levels")]

    return VQExperimentSpec(
        variant=match.group("variant"),
        toks=int(match.group("toks")),
        enc=match.group("enc"),
        dec=match.group("dec"),
        levels=levels,
        half_layers=match.group("hl") is not None,
        encoder_disposable_registers=int(match.group("er")) if match.group("er") else 0,
        decoder_disposable_registers=int(match.group("dr")) if match.group("dr") else 0,
        nested_dropout=match.group("nd") is not None,
    )


def build_vq_config(spec: VQExperimentSpec) -> VQDinoConfig:
    """Build VQDinoConfig from parsed experiment spec."""
    vit_map = {"t": "vit-t", "s": "vit-s", "b": "vit-b", "l": "vit-l"}
    epochs_map = {"fast": 50, "med": 100, "long": 150, "ext": 300}
    epochs = epochs_map[spec.variant]

    enc_cfg = VIT_CONFIGS[vit_map[spec.enc]].copy()
    dec_cfg = VIT_CONFIGS[vit_map[spec.dec]].copy()

    if spec.half_layers:
        enc_cfg["num_layers"] = max(1, enc_cfg["num_layers"] // 2)
        dec_cfg["num_layers"] = max(1, dec_cfg["num_layers"] // 2)

    num_levels = len(spec.levels)

    train_kwargs: dict = {
        "epochs": epochs,
        "warmup_epochs": 5,
        "lr_schedule": "wsd",
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

    enc_total_registers = spec.toks + spec.encoder_disposable_registers
    dec_total_registers = baseline.decoder.num_registers + spec.decoder_disposable_registers

    # When nested_dropout is enabled, latent_proj handles the projection from
    # len(levels) → embed_dim, so decoder's input_dim should equal embed_dim
    # to skip ViTEncoder's own input_proj.
    decoder_input_dim = dec_cfg["embed_dim"] if spec.nested_dropout else num_levels

    return replace(
        baseline,
        train=replace(baseline.train, **train_kwargs),
        levels=list(spec.levels),
        nested_dropout=spec.nested_dropout,
        encoder=replace(
            baseline.encoder,
            num_registers=enc_total_registers,
            output_dim=num_levels,
            transformer=replace(baseline.encoder.transformer, mlp_type="swiglu", **enc_cfg),
        ),
        decoder=replace(
            baseline.decoder,
            input_dim=decoder_input_dim,
            num_patches=spec.toks,
            num_registers=dec_total_registers,
            output_dim=768,
            transformer=replace(baseline.decoder.transformer, mlp_type="swiglu", **dec_cfg),
        ),
        encoder_disposable_registers=spec.encoder_disposable_registers,
        decoder_disposable_registers=spec.decoder_disposable_registers,
    )


_static_experiments = {
    "baseline": baseline,
}


def get_experiment(name: str) -> VQDinoConfig:
    """Get experiment config by name.

    First checks static experiments, then tries to parse the name dynamically.
    """
    if name in _static_experiments:
        return _static_experiments[name]

    spec = parse_vq_experiment(name)
    if spec is None:
        raise ValueError(
            f"Unknown experiment: {name}. "
            f"Expected format: variant-toks-enc_dec-L<levels>[-hl][-nd][-erN][-drN] "
            f"(e.g., 'fast-32-sb-L8555', 'med-64-tb-L75555-hl', 'fast-32-sb-L8555-nd') "
            f"or one of: {list(_static_experiments.keys())}"
        )

    return build_vq_config(spec)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    args: Args = tyro.cli(Args)

    config = get_experiment(args.experiment)
    main(args, config)
