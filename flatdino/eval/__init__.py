"""FlatDINO evaluation utilities.

This module provides utilities for evaluating FlatDINO models, including:
- Checkpoint restoration via restore_encoder() for backward compatibility
- Evaluation result saving/loading
- Re-exports FlatDinoAutoencoder for direct use in eval scripts

For new eval scripts, prefer using FlatDinoAutoencoder directly with the mngr argument
for checkpoint restoration, as this automatically handles tanh and other config options.
"""

import fcntl
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jmp
import orbax.checkpoint as ocp
from dacite import Config as DaciteConfig, from_dict

from flatdino.data import DataConfig
from flatdino.models.vit import ViTConfig, ViTEncoder
from flatdino.pretrained.dinov2 import DinoWithRegisters
from flatdino.augmentations import FlatDinoAugConfig
from flatdino.autoencoder import FlatDinoAutoencoder, FlatDinoConfig

__all__ = [
    "load_eval_results",
    "save_eval_results",
    "restore_encoder",
    "RestoredComponents",
    "FlatDinoAutoencoder",
    "FlatDinoConfig",
    "EVAL_RESULTS_FILENAME",
]

EVAL_RESULTS_FILENAME = "eval_results.json"


def load_eval_results(checkpoint_path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file in checkpoint folder."""
    results_path = checkpoint_path / EVAL_RESULTS_FILENAME
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def save_eval_results(checkpoint_path: Path, key: str, results: dict[str, Any]) -> None:
    """Save evaluation results to JSON file in checkpoint folder.

    Merges new results under the given key with existing results.
    Uses file locking to handle concurrent writes from multiple workers.
    """
    results_path = checkpoint_path / EVAL_RESULTS_FILENAME
    lock_path = checkpoint_path / f".{EVAL_RESULTS_FILENAME}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            existing = load_eval_results(checkpoint_path)
            existing[key] = results
            with open(results_path, "w") as f:
                json.dump(existing, f, indent=2)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


@dataclass(frozen=True)
class RestoredComponents:
    dino: DinoWithRegisters
    encoder: ViTEncoder | None = None
    decoder: ViTEncoder | None = None
    data_cfg: DataConfig | None = None
    aug_cfg: FlatDinoAugConfig | None = None
    num_flat_tokens: int = 0
    encoder_disposable_registers: int = 0
    decoder_disposable_registers: int = 0
    tanh: bool = False


def restore_encoder(
    path: Path | str,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    encoder: bool = True,
    decoder: bool = False,
) -> RestoredComponents:
    dacite_config = DaciteConfig(cast=[tuple])

    # Handle both local paths and GCS paths (gs://...)
    path_str = str(path)
    if path_str.startswith("gs://"):
        # GCS path - skip existence check, orbax handles it
        ckpt_path: Path | str = path_str
    else:
        local_path = Path(path_str)
        assert local_path.exists(), f"Checkpoint path does not exist: {local_path}"
        ckpt_path = local_path.absolute()

    mngr = ocp.CheckpointManager(
        ckpt_path,
        item_names=FlatDinoAutoencoder.get_item_names() + ["optim", "loader", "config"],
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    step = mngr.best_step()
    assert step is not None, "failed to load best step"
    print(f"Found checkpoint at step {step}. Restoring...")

    cfg_ckpt = mngr.restore(step, args=ocp.args.Composite(config=ocp.args.JsonRestore()))["config"]

    enc = dec = None
    enc_vit_cfg = from_dict(ViTConfig, cfg_ckpt["encoder"], config=dacite_config)
    encoder_disposable_registers = cfg_ckpt.get("encoder_disposable_registers", 0)
    decoder_disposable_registers = cfg_ckpt.get("decoder_disposable_registers", 0)
    tanh = cfg_ckpt.get("tanh", False)
    num_flat_tokens = enc_vit_cfg.num_registers - encoder_disposable_registers
    if encoder:
        enc = ViTEncoder.restore(mngr, step, "encoder", mesh, cfg=enc_vit_cfg, mp=mp)

    if decoder:
        dec_vit_cfg = from_dict(ViTConfig, cfg_ckpt["decoder"], config=dacite_config)
        dec = ViTEncoder.restore(mngr, step, "decoder", mesh, cfg=dec_vit_cfg, mp=mp)

    dino_name = cfg_ckpt["dino_name"]
    dino = DinoWithRegisters(dino_name, resolution=224, dtype=mp.param_dtype)

    data_cfg = from_dict(DataConfig, cfg_ckpt["data"], config=dacite_config)
    aug_cfg = from_dict(FlatDinoAugConfig, cfg_ckpt["aug"], config=dacite_config)

    return RestoredComponents(
        dino=dino,
        encoder=enc,
        decoder=dec,
        data_cfg=data_cfg,
        aug_cfg=aug_cfg,
        num_flat_tokens=num_flat_tokens,
        encoder_disposable_registers=encoder_disposable_registers,
        decoder_disposable_registers=decoder_disposable_registers,
        tanh=tanh,
    )
