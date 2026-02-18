# FlatDINO

Source code for [**Laminating Representation Autoencoders for Efficient Diffusion**](https://arxiv.org/abs/2602.04873).

Compressing DINOv2 patch features into a smaller set of latent tokens using learned encoder-decoder architectures. Supports both VAE (continuous) and FSQ (discrete) bottlenecks. The paper explores the VAE setting; the FSQ variant is an additional experimental feature included in this codebase.

## Installation

```bash
uv sync
```

On Linux with CUDA, JAX will be installed with GPU support automatically. On macOS, JAX runs on CPU.

## Overview

FlatDINO takes the 256 patch tokens from a frozen DINOv2-B/14 encoder (each 768-dimensional) and compresses them into a much smaller latent representation (e.g., 32 tokens of 128 dimensions) using a ViT-based autoencoder. The decoder reconstructs the original DINO patches from these latents.

Two bottleneck types are supported:

- **VAE** (`train.py`): Continuous latents with KL regularization
- **VQ/FSQ** (`train_vq.py`): Discrete latents using Finite Scalar Quantization

On top of the encoder, two downstream training stages are available:

- **Decoder** (`train_decoder.py`): Adversarial image reconstruction from latent tokens (GAN + LPIPS + L1)
- **Generator** (`train_generator.py`): Flow-matching diffusion model for class-conditional generation from latents

## Training

### VAE encoder

The experiment name encodes all hyperparameters:

```
variant-tokens-enc_dec-features[-options]

  variant:  fast (50ep), med (100ep), long (150ep), ext (300ep)
  tokens:   number of latent tokens (e.g., 32)
  enc_dec:  encoder/decoder size (t=tiny, s=small, b=base, l=large)
  features: latent feature dimension per token
```

Examples:

```bash
# 32 latent tokens, small encoder / base decoder, 128-dim features, 50 epochs
uv run python -m flatdino.train --experiment fast-32-sb-128

# With KL weight 1e-3 and 5k step annealing
uv run python -m flatdino.train --experiment med-32-sb-256-kl3-an5k

# With free bits 0.1
uv run python -m flatdino.train --experiment long-32-sb-128-fb0.1

# Deterministic autoencoder (no KL)
uv run python -m flatdino.train --experiment fast-32-sb-128-nokl

# Resume from checkpoint
uv run python -m flatdino.train --experiment med-32-sb-256 --restore default

# With Weights & Biases logging
uv run python -m flatdino.train --experiment fast-32-sb-128 --use-wandb --wandb-name my_run
```

Optional experiment name suffixes:

| Suffix | Meaning | Example |
|--------|---------|---------|
| `-hl` | Half the number of transformer layers | `fast-32-sb-128-hl` |
| `-nokl` | Disable KL loss (deterministic AE) | `fast-32-sb-128-nokl` |
| `-klN` | KL weight = 1e-N | `fast-32-sb-128-kl3` |
| `-MklN` | KL weight = M * 1e-N (M is mantissa) | `fast-32-sb-128-25kl7` (2.5e-7) |
| `-anNk` | KL annealing over N*1000 steps | `fast-32-sb-128-kl3-an5k` |
| `-fbX` | Free bits threshold X | `fast-32-sb-128-fb0.1` |
| `-klvarN` | KL variance loss weight 1e-N | `fast-32-sb-128-klvar3` |
| `-erN` | N encoder disposable registers | `fast-32-sb-128-er4` |
| `-drN` | N decoder disposable registers | `fast-32-sb-128-dr4` |
| `-tanh` | Apply tanh to latents | `fast-32-sb-128-tanh` |
| `-ilN` | Reconstruct intermediate DINO layer N | `fast-32-sb-128-il4` |
| `-iwX` | Intermediate reconstruction weight X | `fast-32-sb-128-il4-iw05` |

### VQ/FSQ encoder

```bash
# 32 tokens, base encoder/decoder, FSQ levels [8,5,5,5]
uv run python -m flatdino.train_vq --experiment fast-32-bb-8555

# With nested dropout
uv run python -m flatdino.train_vq --experiment fast-32-bb-8555-nd
```

### Image decoder (adversarial)

Trains an image reconstruction decoder on top of a frozen FlatDINO encoder:

```bash
uv run python -m flatdino.train_decoder \
    --encoder-path output/flatdino/vae/fast-32-sb-128 \
    --experiment my_decoder
```

### Diffusion generator

Trains a flow-matching diffusion model to generate latent tokens:

```bash
uv run python -m flatdino.train_generator \
    --encoder-path output/flatdino/vae/fast-32-sb-128 \
    --experiment my_generator
```

### Common flags

| Flag | Description |
|------|-------------|
| `--restore PATH` | Restore from checkpoint |
| `--maybe-restore default` | Resume if checkpoint exists, otherwise train from scratch |
| `--use-wandb` | Enable Weights & Biases logging |
| `--gpu-batch-size N` | Per-device batch size (gradient accumulation handles the rest) |
| `--fsdp N` | FSDP sharding across N devices |
| `--implementation cudnn` | Use Flash Attention (cuDNN) |
| `--gcs-bucket NAME` | Save checkpoints to GCS |
| `--distributed` | Enable multi-host training (TPU pods) |

## Evaluation

All evaluation scripts are runnable as modules:

```bash
# k-NN classification on downstream datasets
uv run python -m flatdino.eval.knn --path output/flatdino/vae/fast-32-sb-128

# k-NN on CIFAR-10
uv run python -m flatdino.eval.knn --path output/flatdino/vae/fast-32-sb-128 --dataset cifar10

# Reconstruction metrics (FID, PSNR)
uv run python -m flatdino.eval.metrics --checkpoint output/flatdino/vae/fast-32-sb-128

# Reconstruction visualizations
uv run python -m flatdino.eval.reconstruction --checkpoint output/flatdino/vae/fast-32-sb-128

# Total correlation of latent tokens
uv run python -m flatdino.eval.total_correlation --checkpoint output/flatdino/vae/fast-32-sb-128

# Mutual information between tokens
uv run python -m flatdino.eval.mutual_information --checkpoint output/flatdino/vae/fast-32-sb-128

# Token ablation (importance analysis)
uv run python -m flatdino.eval.token_ablation --flatdino-path output/flatdino/vae/fast-32-sb-128

# Attention map visualization
uv run python -m flatdino.eval.attention_maps --checkpoint output/flatdino/vae/fast-32-sb-128

# Latent interpolation videos
uv run python -m flatdino.eval.interpolate --checkpoint output/flatdino/vae/fast-32-sb-128

# gFID (generative FID) for diffusion models
uv run python -m flatdino.eval.gfid --checkpoint output/flatdino/generator/my_generator

# rFID using ADM evaluator
uv run python -m flatdino.eval.rfid_adm --checkpoint output/flatdino/vae/fast-32-sb-128

# DINO feature spatial redundancy analysis
uv run python -m flatdino.eval.dino_spatial_redundancy

# DINO intrinsic dimensionality analysis
uv run python -m flatdino.eval.dino_intrinsic_dimensionality
```

Use `--help` on any script for full options.

## Latent statistics

Precompute mean/variance of latents (used by the diffusion generator):

```bash
uv run python -m flatdino.compute_stats --checkpoint output/flatdino/vae/fast-32-sb-128
```

## Project structure

```
flatdino/
  train.py              VAE training
  train_vq.py           FSQ training
  train_decoder.py      Adversarial image decoder training
  train_generator.py    Flow-matching diffusion training
  compute_stats.py      Latent statistics computation
  autoencoder.py        FlatDinoAutoencoder (VAE)
  vq_autoencoder.py     VQDinoAutoencoder (FSQ)
  augmentations.py      Data augmentations
  utils.py              LR schedules, checkpoint utilities
  distributed.py        JAX distributed training utilities

  models/
    vit.py              ViT encoder with register tokens
    transformer.py      Transformer blocks (attention, MLP, norms)
    dit.py              LightningDiT (diffusion transformer)
    fsq.py              Finite Scalar Quantization

  data/
    data.py             Dataset loading (grain + tfds)
    utils.py            Download and hashing utilities

  pretrained/
    dinov2.py           Frozen DINOv2-with-registers feature extractor
    dino.py             Frozen DINO ViT (for discriminator)
    rae_decoder.py      RAE ViT-MAE decoder (pixel reconstruction)
    inception.py        InceptionV3 (for FID)

  decoder/
    rae_decoder.py      RAE image decoder model
    discriminator.py    DINO-based discriminator (GAN training)
    dit_dh.py           DiT with dual heads
    diffaug.py          Differentiable augmentation
    sampling.py         Flow-matching sampling utilities
    augmentations.py    Decoder/ADM augmentations

  metrics/
    fid.py              FID computation
    gfid.py             Generative FID (class-conditional)
    knn.py              k-NN classifier
    linear_probe.py     Linear probe evaluation
    inversion.py        Token inversion (deep image prior)
    adm.py              OpenAI ADM evaluator (FID/sFID/IS/P/R)

  eval/                 25 evaluation scripts (see above)
```

## Tests

```bash
uv run pytest tests/ -v
```
