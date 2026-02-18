"""Estimate Total Correlation (TC) of FlatDINO latent tokens.

Total Correlation measures the redundancy in the latent representation:
    TC(z) = KL(q(z) || prod_i q(z_i))

Lower TC means more independent components = better compression efficiency.

Three granularity levels are supported:
- token: TC between tokens as multivariate units (default)
    TC(Z_1, ..., Z_T) where each Z_i is F-dimensional
    Answers: "How much do tokens share information?"

- within_token: TC within each token's features (averaged)
    (1/T) * sum_t TC(z_t1, ..., z_tF)
    Answers: "Are features within a token redundant?"

- scalar: TC across all scalar dimensions (original behavior)
    TC(z_1, ..., z_{T*F}) where each z_i is a scalar
    Answers: "Total redundancy across all dimensions"

Example usage:
    # Token-level TC (default, recommended)
    python -m flatdino.eval.total_correlation --checkpoint output/flatdino/vae/med-32-bl-64

    # Within-token TC
    python -m flatdino.eval.total_correlation --checkpoint ... --granularity within_token

    # Scalar-level TC (original behavior)
    python -m flatdino.eval.total_correlation --checkpoint ... --granularity scalar

    # With Gaussian approximation (faster, no training)
    python -m flatdino.eval.total_correlation --checkpoint ... --method gaussian
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import jmp
import flax.nnx as nnx
import numpy as np
import optax
import tyro
from absl import logging
from tqdm import tqdm

from flatdino.data import DataLoaders, create_dataloaders
from flatdino.augmentations import FlatDinoValAugmentations
from flatdino.eval import restore_encoder, save_eval_results
from flatdino.distributed import prefetch_to_mesh


@dataclass
class Args:
    checkpoint: Path
    """Path to FlatDINO checkpoint."""
    granularity: Literal["token", "within_token", "scalar"] = "token"
    """TC granularity: 'token' (between tokens), 'within_token' (within each token), 'scalar' (all dims)."""
    method: Literal["discriminator", "gaussian"] = "gaussian"
    """Estimation method: 'discriminator' (accurate) or 'gaussian' (fast approximation)."""
    num_images: int = 50000
    """Number of images to encode (max 50k for ImageNet val)."""
    discriminator_steps: int = 2000
    """Training steps for discriminator."""
    discriminator_hidden: int = 512
    """Hidden dimension of discriminator MLP."""
    batch_size: int = 256
    """Batch size for encoding and discriminator training."""
    lr: float = 1e-3
    """Learning rate for discriminator."""
    seed: int = 42


class TCDiscriminator(nnx.Module):
    """MLP discriminator for TC estimation.

    Classifies whether a sample comes from q(z) (real) or prod_i q(z_i) (permuted).
    """

    def __init__(self, input_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.layers = [
            nnx.Linear(input_dim, hidden_dim, rngs=rngs),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.Linear(hidden_dim, 1, rngs=rngs),
        ]
        self.norm1 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.layers[0](x)
        x = self.norm1(x)
        x = nnx.relu(x)
        x = self.layers[1](x)
        x = self.norm2(x)
        x = nnx.relu(x)
        x = self.layers[2](x)
        return x.squeeze(-1)


def permute_dimensions(key: jax.Array, x: jax.Array) -> jax.Array:
    """Permute each dimension independently to break dependencies.

    This transforms samples from q(z) to samples from prod_i q(z_i).
    """
    n, d = x.shape
    keys = jax.random.split(key, d)

    def permute_column(key, col):
        return jax.random.permutation(key, col)

    permuted = jax.vmap(permute_column, in_axes=(0, 1), out_axes=1)(keys, x)
    return permuted


def discriminator_loss(
    discriminator: TCDiscriminator,
    real: jax.Array,
    fake: jax.Array,
) -> tuple[jax.Array, dict]:
    """Binary cross-entropy loss for discriminator."""
    real_logits = discriminator(real)
    fake_logits = discriminator(fake)

    # Real samples should be classified as 1, fake as 0
    real_loss = jnp.mean(jax.nn.softplus(-real_logits))
    fake_loss = jnp.mean(jax.nn.softplus(fake_logits))
    loss = real_loss + fake_loss

    # Accuracy metrics
    real_acc = jnp.mean(real_logits > 0)
    fake_acc = jnp.mean(fake_logits < 0)

    return loss, {
        "loss": loss,
        "real_acc": real_acc,
        "fake_acc": fake_acc,
        "acc": (real_acc + fake_acc) / 2,
    }


@nnx.jit
def train_step(
    discriminator: TCDiscriminator,
    optim: nnx.Optimizer,
    real: jax.Array,
    fake: jax.Array,
) -> dict:
    """Single training step for discriminator."""
    (loss, metrics), grads = nnx.value_and_grad(discriminator_loss, has_aux=True)(
        discriminator, real, fake
    )
    optim.update(discriminator, grads)
    return metrics


def estimate_tc_discriminator(
    discriminator: TCDiscriminator,
    latents: jax.Array,
    batch_size: int = 1024,
) -> float:
    """Estimate TC using trained discriminator.

    TC ≈ E_{z~q(z)}[log D(z) / (1 - D(z))]
       = E_{z~q(z)}[logit(D(z))]

    where D(z) is the probability that z is real (from q(z)).
    """
    n = len(latents)
    total_logits = 0.0

    for i in range(0, n, batch_size):
        batch = latents[i:i + batch_size]
        logits = discriminator(batch)
        total_logits += float(logits.sum())

    tc = total_logits / n
    return tc


def estimate_tc_gaussian_scalar(latents: np.ndarray) -> float:
    """Estimate scalar-level TC using Gaussian approximation.

    Treats each scalar dimension as a separate variable.

    Under Gaussian assumption:
        TC = 0.5 * (log|Σ_diag| - log|Σ_full|)

    where Σ_diag is the diagonal of the covariance matrix.
    """
    n, d = latents.shape

    # Compute covariance matrix
    cov = np.cov(latents.T)

    # Add small regularization for numerical stability
    cov = cov + 1e-6 * np.eye(d)

    # Diagonal covariance (product of marginals)
    cov_diag = np.diag(np.diag(cov))

    # TC = 0.5 * (log|Σ_diag| - log|Σ_full|)
    sign_full, logdet_full = np.linalg.slogdet(cov)
    sign_diag, logdet_diag = np.linalg.slogdet(cov_diag)

    if sign_full <= 0 or sign_diag <= 0:
        logging.warning("Covariance matrix is not positive definite")
        return float("nan")

    tc = 0.5 * (logdet_diag - logdet_full)
    return float(tc)


def estimate_tc_gaussian_token(latents: np.ndarray, num_tokens: int) -> tuple[float, dict]:
    """Estimate token-level TC using Gaussian approximation.

    Treats each token as a multivariate random variable.

    TC(Z_1, ..., Z_T) = sum_i H(Z_i) - H(Z_1, ..., Z_T)

    Under Gaussian assumption:
        TC = 0.5 * (sum_i log|Σ_i| - log|Σ_full|)

    where Σ_i is the covariance of token i and Σ_full is the joint covariance.

    Returns:
        Tuple of (TC value, diagnostics dict)
    """
    n, d = latents.shape
    features_per_token = d // num_tokens

    # Reshape to (N, T, F)
    tokens = latents.reshape(n, num_tokens, features_per_token)

    # Diagnostics
    diagnostics = {
        "latent_mean": float(np.mean(latents)),
        "latent_std": float(np.std(latents)),
        "latent_min": float(np.min(latents)),
        "latent_max": float(np.max(latents)),
    }

    # Compute full covariance (over all dimensions)
    cov_full = np.cov(latents.T)

    # Check condition number before regularization
    eigvals_full = np.linalg.eigvalsh(cov_full)
    cond_number = eigvals_full.max() / max(eigvals_full.min(), 1e-10)
    diagnostics["cov_full_condition_number"] = float(cond_number)
    diagnostics["cov_full_min_eigval"] = float(eigvals_full.min())
    diagnostics["cov_full_max_eigval"] = float(eigvals_full.max())

    # Regularization scaled to eigenvalue magnitude
    reg = max(1e-6, eigvals_full.max() * 1e-6)
    cov_full = cov_full + reg * np.eye(d)
    diagnostics["regularization"] = float(reg)

    sign_full, logdet_full = np.linalg.slogdet(cov_full)
    diagnostics["logdet_full"] = float(logdet_full)

    if sign_full <= 0:
        logging.warning("Full covariance matrix is not positive definite")
        return float("nan"), diagnostics

    # Sum of log determinants of individual token covariances
    sum_logdet_tokens = 0.0
    token_logdets = []
    for t in range(num_tokens):
        token_t = tokens[:, t, :]  # (N, F)
        cov_t = np.cov(token_t.T)
        cov_t = cov_t + reg * np.eye(features_per_token)

        sign_t, logdet_t = np.linalg.slogdet(cov_t)
        if sign_t <= 0:
            logging.warning(f"Token {t} covariance matrix is not positive definite")
            return float("nan"), diagnostics

        token_logdets.append(float(logdet_t))
        sum_logdet_tokens += logdet_t

    diagnostics["sum_logdet_tokens"] = float(sum_logdet_tokens)
    diagnostics["mean_logdet_token"] = float(np.mean(token_logdets))

    # TC = 0.5 * (sum_i log|Σ_i| - log|Σ_full|)
    tc = 0.5 * (sum_logdet_tokens - logdet_full)

    # Also compute normalized TC (per token pair)
    num_pairs = num_tokens * (num_tokens - 1) / 2
    diagnostics["tc_per_token_pair"] = float(tc / num_pairs) if num_pairs > 0 else 0.0

    return float(tc), diagnostics


def estimate_tc_gaussian_within_token(latents: np.ndarray, num_tokens: int) -> tuple[float, list[float]]:
    """Estimate within-token TC using Gaussian approximation.

    Computes TC within each token's features, then averages.

    TC_within(Z_t) = 0.5 * (log|Σ_t_diag| - log|Σ_t|)

    Returns:
        Tuple of (mean TC across tokens, list of per-token TC values)
    """
    n, d = latents.shape
    features_per_token = d // num_tokens

    # Reshape to (N, T, F)
    tokens = latents.reshape(n, num_tokens, features_per_token)

    per_token_tc = []
    for t in range(num_tokens):
        token_t = tokens[:, t, :]  # (N, F)
        cov_t = np.cov(token_t.T)
        cov_t = cov_t + 1e-6 * np.eye(features_per_token)

        # Diagonal covariance
        cov_t_diag = np.diag(np.diag(cov_t))

        sign_full, logdet_full = np.linalg.slogdet(cov_t)
        sign_diag, logdet_diag = np.linalg.slogdet(cov_t_diag)

        if sign_full <= 0 or sign_diag <= 0:
            logging.warning(f"Token {t} covariance matrix is not positive definite")
            per_token_tc.append(float("nan"))
        else:
            tc_t = 0.5 * (logdet_diag - logdet_full)
            per_token_tc.append(float(tc_t))

    mean_tc = float(np.nanmean(per_token_tc))
    return mean_tc, per_token_tc


def compute_correlation_matrix(latents: np.ndarray, num_tokens: int) -> np.ndarray:
    """Compute correlation matrix between tokens (averaged over features)."""
    n, d = latents.shape
    features_per_token = d // num_tokens

    # Reshape to (N, T, F) and average over features
    reshaped = latents.reshape(n, num_tokens, features_per_token)
    token_means = reshaped.mean(axis=-1)  # (N, T)

    # Correlation matrix between tokens
    corr = np.corrcoef(token_means.T)
    return corr


def main(args: Args):
    logging.set_verbosity(logging.INFO)
    rngs = nnx.Rngs(args.seed)
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, output_dtype=jnp.float32)

    # Load encoder
    logging.info(f"Loading checkpoint from {args.checkpoint}")
    components = restore_encoder(args.checkpoint, mesh=mesh, mp=mp, encoder=True, decoder=False)
    encoder = components.encoder
    dino = components.dino
    num_tokens = components.num_flat_tokens

    # Create validation dataloader
    logging.info("Creating validation dataloader")
    data: DataLoaders = create_dataloaders(
        components.data_cfg,
        args.batch_size,
        val_aug=FlatDinoValAugmentations(components.aug_cfg, components.data_cfg),
        val_epochs=1,
        drop_remainder_val=False,
    )

    num_images = min(args.num_images, data.val_ds_size)
    logging.info(f"Encoding {num_images} images...")

    # Encode all images
    all_latents = []
    num_encoded = 0

    @nnx.jit
    def encode_batch(imgs: jax.Array) -> jax.Array:
        b, h, w, c = imgs.shape
        imgs = jax.image.resize(imgs, (b, 224, 224, c), method="bilinear")
        dino_patches = dino(imgs)[:, 5:]  # Remove CLS and registers
        encoded = encoder(dino_patches)
        # Extract mu (first half of output dim)
        mu = encoded[:, :, :encoded.shape[-1] // 2]
        return mu

    val_iter = iter(data.val_loader)
    for batch in tqdm(
        prefetch_to_mesh(val_iter, 1, mesh),
        total=(num_images + args.batch_size - 1) // args.batch_size,
        desc="Encoding",
    ):
        if num_encoded >= num_images:
            break

        imgs = batch["image"]
        mu = encode_batch(imgs)  # (B, T, F)

        # Flatten tokens: (B, T, F) -> (B, T*F)
        flat = mu.reshape(mu.shape[0], -1)
        all_latents.append(np.array(flat))
        num_encoded += len(imgs)

    latents = np.concatenate(all_latents, axis=0)[:num_images]
    logging.info(f"Encoded {len(latents)} images, latent shape: {latents.shape}")

    n, d = latents.shape
    features_per_token = d // num_tokens
    logging.info(f"Tokens: {num_tokens}, features per token: {features_per_token}")

    # Compute correlation matrix
    corr_matrix = compute_correlation_matrix(latents, num_tokens)
    mask = ~np.eye(num_tokens, dtype=bool)
    mean_off_diag_corr = np.abs(corr_matrix[mask]).mean()
    logging.info(f"Mean off-diagonal token correlation: {mean_off_diag_corr:.4f}")

    # Compute TC based on granularity and method
    per_token_tc = None  # Only populated for within_token granularity
    diagnostics = None  # Only populated for token granularity with gaussian method

    if args.method == "gaussian":
        logging.info(f"Estimating TC using Gaussian approximation (granularity: {args.granularity})...")

        if args.granularity == "token":
            tc, diagnostics = estimate_tc_gaussian_token(latents, num_tokens)
            logging.info(f"Token-level TC (Gaussian): {tc:.2f} nats")
        elif args.granularity == "within_token":
            tc, per_token_tc = estimate_tc_gaussian_within_token(latents, num_tokens)
            logging.info(f"Within-token TC (Gaussian): {tc:.2f} nats (mean across {num_tokens} tokens)")
        else:  # scalar
            tc = estimate_tc_gaussian_scalar(latents)
            logging.info(f"Scalar-level TC (Gaussian): {tc:.2f} nats")

    else:
        # Discriminator-based estimation (only supports scalar granularity)
        if args.granularity != "scalar":
            logging.warning(
                f"Discriminator method only supports scalar granularity. "
                f"Ignoring granularity={args.granularity} and using scalar."
            )

        logging.info("Training TC discriminator...")

        latents_jax = jnp.array(latents)
        discriminator = TCDiscriminator(d, args.discriminator_hidden, rngs=rngs)
        optim = nnx.Optimizer(discriminator, optax.adam(args.lr))

        pbar = tqdm(range(args.discriminator_steps), desc="Training discriminator")
        for step in pbar:
            # Sample batch
            key = rngs()
            idx_key, perm_key = jax.random.split(key)

            idx = jax.random.randint(idx_key, (args.batch_size,), 0, n)
            real = latents_jax[idx]
            fake = permute_dimensions(perm_key, real)

            metrics = train_step(discriminator, optim, real, fake)

            if step % 100 == 0:
                pbar.set_postfix({
                    "loss": f"{metrics['loss']:.3f}",
                    "acc": f"{metrics['acc']:.3f}",
                })

        # Estimate TC
        logging.info("Estimating TC...")
        tc = estimate_tc_discriminator(discriminator, latents_jax, batch_size=args.batch_size)
        logging.info(f"Total Correlation (discriminator): {tc:.2f} nats")

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Granularity: {args.granularity}")
    print(f"Latent shape: {num_tokens} tokens × {features_per_token} features = {d} dims")
    print(f"Mean off-diagonal correlation: {mean_off_diag_corr:.4f}")
    print(f"Total Correlation ({args.granularity}, {args.method}): {tc:.2f} nats")

    if per_token_tc is not None:
        print("\nPer-token TC (nats):")
        for t, tc_t in enumerate(per_token_tc):
            print(f"  Token {t}: {tc_t:.2f}")

    if diagnostics is not None:
        print("\n" + "-" * 50)
        print("DIAGNOSTICS (for debugging high TC values)")
        print("-" * 50)
        print("Latent statistics:")
        print(f"  Mean: {diagnostics['latent_mean']:.4f}")
        print(f"  Std:  {diagnostics['latent_std']:.4f}")
        print(f"  Range: [{diagnostics['latent_min']:.4f}, {diagnostics['latent_max']:.4f}]")
        print("Covariance matrix:")
        print(f"  Condition number: {diagnostics['cov_full_condition_number']:.2e}")
        print(f"  Eigenvalue range: [{diagnostics['cov_full_min_eigval']:.2e}, {diagnostics['cov_full_max_eigval']:.2e}]")
        print(f"  Regularization used: {diagnostics['regularization']:.2e}")
        print("Log determinants:")
        print(f"  log|Σ_full|: {diagnostics['logdet_full']:.2f}")
        print(f"  Σ log|Σ_i|: {diagnostics['sum_logdet_tokens']:.2f}")
        print(f"  Mean log|Σ_i|: {diagnostics['mean_logdet_token']:.2f}")
        print("Normalized TC:")
        print(f"  TC per token pair: {diagnostics['tc_per_token_pair']:.2f} nats")

    print("=" * 50)

    # Save results to JSON
    results = {
        "granularity": args.granularity,
        "method": args.method,
        "num_images": len(latents),
        "num_tokens": num_tokens,
        "features_per_token": features_per_token,
        "latent_dim": d,
        "mean_off_diag_corr": float(mean_off_diag_corr),
        "total_correlation": float(tc),
    }
    if per_token_tc is not None:
        results["per_token_tc"] = per_token_tc
    if diagnostics is not None:
        results["diagnostics"] = diagnostics

    save_eval_results(args.checkpoint, "total_correlation", results)
    print(f"Saved total correlation results to {args.checkpoint}/eval_results.json")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
