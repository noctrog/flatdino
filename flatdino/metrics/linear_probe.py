from typing import Callable
from dataclasses import dataclass, field
import itertools

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import flax.nnx as nnx
import numpy as np
import optax
import pandas as pd

from tqdm import tqdm

from flatdino.data import DataLoaders
from flatdino.metrics.utils import precompute_features


@dataclass
class LinearProbeConfig:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 16384

    learning_rates: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.001])
    weight_decays: list[float] = field(default_factory=lambda: [0.0, 0.0005])


class LinearClassifier(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(input_dim, output_dim, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.linear(x)


def train_classifier(
    cfg: LinearProbeConfig,
    base_lr: float,
    weight_decay: float,
    mesh: jax.sharding.Mesh,
    train_feats: jax.Array,
    train_lbls: jax.Array,
):
    @nnx.jit
    def train_step(
        classifier: LinearClassifier, optim: nnx.Optimizer, feats: jax.Array, y: jax.Array
    ):
        def loss_fn(classifier: LinearClassifier, x: jax.Array, y: jax.Array):
            y_pred_logits = classifier(x)
            loss = optax.softmax_cross_entropy_with_integer_labels(y_pred_logits, y)
            return jnp.mean(loss)

        loss, grads = nnx.value_and_grad(loss_fn)(classifier, feats, y)
        optim.update(classifier, grads)
        return loss

    def make_lr_schedule(base_lr: float, divide_every_k_steps: int, factor: float = 10.0):
        def lr_schedule(step: int):
            steps = step // divide_every_k_steps
            return base_lr / (factor**steps)

        return lr_schedule

    samples_per_epoch = (train_feats.shape[0] // jax.device_count()) * jax.device_count()
    total_iterations = (samples_per_epoch * cfg.epochs) // cfg.batch_size
    iters_per_epoch = total_iterations // cfg.epochs

    classifier = LinearClassifier(train_feats.shape[-1], 1_000, rngs=nnx.Rngs(cfg.seed))

    lr_sched = make_lr_schedule(base_lr, iters_per_epoch * 15)
    chain = optax.chain(
        optax.clip_by_global_norm(10.0), optax.lars(lr_sched, weight_decay=weight_decay)
    )
    optim = nnx.Optimizer(classifier, tx=chain, wrt=nnx.Param)

    id_rng = np.random.default_rng(cfg.seed)
    pbar = tqdm(
        range(total_iterations),
        desc="train",
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
    )
    for _ in pbar:
        ids = id_rng.choice(samples_per_epoch, size=cfg.batch_size, replace=False)
        feats, y = train_feats[ids], train_lbls[ids]

        loss = train_step(classifier, optim, feats, y)
        pbar.set_postfix({"loss": loss.item()})

    return classifier


def eval_top1_classifier(
    cfg: LinearProbeConfig,
    classifier: LinearClassifier,
    val_feats: jax.Array,
    val_lbls: jax.Array,
    mesh: jax.sharding.Mesh,
):
    @nnx.jit
    def eval_step(classifier: LinearClassifier, feats: jax.Array, y: jax.Array):
        y_pred = classifier(feats)
        y_pred = jnp.argmax(y_pred, axis=1, keepdims=False)
        correct = jnp.sum(y_pred == y)
        return correct

    samples_per_epoch = (val_feats.shape[0] // jax.device_count()) * jax.device_count()
    total_iterations = (samples_per_epoch + cfg.batch_size - 1) // cfg.batch_size

    correct = 0
    total_samples = 0
    all_ids = np.arange(samples_per_epoch)
    for i in tqdm(
        range(total_iterations),
        desc="validating",
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
    ):
        start, end = i * cfg.batch_size, min((i + 1) * cfg.batch_size, samples_per_epoch)
        ids = all_ids[start:end]
        feats = jax.device_put(val_feats[ids], NamedSharding(mesh, P("data", None)))
        y = jax.device_put(val_lbls[ids], NamedSharding(mesh, P("data")))

        batch_correct = eval_step(classifier, feats, y)

        correct += batch_correct.item()
        total_samples += feats.shape[0]

    top_1 = correct / total_samples
    return top_1


def linear_probe(
    cfg: LinearProbeConfig,
    collect_batch_size: int,
    feat_fn: Callable,
    data: DataLoaders,
    *,
    mesh: jax.sharding.Mesh,
    cache_features: bool = False,
):
    if cache_features:
        train_feats, train_lbls, val_feats, val_lbls = precompute_features(
            collect_batch_size, feat_fn, data, mesh=mesh
        )

        results = []
        for base_lr, wd in itertools.product(cfg.learning_rates, cfg.weight_decays):
            classifier = train_classifier(cfg, base_lr, wd, mesh, train_feats, train_lbls)

            top_1 = eval_top1_classifier(cfg, classifier, val_feats, val_lbls, mesh)
            print(f"base_lr={base_lr}, wd={wd}, top_1={top_1}")
            results.append({"base_lr": base_lr, "wd": wd, "top_1": top_1})

        df = pd.DataFrame(results)
        df.to_csv("top1.csv", index=False)
        print("Grid search results saved to top1.csv")
        return df
    else:
        raise NotImplementedError()
