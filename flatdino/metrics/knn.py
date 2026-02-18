from typing import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import pandas as pd

from flatdino.data import DataLoaders
from flatdino.metrics.utils import precompute_features


@dataclass
class KNNConfig:
    nb_knn: list[int] = field(default_factory=lambda: [10, 20, 100])
    """Number of nearest neightbours"""
    temperature: float = 0.07
    """Temperature to use in the voting coefficient"""
    gpu_batch_size: int = 256
    """Batch size per gpu"""
    n_class_per_list: int = -1
    """Number to take per class"""
    n_tries: int = 1
    """Number of tries"""


def knn_classifier(
    train_features: jax.Array,
    train_labels: jax.Array,
    val_features: jax.Array,
    val_labels: jax.Array,
    k: int,
    temperature: float,
    num_classes: int = 1000,
    val_chunk_size: int = 256,
) -> tuple[float, float]:
    num_val_chunks = (val_labels.shape[0] + val_chunk_size - 1) // val_chunk_size

    def inner_loop_step(features: jax.Array, targets: jax.Array):
        similarity = jnp.matmul(features, train_features.T)
        dist, ids = jax.lax.top_k(similarity, k=k)
        dist = jnp.exp(dist / temperature)

        probs = jax.vmap(
            lambda labels, weights: jnp.bincount(labels, weights=weights, length=num_classes),
            in_axes=(0, 0),
        )(train_labels[ids], dist)
        preds = jnp.argsort(probs, axis=1, descending=True)
        correct = preds == targets[:, None]

        top_1 = correct[:, 0].sum()
        top_5 = correct[:, : min(5, k)].sum()
        return top_1, top_5

    top_1, top_5, total = 0.0, 0.0, 0
    for i in range(num_val_chunks):
        features = val_features[i * val_chunk_size : (i + 1) * val_chunk_size]
        targets = val_labels[i * val_chunk_size : (i + 1) * val_chunk_size]

        top_1_chunk, top_5_chunk = inner_loop_step(features, targets)

        top_1 += top_1_chunk.item()
        top_5 += top_5_chunk.item()
        total += targets.size

    top_1 = top_1 * 100.0 / total
    top_5 = top_5 * 100.0 / total

    return top_1, top_5


def knn(
    cfg: KNNConfig,
    collect_batch_size: int,
    feat_fn: Callable,
    data: DataLoaders,
    *,
    num_classes,
    mesh: jax.sharding.Mesh,
    to_delete_after_precompute: nnx.Module | list[nnx.Module] | None = None,
) -> pd.DataFrame:
    train_feats, train_lbls, val_feats, val_lbls = precompute_features(
        collect_batch_size, feat_fn, data, mesh=mesh
    )
    print("train features: ", train_feats.shape[0])
    print("val features: ", val_feats.shape[0])

    if to_delete_after_precompute is not None:
        if isinstance(to_delete_after_precompute, nnx.Module):
            del to_delete_after_precompute
        else:
            for m in to_delete_after_precompute:
                del m

    results = []
    for k in cfg.nb_knn:
        top_1, top_5 = knn_classifier(
            train_feats,
            train_lbls,
            val_feats,
            val_lbls,
            k=k,
            temperature=cfg.temperature,
            val_chunk_size=cfg.gpu_batch_size,
            num_classes=num_classes,
        )
        results.append({"k": k, "top_1": top_1, "top_5": top_1})
        print(f"K: {k}\ttop_1: {top_1}\ttop_5: {top_5}")

    return pd.DataFrame(results)
