from typing import Callable, Iterable
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from tqdm import tqdm

from flatdino.data import DataLoaders
from flatdino.distributed import prefetch_to_mesh


def precompute_dataset_features(
    feat_fn: Callable,
    loader: Iterable,
    iters: int | None = None,
    *,
    mesh: jax.sharding.Mesh,
    gather: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Helper function to precompute the features of a pre-trained model.

    Args:
      model (nnx.Module): the pre-trained model.
      feat_fn (Callable): function that takes the model and the input and computes the feature vector.
      loader: (grain.Dataloader): the data loader. Assumes that the dataset provides 'image' and 'label' keys.
      iters (Optional[int]): the total number of samples in the dataset (for visualization purposes)
      mesh (jax.sharding.Mesh): the device mesh.
      gather (bool): if True, the dataset features will be replicated to all devices.

    Returns:
      features (jax.Array): an array containing all the features, replicated to each accelerator.
      labels (jax.Array): an array containing all the labels, replicated to each accelerator.
    """

    all_feature_list = []
    all_labels_list = []
    image_sharding = NamedSharding(mesh, P("data", None, None, None))
    label_sharding = NamedSharding(mesh, P("data"))

    for batch in tqdm(
        prefetch_to_mesh(loader, 1, mesh), desc="extracting", total=iters, leave=False
    ):
        imgs, lbls = batch["image"], batch["label"]

        # Pick samples so that they are divisible by the number of devices
        nb_samples = (imgs.shape[0] // jax.device_count()) * jax.device_count()
        imgs, lbls = imgs[:nb_samples], lbls[:nb_samples]

        imgs = jax.device_put(imgs, image_sharding)
        lbls = jax.device_put(lbls, label_sharding)

        feats = feat_fn(imgs)

        all_feature_list.append(feats)
        all_labels_list.append(lbls)

    all_feats = jnp.concatenate(all_feature_list, axis=0)
    all_labels = jnp.concatenate(all_labels_list, axis=0)

    norm = jnp.linalg.norm(all_feats, axis=-1, keepdims=True)
    all_feats = all_feats / jnp.maximum(norm, 1e-6)

    if gather:

        @partial(
            jax.shard_map,
            mesh=mesh,
            in_specs=(P("data", None), P("data")),
            out_specs=(P(None, None), P(None)),
            check_vma=False,
        )
        def gather_dataset(feats: jax.Array, lbls: jax.Array) -> tuple[jax.Array, jax.Array]:
            feats_gathered = jax.lax.all_gather(feats, axis_name="data", axis=0, tiled=True)
            lbls_gathered = jax.lax.all_gather(lbls, axis_name="data", axis=0, tiled=True)
            return feats_gathered, lbls_gathered

        all_feats, all_labels = gather_dataset(all_feats, all_labels)

    return all_feats, all_labels


def precompute_features(
    collect_batch_size: int,
    feat_fn: Callable,
    data: DataLoaders,
    *,
    mesh: jax.sharding.Mesh,
    gather: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    train_loader, val_loader = data.train_loader, data.val_loader
    train_iters = data.train_ds_size // collect_batch_size
    val_iters = (data.val_ds_size + collect_batch_size - 1) // collect_batch_size

    train_feats, train_lbls = precompute_dataset_features(
        feat_fn, iter(train_loader), train_iters, mesh=mesh, gather=gather
    )
    val_feats, val_lbls = precompute_dataset_features(
        feat_fn, iter(val_loader), val_iters, mesh=mesh, gather=gather
    )

    return train_feats, train_lbls, val_feats, val_lbls
