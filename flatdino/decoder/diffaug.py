from dataclasses import dataclass

import chex
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jax.lax as lax


@dataclass
class DiffAugConfig:
    prob: float = 1.0
    cutout: float = 0.2


class DiffAug(nnx.Module):
    def __init__(self, cfg: DiffAugConfig):
        self.prob = abs(cfg.prob)

    def __call__(
        self,
        key: jax.Array,
        imgs: jax.Array,
    ) -> jax.Array:
        chex.assert_rank(imgs, 4)
        chex.assert_type(imgs, jnp.float32)
        b, h, w, _ = imgs.shape

        if self.prob < 1e-6:
            return imgs

        p_key, p_rand = jax.random.split(key)
        p_trans, p_color = jax.random.uniform(p_key, (2,))
        rand01 = jax.random.uniform(p_rand, (7, b, 1, 1))

        def apply_trans(imgs: jax.Array) -> jax.Array:
            ratio = 0.125
            delta_h = int(round(h * ratio))
            delta_w = int(round(w * ratio))

            translation_h = jnp.floor(rand01[0] * (2 * delta_h + 1)).astype(jnp.int32) - delta_h
            translation_w = jnp.floor(rand01[1] * (2 * delta_w + 1)).astype(jnp.int32) - delta_w

            grid_b, grid_h, grid_w = jnp.meshgrid(
                jnp.arange(b, dtype=jnp.int32),
                jnp.arange(h, dtype=jnp.int32),
                jnp.arange(w, dtype=jnp.int32),
                indexing="ij",
            )
            grid_h = lax.clamp(0, grid_h + translation_h + 1, h + 1)
            grid_w = lax.clamp(0, grid_w + translation_w + 1, w + 1)
            pad_width = ((0, 0), (1, 1), (1, 1), (0, 0))  # BHWC: pad H and W
            imgs_pad = jnp.pad(imgs, pad_width, mode="constant", constant_values=0)
            return imgs_pad[grid_b, grid_h, grid_w]

        def apply_color(imgs: jax.Array) -> jax.Array:
            imgs = imgs + (jnp.expand_dims(rand01[2], -1) - 0.5)
            imgs_mean = jnp.mean(imgs, axis=-1, keepdims=True)
            imgs = (imgs - imgs_mean) * jnp.expand_dims(2 * rand01[3], -1) + imgs_mean
            imgs_mean = jnp.mean(imgs, axis=(1, 2, 3), keepdims=True)
            imgs = (imgs - imgs_mean) * jnp.expand_dims(rand01[4] + 0.5, -1) + imgs_mean
            return imgs

        imgs = jax.lax.cond(p_trans <= self.prob, apply_trans, lambda x: x, imgs)
        imgs = jax.lax.cond(p_color <= self.prob, apply_color, lambda x: x, imgs)
        return imgs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import tensorflow as tf
    import tensorflow_datasets as tfds

    def load_imagenet_batch(batch_size: int = 16, image_size: int = 224, seed: int = 0) -> jax.Array:
        ds = tfds.load("imagenet2012", split="validation", shuffle_files=False, read_config=tfds.ReadConfig(shuffle_seed=seed))
        ds = ds.map(lambda x: {"image": tf.image.resize(x["image"], (image_size, image_size))}, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.take(batch_size)
        imgs = [ex["image"].numpy() for ex in tfds.as_numpy(ds)]
        imgs = np.stack(imgs, axis=0).astype(np.float32) / 255.0
        return jnp.asarray(imgs)

    key = jax.random.PRNGKey(0)
    imgs = load_imagenet_batch()

    aug = DiffAug(DiffAugConfig(prob=1.0))
    key, aug_key = jax.random.split(key)
    aug_imgs = aug(aug_key, imgs)

    imgs_np = np.clip(np.array(aug_imgs), 0.0, 1.0)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()
    for ax, img in zip(axes, imgs_np):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
