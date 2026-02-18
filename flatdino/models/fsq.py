"""Finite Scalar Quantization (FSQ).

Implements the quantization scheme from:
    "Finite Scalar Quantization: VQ-VAE Made Simple"
    Mentzer et al., ICLR 2024.
    https://arxiv.org/abs/2309.15505

FSQ replaces learned VQ codebooks with per-channel scalar quantization.
Each channel is bounded via tanh, rounded to the nearest integer (with
straight-through gradients), and normalized to [-1, 1]. The implicit
codebook is the Cartesian product of per-channel grids, so codebook_size
= prod(levels). This eliminates codebook collapse, commitment losses,
and EMA updates entirely.
"""

from functools import reduce
import operator

import jax
import jax.numpy as jnp
import numpy as np


def round_ste(z: jax.Array) -> jax.Array:
    """Round to nearest integer with straight-through gradient estimator."""
    return z + jax.lax.stop_gradient(jnp.round(z) - z)


class FSQ:
    """Finite Scalar Quantization.

    Quantizes each channel of the input independently to one of a fixed
    number of discrete levels. The implicit codebook is the Cartesian
    product of per-channel grids.

    Args:
        levels: Number of discrete levels per channel. The last dimension
            of the input must equal ``len(levels)``. For example,
            ``[8, 5, 5, 5]`` gives ``codebook_size = 1000``.
        eps: Small value to prevent tanh saturation at boundaries.

    Example::

        fsq = FSQ(levels=[8, 5, 5, 5])
        z = jnp.ones((2, 16, 4))  # (batch, seq, d)
        zhat = fsq(z)             # quantized, same shape
        indices = fsq.codes_to_indices(zhat)  # (2, 16)
        codes = fsq.indices_to_codes(indices) # round-trips back
    """

    def __init__(self, levels: list[int], *, eps: float = 1e-3):
        self.levels = levels
        self.eps = eps
        self.num_dimensions = len(levels)
        self.codebook_size = reduce(operator.mul, levels, 1)

        _levels = np.asarray(levels, dtype=np.float32)
        self._basis = np.concatenate(
            ([1], np.cumprod(levels[:-1]))
        ).astype(np.int32)
        self._half_width = (_levels // 2).astype(np.float32)

        # Precompute bound parameters.
        half_l = (_levels - 1) * (1 - eps) / 2
        offset = np.where(np.asarray(levels) % 2 == 1, 0.0, 0.5).astype(np.float32)
        shift = np.where(half_l > 0, np.arctan(offset / half_l), 0.0).astype(np.float32)
        self._half_l = half_l
        self._offset = offset
        self._shift = shift

    def bound(self, z: jax.Array) -> jax.Array:
        """Bound each channel to its discrete integer range via tanh."""
        return jnp.tanh(z + self._shift) * self._half_l - self._offset

    def quantize(self, z: jax.Array) -> jax.Array:
        """Quantize and normalize to approximately [-1, 1].

        Args:
            z: Input array with last dimension equal to ``num_dimensions``.

        Returns:
            Quantized codes with the same shape, values in ~[-1, 1].
        """
        quantized = round_ste(self.bound(z))
        return quantized / self._half_width

    def codes_to_indices(self, zhat: jax.Array) -> jax.Array:
        """Convert normalized codes to flat integer indices.

        Args:
            zhat: Quantized codes in ~[-1, 1], shape ``(..., d)``.

        Returns:
            Integer indices in ``[0, codebook_size)``, shape ``(...,)``.
        """
        zhat_int = jnp.round(zhat * self._half_width) + self._half_width
        return (zhat_int * self._basis).sum(axis=-1).astype(jnp.int32)

    def indices_to_codes(self, indices: jax.Array) -> jax.Array:
        """Convert flat integer indices back to normalized codes.

        Args:
            indices: Integer indices in ``[0, codebook_size)``, shape ``(...,)``.

        Returns:
            Normalized codes in ~[-1, 1], shape ``(..., d)``.
        """
        indices = indices[..., None]
        codes = (indices // self._basis) % jnp.asarray(self.levels, dtype=jnp.int32)
        return (codes - self._half_width) / self._half_width

    @property
    def codebook(self) -> jax.Array:
        """The full implicit codebook, shape ``(codebook_size, d)``."""
        return self.indices_to_codes(jnp.arange(self.codebook_size))

    def __call__(self, z: jax.Array) -> jax.Array:
        """Quantize the input (forward pass)."""
        return self.quantize(z)
