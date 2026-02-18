"""Tests for Finite Scalar Quantization (FSQ).

Verifies the implementation matches the specification from:
    "Finite Scalar Quantization: VQ-VAE Made Simple"
    Mentzer et al., ICLR 2024.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from flatdino.models.fsq import FSQ, round_ste


# ---- round_ste ----------------------------------------------------------


class TestRoundSTE:
    def test_forward_rounds(self):
        z = jnp.array([0.3, 1.7, -0.5, 2.0])
        out = round_ste(z)
        np.testing.assert_array_equal(out, jnp.round(z))

    def test_gradient_is_identity(self):
        grad_fn = jax.grad(lambda z: round_ste(z).sum())
        grads = grad_fn(jnp.array([0.3, 1.7, -0.5]))
        np.testing.assert_allclose(grads, jnp.ones(3))


# ---- FSQ basics ---------------------------------------------------------


class TestFSQInit:
    def test_codebook_size(self):
        fsq = FSQ(levels=[8, 5, 5, 5])
        assert fsq.codebook_size == 8 * 5 * 5 * 5  # 1000

    def test_num_dimensions(self):
        fsq = FSQ(levels=[8, 6, 5])
        assert fsq.num_dimensions == 3

    def test_single_level(self):
        fsq = FSQ(levels=[7])
        assert fsq.codebook_size == 7
        assert fsq.num_dimensions == 1


# ---- Quantization -------------------------------------------------------


class TestQuantize:
    @pytest.fixture
    def fsq(self):
        return FSQ(levels=[5, 5, 5])

    def test_output_shape(self, fsq):
        z = jnp.ones((2, 16, 3))
        out = fsq(z)
        assert out.shape == z.shape

    def test_output_in_range(self, fsq):
        key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (100, 3)) * 10
        out = fsq(z)
        assert jnp.all(out >= -1.0)
        assert jnp.all(out <= 1.0)

    def test_output_discrete(self, fsq):
        """Outputs should be multiples of 1/half_width."""
        key = jax.random.PRNGKey(42)
        z = jax.random.normal(key, (100, 3))
        out = fsq(z)
        # For levels=[5,5,5], half_width=2, valid values: {-1, -0.5, 0, 0.5, 1}
        rescaled = out * 2  # should be integers
        np.testing.assert_allclose(rescaled, jnp.round(rescaled), atol=1e-6)

    def test_number_of_unique_values(self, fsq):
        """Each channel should produce at most L unique values."""
        key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (10000, 3)) * 10
        out = fsq(z)
        for i in range(3):
            unique = jnp.unique(out[:, i])
            assert len(unique) == 5

    def test_even_levels(self):
        """Even levels should still produce exactly L unique values."""
        fsq = FSQ(levels=[8, 6])
        key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (10000, 2)) * 10
        out = fsq(z)
        assert len(jnp.unique(out[:, 0])) == 8
        assert len(jnp.unique(out[:, 1])) == 6

    def test_zero_maps_near_zero(self, fsq):
        """Input of zeros should quantize near zero."""
        z = jnp.zeros((1, 3))
        out = fsq(z)
        np.testing.assert_allclose(out, 0.0, atol=0.51)

    def test_deterministic(self, fsq):
        z = jnp.array([[0.3, -1.2, 0.7]])
        out1 = fsq(z)
        out2 = fsq(z)
        np.testing.assert_array_equal(out1, out2)

    def test_batched(self, fsq):
        """Quantization should be independent per element."""
        z1 = jnp.array([[0.1, 0.2, 0.3]])
        z2 = jnp.array([[1.0, -1.0, 0.0]])
        batched = jnp.concatenate([z1, z2], axis=0)
        out = fsq(batched)
        np.testing.assert_array_equal(out[0], fsq(z1)[0])
        np.testing.assert_array_equal(out[1], fsq(z2)[0])


# ---- Gradient flow -------------------------------------------------------


class TestGradients:
    def test_gradients_flow(self):
        fsq = FSQ(levels=[5, 5, 5])

        def loss_fn(z):
            return jnp.sum(fsq(z) ** 2)

        z = jnp.array([[0.5, -0.3, 1.0]])
        grads = jax.grad(loss_fn)(z)
        # Gradients should be nonzero (straight-through)
        assert jnp.any(grads != 0)

    def test_gradient_shape(self):
        fsq = FSQ(levels=[8, 5, 5, 5])

        def loss_fn(z):
            return jnp.mean(fsq(z))

        z = jnp.ones((4, 16, 4))
        grads = jax.grad(loss_fn)(z)
        assert grads.shape == z.shape


# ---- Index conversions ---------------------------------------------------


class TestIndices:
    @pytest.fixture
    def fsq(self):
        return FSQ(levels=[8, 5, 5, 5])

    def test_round_trip(self, fsq):
        """codes -> indices -> codes should be identity."""
        key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (50, 4))
        codes = fsq(z)
        indices = fsq.codes_to_indices(codes)
        recovered = fsq.indices_to_codes(indices)
        np.testing.assert_allclose(recovered, codes, atol=1e-6)

    def test_indices_range(self, fsq):
        """All indices should be in [0, codebook_size)."""
        key = jax.random.PRNGKey(1)
        z = jax.random.normal(key, (1000, 4)) * 10
        codes = fsq(z)
        indices = fsq.codes_to_indices(codes)
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < fsq.codebook_size)

    def test_indices_dtype(self, fsq):
        z = jnp.zeros((1, 4))
        indices = fsq.codes_to_indices(fsq(z))
        assert indices.dtype == jnp.int32

    def test_unique_indices(self, fsq):
        """Different quantized inputs should map to different indices."""
        key = jax.random.PRNGKey(2)
        z = jax.random.normal(key, (10000, 4)) * 10
        codes = fsq(z)
        indices = fsq.codes_to_indices(codes)
        # Number of unique indices should be >1 (many distinct codes)
        assert len(jnp.unique(indices)) > 1

    def test_codebook_covers_all_indices(self, fsq):
        """Full codebook should have exactly codebook_size entries."""
        cb = fsq.codebook
        assert cb.shape == (fsq.codebook_size, fsq.num_dimensions)

    def test_codebook_round_trip(self, fsq):
        """Codebook -> indices -> codebook should be identity."""
        cb = fsq.codebook
        indices = fsq.codes_to_indices(cb)
        # Indices should be 0, 1, 2, ..., codebook_size-1
        np.testing.assert_array_equal(indices, jnp.arange(fsq.codebook_size))

    def test_different_levels_round_trip(self):
        """Round-trip works for various level configurations."""
        for levels in [[3, 3], [7], [8, 6, 5], [5, 5, 5, 5, 5]]:
            fsq = FSQ(levels=levels)
            key = jax.random.PRNGKey(0)
            z = jax.random.normal(key, (100, len(levels))) * 5
            codes = fsq(z)
            indices = fsq.codes_to_indices(codes)
            recovered = fsq.indices_to_codes(indices)
            np.testing.assert_allclose(
                recovered, codes, atol=1e-6,
                err_msg=f"Round-trip failed for levels={levels}",
            )


# ---- Codebook exhaustiveness --------------------------------------------


class TestCodebook:
    def test_small_codebook_exhaustive(self):
        """For a small codebook, quantization should hit every entry."""
        fsq = FSQ(levels=[3, 3])  # codebook_size=9
        # Sweep a fine grid to hit all 9 entries
        x = jnp.linspace(-5, 5, 50)
        grid = jnp.stack(jnp.meshgrid(x, x), axis=-1).reshape(-1, 2)
        codes = fsq(grid)
        indices = fsq.codes_to_indices(codes)
        assert len(jnp.unique(indices)) == 9

    def test_codebook_values_in_range(self):
        fsq = FSQ(levels=[8, 5, 5, 5])
        cb = fsq.codebook
        assert jnp.all(cb >= -1.0)
        assert jnp.all(cb <= 1.0)


# ---- JIT compatibility ---------------------------------------------------


class TestJIT:
    def test_jit_quantize(self):
        fsq = FSQ(levels=[5, 5, 5])
        z = jnp.ones((2, 3))
        out_eager = fsq(z)
        out_jit = jax.jit(fsq)(z)
        np.testing.assert_array_equal(out_eager, out_jit)

    def test_jit_codes_to_indices(self):
        fsq = FSQ(levels=[5, 5, 5])
        codes = fsq(jnp.ones((2, 3)))
        idx_eager = fsq.codes_to_indices(codes)
        idx_jit = jax.jit(fsq.codes_to_indices)(codes)
        np.testing.assert_array_equal(idx_eager, idx_jit)

    def test_jit_indices_to_codes(self):
        fsq = FSQ(levels=[5, 5, 5])
        indices = jnp.array([0, 10, 100])
        c_eager = fsq.indices_to_codes(indices)
        c_jit = jax.jit(fsq.indices_to_codes)(indices)
        np.testing.assert_array_equal(c_eager, c_jit)
