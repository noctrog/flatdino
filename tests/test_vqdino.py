"""Tests for VQDinoAutoencoder (FlatDINO with FSQ quantization)."""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import jmp
import flax.nnx as nnx
import numpy as np
import pytest

from flatdino.models.transformer import TransformerConfig
from flatdino.models.vit import ViTConfig
from flatdino.vq_autoencoder import VQDinoAutoencoder, VQDinoConfig


def make_config(
    levels: list[int] = [8, 5, 5, 5],
    num_latents: int = 4,
    num_output_patches: int = 8,
    embed_dim: int = 32,
    num_layers: int = 2,
    num_heads: int = 1,
    encoder_disposable_registers: int = 0,
    decoder_disposable_registers: int = 0,
) -> VQDinoConfig:
    """Create a small VQDinoConfig for testing."""
    num_levels = len(levels)
    return VQDinoConfig(
        levels=levels,
        encoder_disposable_registers=encoder_disposable_registers,
        decoder_disposable_registers=decoder_disposable_registers,
        encoder=ViTConfig(
            patch=None,
            num_patches=16,
            input_dim=64,
            num_registers=num_latents + encoder_disposable_registers,
            output_dim=num_levels,
            transformer=TransformerConfig(
                embed_dim=embed_dim,
                num_layers=num_layers,
                mlp_hidden_dim=embed_dim * 4,
                num_heads=num_heads,
            ),
        ),
        decoder=ViTConfig(
            patch=None,
            num_patches=num_latents,
            input_dim=num_levels,
            num_registers=num_output_patches + decoder_disposable_registers,
            output_dim=64,
            transformer=TransformerConfig(
                embed_dim=embed_dim,
                num_layers=num_layers,
                mlp_hidden_dim=embed_dim * 4,
                num_heads=num_heads,
            ),
        ),
    )


@pytest.fixture(autouse=True)
def mesh():
    """Set up a single-device mesh for FSDP sharding annotations."""
    devices = np.array(jax.devices()[:1]).reshape(1, 1)
    m = Mesh(devices, ("data", "model"))
    jax.set_mesh(m)
    yield m


@pytest.fixture
def mp():
    return jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )


@pytest.fixture
def cfg():
    return make_config()


@pytest.fixture
def model(cfg, mp):
    return VQDinoAutoencoder(cfg, mp=mp, rngs=nnx.Rngs(0))


# ---- Config properties ---------------------------------------------------


class TestVQDinoConfig:
    def test_codebook_size(self):
        cfg = make_config(levels=[8, 5, 5, 5])
        assert cfg.codebook_size == 8 * 5 * 5 * 5  # 1000

    def test_codebook_size_other_levels(self):
        cfg = make_config(levels=[7, 5, 5, 5, 5])
        assert cfg.codebook_size == 7 * 5 * 5 * 5 * 5  # 4375

    def test_num_latents(self):
        cfg = make_config(num_latents=4)
        assert cfg.num_latents == 4

    def test_num_latents_with_disposable(self):
        cfg = make_config(num_latents=4, encoder_disposable_registers=2)
        assert cfg.num_latents == 4

    def test_num_output_patches(self):
        cfg = make_config(num_output_patches=8)
        assert cfg.num_output_patches == 8

    def test_num_output_patches_with_disposable(self):
        cfg = make_config(num_output_patches=8, decoder_disposable_registers=2)
        assert cfg.num_output_patches == 8


# ---- Model initialization ------------------------------------------------


class TestModelInit:
    def test_init_basic(self, model):
        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert hasattr(model, "fsq")

    def test_init_various_levels(self, mp):
        for levels in [[3, 3], [7], [8, 6, 5], [5, 5, 5, 5, 5]]:
            cfg = make_config(levels=levels)
            model = VQDinoAutoencoder(cfg, mp=mp, rngs=nnx.Rngs(0))
            assert model.fsq.codebook_size == cfg.codebook_size

    def test_fsq_is_not_nnx_module(self, model):
        """FSQ should be a plain class, not an nnx.Module (no learnable params)."""
        assert not isinstance(model.fsq, nnx.Module)


# ---- Encode / Decode shapes ---------------------------------------------


class TestEncodeDecodeShapes:
    def test_encode_shape(self, model, cfg):
        x = jnp.ones((2, 16, 64))  # (batch, patches, input_dim)
        z, aux = model.encode(x)
        assert z.shape == (2, cfg.num_latents, len(cfg.levels))

    def test_encode_aux_indices_shape(self, model, cfg):
        x = jnp.ones((2, 16, 64))
        _, aux = model.encode(x)
        assert aux["indices"].shape == (2, cfg.num_latents)

    def test_encode_aux_z_pre_shape(self, model, cfg):
        x = jnp.ones((2, 16, 64))
        _, aux = model.encode(x)
        assert aux["z_pre"].shape == (2, cfg.num_latents, len(cfg.levels))

    def test_decode_shape(self, model, cfg):
        z = jnp.ones((2, cfg.num_latents, len(cfg.levels)))
        x_hat = model.decode(z)
        assert x_hat.shape == (2, cfg.num_output_patches, 64)

    def test_encode_decode_roundtrip_shapes(self, model, cfg):
        x = jnp.ones((2, 16, 64))
        z, _ = model.encode(x)
        x_hat = model.decode(z)
        assert x_hat.shape == (2, cfg.num_output_patches, 64)

    def test_with_disposable_registers(self, mp):
        cfg = make_config(
            num_latents=4,
            num_output_patches=8,
            encoder_disposable_registers=2,
            decoder_disposable_registers=2,
        )
        model = VQDinoAutoencoder(cfg, mp=mp, rngs=nnx.Rngs(0))
        x = jnp.ones((2, 16, 64))
        z, aux = model.encode(x)
        assert z.shape == (2, 4, len(cfg.levels))
        assert aux["indices"].shape == (2, 4)
        x_hat = model.decode(z)
        assert x_hat.shape == (2, 8, 64)


# ---- Quantization properties --------------------------------------------


class TestQuantization:
    def test_output_is_discrete(self, model, cfg):
        """Quantized output should round-trip through indices."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (4, 16, 64))
        z, aux = model.encode(x)
        indices = aux["indices"]
        recovered = model.fsq.indices_to_codes(indices)
        np.testing.assert_allclose(recovered, z, atol=1e-5)

    def test_indices_in_range(self, model, cfg):
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (4, 16, 64))
        _, aux = model.encode(x)
        indices = aux["indices"]
        assert jnp.all(indices >= 0)
        assert jnp.all(indices < cfg.codebook_size)

    def test_quantized_in_range(self, model):
        key = jax.random.PRNGKey(1)
        x = jax.random.normal(key, (4, 16, 64))
        z, _ = model.encode(x)
        assert jnp.all(z >= -1.0)
        assert jnp.all(z <= 1.0)


# ---- Gradient flow (STE) ------------------------------------------------


class TestGradientFlow:
    def test_gradients_flow_through_encoder(self, model):
        """Gradients should flow through FSQ via straight-through estimator."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (2, 16, 64))

        def loss_fn(model, x):
            z, _ = model.encode(x)
            return jnp.mean(z**2)

        grads = nnx.grad(loss_fn)(model, x)
        encoder_grads = nnx.state(grads.encoder, nnx.Param)
        has_nonzero = any(
            jnp.any(v != 0) for v in jax.tree.leaves(encoder_grads)
        )
        assert has_nonzero, "Encoder should receive gradients through STE"

    def test_gradients_flow_through_decoder(self, model, cfg):
        """Decoder should receive normal gradients (no STE needed)."""
        z = jnp.ones((2, cfg.num_latents, len(cfg.levels)))

        def loss_fn(model, z):
            x_hat = model.decode(z)
            return jnp.mean(x_hat**2)

        grads = nnx.grad(loss_fn)(model, z)
        decoder_grads = nnx.state(grads.decoder, nnx.Param)
        has_nonzero = any(
            jnp.any(v != 0) for v in jax.tree.leaves(decoder_grads)
        )
        assert has_nonzero, "Decoder should receive gradients"

    def test_end_to_end_gradients(self, model):
        """End-to-end loss should produce gradients for both encoder and decoder."""
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (2, 16, 64))

        def loss_fn(model, x):
            z, _ = model.encode(x)
            x_hat = model.decode(z)
            # Use MSE on decoder output (no shape mismatch with input)
            return jnp.mean(x_hat**2)

        grads = nnx.grad(loss_fn)(model, x)
        enc_grads = nnx.state(grads.encoder, nnx.Param)
        dec_grads = nnx.state(grads.decoder, nnx.Param)
        assert any(jnp.any(v != 0) for v in jax.tree.leaves(enc_grads))
        assert any(jnp.any(v != 0) for v in jax.tree.leaves(dec_grads))


# ---- trainable_pytree / get_state ----------------------------------------


class TestInterface:
    def test_trainable_pytree_keys(self, model):
        pytree = model.trainable_pytree
        assert set(pytree.keys()) == {"encoder", "decoder"}

    def test_trainable_pytree_has_no_fsq(self, model):
        """FSQ should not appear in trainable pytree (no learnable params)."""
        pytree = model.trainable_pytree
        assert "fsq" not in pytree

    def test_state_captures_all_components(self, model):
        """nnx.state should capture encoder and decoder params."""
        pure = nnx.to_pure_dict(nnx.state(model))
        assert "encoder" in pure
        assert "decoder" in pure

    def test_trainable_state(self, model):
        state = model.trainable_state
        # Should be a nested state dict with params from encoder and decoder
        flat = state.flat_state()
        assert len(flat) > 0


# ---- Experiment parsing (from train_vq) ----------------------------------


class TestExperimentParsing:
    def test_parse_basic(self):
        from flatdino.train_vq import parse_vq_experiment

        spec = parse_vq_experiment("fast-32-sb-L8555")
        assert spec is not None
        assert spec.variant == "fast"
        assert spec.toks == 32
        assert spec.enc == "s"
        assert spec.dec == "b"
        assert spec.levels == [8, 5, 5, 5]
        assert spec.half_layers is False

    def test_parse_5_levels(self):
        from flatdino.train_vq import parse_vq_experiment

        spec = parse_vq_experiment("med-32-tb-L75555")
        assert spec is not None
        assert spec.levels == [7, 5, 5, 5, 5]

    def test_parse_half_layers(self):
        from flatdino.train_vq import parse_vq_experiment

        spec = parse_vq_experiment("fast-32-sb-L8555-hl")
        assert spec is not None
        assert spec.half_layers is True

    def test_parse_disposable_registers(self):
        from flatdino.train_vq import parse_vq_experiment

        spec = parse_vq_experiment("fast-32-sb-L8555-er4-dr2")
        assert spec is not None
        assert spec.encoder_disposable_registers == 4
        assert spec.decoder_disposable_registers == 2

    def test_parse_invalid(self):
        from flatdino.train_vq import parse_vq_experiment

        assert parse_vq_experiment("invalid-name") is None
        assert parse_vq_experiment("baseline") is None

    def test_get_experiment_baseline(self):
        from flatdino.train_vq import get_experiment

        cfg = get_experiment("baseline")
        assert isinstance(cfg, VQDinoConfig)

    def test_get_experiment_dynamic(self):
        from flatdino.train_vq import get_experiment

        cfg = get_experiment("fast-32-sb-L8555")
        assert isinstance(cfg, VQDinoConfig)
        assert cfg.levels == [8, 5, 5, 5]
        assert cfg.codebook_size == 1000
        assert cfg.num_latents == 32

    def test_get_experiment_invalid(self):
        from flatdino.train_vq import get_experiment

        with pytest.raises(ValueError):
            get_experiment("not-a-valid-experiment")

    def test_build_config_encoder_output_dim(self):
        from flatdino.train_vq import get_experiment

        cfg = get_experiment("fast-32-sb-L88865")
        assert cfg.encoder.output_dim == 5  # 5 levels
        assert cfg.decoder.input_dim == 5
        assert cfg.levels == [8, 8, 8, 6, 5]
        assert cfg.codebook_size == 8 * 8 * 8 * 6 * 5  # 15360

    def test_parse_nd_experiment(self):
        from flatdino.train_vq import parse_vq_experiment

        spec = parse_vq_experiment("fast-32-sb-L8555-nd")
        assert spec is not None
        assert spec.nested_dropout is True
        assert spec.variant == "fast"
        assert spec.toks == 32

    def test_parse_nd_with_hl(self):
        from flatdino.train_vq import parse_vq_experiment

        spec = parse_vq_experiment("med-64-tb-L99555-hl-nd")
        assert spec is not None
        assert spec.nested_dropout is True
        assert spec.half_layers is True

    def test_build_nd_config_decoder_input_dim(self):
        from flatdino.train_vq import get_experiment

        cfg = get_experiment("fast-32-sb-L8555-nd")
        assert cfg.nested_dropout is True
        # decoder input_dim should equal embed_dim (no input_proj needed)
        assert cfg.decoder.input_dim == cfg.decoder.transformer.embed_dim

    def test_build_nd_config_without_nd(self):
        from flatdino.train_vq import get_experiment

        cfg = get_experiment("fast-32-sb-L8555")
        assert cfg.nested_dropout is False
        # decoder input_dim should equal num_levels
        assert cfg.decoder.input_dim == len(cfg.levels)


# ---- Nested dropout -------------------------------------------------------


def make_nd_config(
    levels: list[int] = [8, 5, 5, 5],
    num_latents: int = 4,
    num_output_patches: int = 8,
    embed_dim: int = 32,
    num_layers: int = 2,
    num_heads: int = 1,
) -> VQDinoConfig:
    """Create a small VQDinoConfig with nested_dropout=True for testing."""
    num_levels = len(levels)
    return VQDinoConfig(
        levels=levels,
        nested_dropout=True,
        encoder=ViTConfig(
            patch=None,
            num_patches=16,
            input_dim=64,
            num_registers=num_latents,
            output_dim=num_levels,
            transformer=TransformerConfig(
                embed_dim=embed_dim,
                num_layers=num_layers,
                mlp_hidden_dim=embed_dim * 4,
                num_heads=num_heads,
            ),
        ),
        decoder=ViTConfig(
            patch=None,
            num_patches=num_latents,
            input_dim=embed_dim,  # equals embed_dim → no input_proj in decoder
            num_registers=num_output_patches,
            output_dim=64,
            transformer=TransformerConfig(
                embed_dim=embed_dim,
                num_layers=num_layers,
                mlp_hidden_dim=embed_dim * 4,
                num_heads=num_heads,
            ),
        ),
    )


class TestNestedDropout:
    @pytest.fixture
    def nd_cfg(self):
        return make_nd_config()

    @pytest.fixture
    def nd_model(self, nd_cfg, mp):
        return VQDinoAutoencoder(nd_cfg, mp=mp, rngs=nnx.Rngs(0))

    def test_causal_mask_shape(self, nd_model, nd_cfg):
        mask = nd_model._encoder_causal_mask()
        R = nd_cfg.encoder.num_registers
        N = nd_cfg.encoder.num_patches
        assert mask.shape == (1, 1, R + N, R + N)

    def test_causal_mask_structure(self, nd_model, nd_cfg):
        mask = nd_model._encoder_causal_mask()
        R = nd_cfg.encoder.num_registers
        mask_2d = mask[0, 0]

        # Register↔Register: lower-triangular (causal)
        reg_block = mask_2d[:R, :R]
        expected_causal = jnp.tril(jnp.ones((R, R), dtype=jnp.bool_))
        np.testing.assert_array_equal(reg_block, expected_causal)

        # Register→Patch: all True
        assert jnp.all(mask_2d[:R, R:])

        # Patch→Register: all True
        assert jnp.all(mask_2d[R:, :R])

        # Patch↔Patch: all True
        assert jnp.all(mask_2d[R:, R:])

    def test_mask_token_shape(self, nd_model, nd_cfg):
        embed_dim = nd_cfg.decoder.transformer.embed_dim
        assert nd_model.mask_token.value.shape == (1, 1, embed_dim)

    def test_latent_proj_exists(self, nd_model):
        assert nd_model.latent_proj is not None

    def test_no_mask_token_when_disabled(self, model):
        """Standard model (nested_dropout=False) should have no mask_token."""
        assert model.mask_token is None
        assert model.latent_proj is None

    def test_no_dropout_without_key(self, nd_model):
        """encode() without key → keep_mask is None in aux."""
        x = jnp.ones((2, 16, 64))
        _, aux = nd_model.encode(x)
        assert aux["keep_mask"] is None
        assert aux["k_keep"] is None

    def test_no_dropout_when_disabled(self, model):
        """nested_dropout=False + key → keep_mask is None."""
        x = jnp.ones((2, 16, 64))
        key = jax.random.PRNGKey(0)
        _, aux = model.encode(x, key=key)
        assert aux["keep_mask"] is None
        assert aux["k_keep"] is None

    def test_encode_with_key_returns_keep_mask(self, nd_model, nd_cfg):
        x = jnp.ones((2, 16, 64))
        key = jax.random.PRNGKey(42)
        _, aux = nd_model.encode(x, key=key)
        assert aux["keep_mask"] is not None
        assert aux["k_keep"] is not None
        K = nd_cfg.num_latents
        assert aux["keep_mask"].shape == (1, K, 1)

    def test_keep_mask_is_pow2(self, nd_model, nd_cfg):
        """k_keep should always be a power of 2."""
        K = nd_cfg.num_latents
        for seed in range(20):
            key = jax.random.PRNGKey(seed)
            keep_mask, k_keep = nd_model._sample_keep_mask(K, key)
            k = int(k_keep)
            assert k > 0 and (k & (k - 1)) == 0, f"k_keep={k} is not a power of 2"
            assert k <= K

    def test_nested_dropout_replaces_tokens(self, nd_model, nd_cfg):
        """Masked tokens should be replaced with mask_token in decode."""
        K = nd_cfg.num_latents
        z = jnp.ones((1, K, len(nd_cfg.levels)))
        # Keep only first token
        keep_mask = jnp.arange(K)[None, :, None] < 1

        out_with_mask = nd_model.decode(z, keep_mask=keep_mask)
        out_without_mask = nd_model.decode(z)
        # Outputs should differ since masked tokens use mask_token
        assert not jnp.allclose(out_with_mask, out_without_mask)

    def test_trainable_pytree_includes_mask_token(self, nd_model):
        pytree = nd_model.trainable_pytree
        assert "mask_token" in pytree
        assert "latent_proj" in pytree

    def test_state_captures_nd_components(self, nd_model):
        """nnx.state should capture mask_token and latent_proj when nested_dropout."""
        pure = nnx.to_pure_dict(nnx.state(nd_model))
        assert "mask_token" in pure
        assert "latent_proj" in pure
        assert "encoder" in pure
        assert "decoder" in pure

    def test_gradient_flow_through_mask_token(self, nd_model, nd_cfg):
        """mask_token should receive gradients when used in decode."""
        K = nd_cfg.num_latents
        z = jnp.ones((1, K, len(nd_cfg.levels)))
        keep_mask = jnp.arange(K)[None, :, None] < 1  # keep only first

        def loss_fn(model, z):
            x_hat = model.decode(z, keep_mask=keep_mask)
            return jnp.mean(x_hat**2)

        grads = nnx.grad(loss_fn)(nd_model, z)
        mask_grad = grads.mask_token.value
        assert jnp.any(mask_grad != 0), "mask_token should receive gradients"

    def test_encode_decode_shapes_with_nd(self, nd_model, nd_cfg):
        """Full encode-decode with nested dropout should produce correct shapes."""
        x = jnp.ones((2, 16, 64))
        key = jax.random.PRNGKey(0)
        z, aux = nd_model.encode(x, key=key)
        assert z.shape == (2, nd_cfg.num_latents, len(nd_cfg.levels))
        x_hat = nd_model.decode(z, keep_mask=aux["keep_mask"])
        assert x_hat.shape == (2, nd_cfg.num_output_patches, 64)
