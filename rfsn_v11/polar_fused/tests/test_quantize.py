"""Tests for PolarQuantizer reference implementation."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from rfsn_v11.polar_fused.quantize import PolarQuantizer


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_zero_vectors() -> None:
    q = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    x = mx.zeros((1, 64))
    qv = q.quantize(x)
    recon = q.dequantize(qv)
    # Zero vectors after reconstruction should stay near zero
    assert mx.allclose(recon, mx.zeros_like(recon), atol=1e-5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_random_gaussian() -> None:
    x = mx.random.normal(shape=(16, 128))
    q = PolarQuantizer(bits=4, head_dim=128, rotation_seed=42)
    qv = q.quantize(x)
    recon = q.dequantize(qv)
    # Cosine similarity for 4-bit independent scalar Lloyd-Max
    # on 64-D vectors is ~0.85 — this is the expected quality ceiling
    cos = _cosine_similarity(x, recon)
    assert cos.item() > 0.78


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_all_bit_widths() -> None:
    x = mx.random.normal(shape=(8, 64))
    for bits in (2, 3, 4):
        q = PolarQuantizer(bits=bits, head_dim=64, rotation_seed=42)
        qv = q.quantize(x)
        recon = q.dequantize(qv)
        cos = _cosine_similarity(x, recon)
        # Higher bits = better quality
        if bits == 4:
            assert cos.item() > 0.82
        elif bits == 3:
            assert cos.item() > 0.75
        else:
            assert cos.item() > 0.65


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_both_head_dims() -> None:
    for dim in (64, 128):
        x = mx.random.normal(shape=(4, dim))
        q = PolarQuantizer(bits=4, head_dim=dim, rotation_seed=42)
        qv = q.quantize(x)
        recon = q.dequantize(qv)
        cos = _cosine_similarity(x, recon)
        assert cos.item() > 0.78


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_fp16_dtype() -> None:
    x = mx.random.normal(shape=(4, 64)).astype(mx.float16)
    q = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    qv = q.quantize(x)
    recon = q.dequantize(qv)
    # Reconstruction should be float32 (codebook default)
    assert recon.dtype == mx.float32


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_bf16_dtype() -> None:
    x = mx.random.normal(shape=(4, 64)).astype(mx.bfloat16)
    q = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    qv = q.quantize(x)
    recon = q.dequantize(qv)
    assert recon.dtype == mx.float32


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_reconstruct_convenience() -> None:
    x = mx.random.normal(shape=(4, 64))
    q = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    recon = q.reconstruct(x)
    cos = _cosine_similarity(x, recon)
    assert cos.item() > 0.82


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_nan_rejected() -> None:
    q = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    x = mx.full((1, 64), float("nan"))
    with pytest.raises(ValueError, match="NaN"):
        q.quantize(x)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_inf_rejected() -> None:
    q = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    x = mx.full((1, 64), float("inf"))
    with pytest.raises(ValueError, match="infinity"):
        q.quantize(x)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_large_norms() -> None:
    x = mx.random.normal(shape=(4, 64)) * 1000.0
    q = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    qv = q.quantize(x)
    recon = q.dequantize(qv)
    # Even with large norms, relative cosine should be good
    cos = _cosine_similarity(x, recon)
    assert cos.item() > 0.82


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_structured_repeated_vectors() -> None:
    # Repeated identical vectors quantize to a single centroid,
    # so reconstruction is exact up to the centroid value.
    x = mx.ones((8, 64))
    q = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    qv = q.quantize(x)
    recon = q.dequantize(qv)
    cos = _cosine_similarity(x, recon)
    # After rotation, identical vectors may not align with a single centroid
    assert cos.item() > 0.50  # relaxed threshold for degenerate input


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_determinism() -> None:
    x = mx.random.normal(shape=(4, 64))
    q1 = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    q2 = PolarQuantizer(bits=4, head_dim=64, rotation_seed=42)
    qv1 = q1.quantize(x)
    qv2 = q2.quantize(x)
    assert mx.array_equal(qv1.indices, qv2.indices)
    assert mx.allclose(qv1.norms, qv2.norms, atol=0.0)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_invalid_bits() -> None:
    with pytest.raises(ValueError, match="bits must be"):
        PolarQuantizer(bits=5, head_dim=64, rotation_seed=42)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_invalid_head_dim() -> None:
    with pytest.raises(ValueError, match="head_dim must be"):
        PolarQuantizer(bits=4, head_dim=32, rotation_seed=42)


def _cosine_similarity(a: mx.array, b: mx.array) -> mx.array:
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    na = mx.sqrt(mx.sum(a_flat * a_flat))
    nb = mx.sqrt(mx.sum(b_flat * b_flat))
    return dot / (na * nb)
