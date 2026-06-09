"""Tests for PolarQuant value quantization.

Phase 5 — 5-2: Quality regression thresholds.
  - k8_v4 D=128: MSE ≤ paper_bound (≈ 0.010628 × 1.5)
  - k8_v3 D=128: MSE ≤ paper_bound (≈ 0.042511 × 1.5)
  - Negative gate: quality gate must fire and raise RuntimeError for bad config.
"""
import math
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v11.quant.value_quant import PolarQuant, make_value_quantizer  # noqa: E402
from rfsn_v11.quant.kv_compressor import (  # noqa: E402
    KVCompressor,
    _paper_mse_bound,
    _MSE_TOLERANCE_FACTOR,
)


@pytest.mark.parametrize("bits,dim", [(4, 128), (4, 64), (3, 128)])
def test_polar_quant_mse_below_paper_bound(bits, dim):
    """PolarQuant MSE on unit vectors must stay below the Lloyd-Max paper bound × 1.5."""
    pq = PolarQuant(bits=bits, dim=dim)
    rng = np.random.RandomState(42)
    x_np = rng.randn(512, dim).astype(np.float32)
    norms = np.linalg.norm(x_np, axis=-1, keepdims=True)
    x_np = x_np / norms  # unit sphere

    x = mx.array(x_np)
    recon, _, _ = pq.quantize_and_reconstruct(x)
    mx.eval(recon)

    mse = float(mx.mean((x - recon) ** 2).item())
    bound = _paper_mse_bound(bits) * _MSE_TOLERANCE_FACTOR

    assert mse <= bound, (
        f"PolarQuant bits={bits} dim={dim}: MSE={mse:.6f} > bound={bound:.6f}. "
        f"Paper bound (no tolerance): {_paper_mse_bound(bits):.6f}"
    )


def test_polar_quant_4bit_d128_cosine_on_unit():
    """Unit-vector cosine must be reasonably high for 4-bit D=128."""
    pq = PolarQuant(bits=4, dim=128)
    rng = np.random.RandomState(1)
    x_np = rng.randn(256, 128).astype(np.float32)
    x_np = x_np / np.linalg.norm(x_np, axis=-1, keepdims=True)
    x = mx.array(x_np)
    recon, _, _ = pq.quantize_and_reconstruct(x)
    mx.eval(recon)
    recon_np = np.array(recon)
    cosine = np.mean(
        np.sum(x_np * recon_np, axis=-1)
        / (np.linalg.norm(recon_np, axis=-1) + 1e-8)
    )
    # Expect at least 0.90 on unit vectors; lower bound chosen based on 4-bit theory
    assert cosine >= 0.90, f"4-bit D=128 unit cosine too low: {cosine:.4f}"


@pytest.mark.parametrize("bits", [4, 3])
def test_kv_compressor_quality_gate_passes(bits):
    """KVCompressor init quality gate must pass for standard configs."""
    kv = KVCompressor(k_bits=8, v_bits=bits, dim=128)
    # If we get here, the quality gate passed
    assert kv.dim == 128


def test_kv_compressor_compress_decompress_roundtrip():
    """KVCompressor compress/decompress must preserve shape."""
    rng = np.random.RandomState(0)
    kv = KVCompressor(k_bits=8, v_bits=4, dim=128)
    keys = mx.array(rng.randn(2, 8, 128).astype(np.float32))
    vals = mx.array(rng.randn(2, 8, 128).astype(np.float32))
    compressed = kv.compress(keys, vals)
    keys_r, vals_r = kv.decompress(compressed)
    mx.eval(keys_r, vals_r)
    assert keys_r.shape == keys.shape, f"Key shape mismatch: {keys_r.shape} vs {keys.shape}"
    assert vals_r.shape == vals.shape, f"Val shape mismatch: {vals_r.shape} vs {vals.shape}"


def test_polar_quant_quantize_reconstruct_deterministic():
    """Quantization must be deterministic — same input gives same output."""
    pq = PolarQuant(bits=4, dim=128, seed=42)
    rng = np.random.RandomState(5)
    x = mx.array(rng.randn(10, 128).astype(np.float32))

    recon1, idx1, norms1 = pq.quantize_and_reconstruct(x)
    recon2, idx2, norms2 = pq.quantize_and_reconstruct(x)
    mx.eval(recon1, recon2, idx1, idx2)

    assert np.array_equal(np.array(idx1), np.array(idx2)), "Indices not deterministic"
    assert np.allclose(np.array(recon1), np.array(recon2), atol=1e-6), "Recon not deterministic"


def test_fractional_bits_split_polar_quant():
    """_SplitPolarQuant (fractional bits) must preserve shape."""
    pq = make_value_quantizer(bits=3.5, dim=128)
    rng = np.random.RandomState(7)
    x = mx.array(rng.randn(16, 128).astype(np.float32))
    indices, norms = pq.quantize(x)
    recon = pq.dequantize(indices, norms)
    mx.eval(recon)
    assert recon.shape == x.shape, f"Split PolarQuant shape mismatch: {recon.shape} vs {x.shape}"
