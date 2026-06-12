"""Tests for CodebookRegistry."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from rfsn_v11.polar_fused.codebooks import CodebookRegistry


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_centroids_shape() -> None:
    reg = CodebookRegistry("polar_lm_v1")
    for bits in (2, 3, 4):
        c = reg.centroids(bits)
        assert c.shape == (2 ** bits,)
        assert c.dtype in (mx.float32, mx.float16)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_boundaries_shape() -> None:
    reg = CodebookRegistry("polar_lm_v1")
    for bits in (2, 3, 4):
        b = reg.boundaries(bits)
        assert b.shape == (2 ** bits + 1,)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_quantize_dequantize_roundtrip() -> None:
    reg = CodebookRegistry("polar_lm_v1")
    for bits in (2, 3, 4):
        # Sample values near centroids
        centroids = reg.centroids(bits)
        # Pick every centroid directly
        indices = reg.quantize(centroids, bits)
        recon = reg.dequantize(indices, bits)
        assert mx.allclose(recon, centroids, atol=1e-5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_checksum_stability() -> None:
    reg = CodebookRegistry("polar_lm_v1")
    c1 = reg.checksum(4)
    c2 = reg.checksum(4)
    assert c1 == c2
    assert len(c1) == 64


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_codebook_id_format() -> None:
    reg = CodebookRegistry("polar_lm_v1")
    assert reg.codebook_id(4) == "polar_lm_v1_4bit"
    assert reg.codebook_id(3) == "polar_lm_v1_3bit"


def test_unknown_version() -> None:
    with pytest.raises(ValueError, match="Unknown codebook version"):
        CodebookRegistry("nonexistent_v999")


def test_invalid_bits() -> None:
    reg = CodebookRegistry("polar_lm_v1")
    with pytest.raises(ValueError, match="bits must be"):
        reg.centroids(5)
