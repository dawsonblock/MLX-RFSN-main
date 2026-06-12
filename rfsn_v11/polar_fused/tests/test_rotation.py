"""Tests for RotationRegistry."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from rfsn_v11.polar_fused.rotations import RotationRegistry


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_rotation_determinism() -> None:
    reg = RotationRegistry()
    r1 = reg.get(64, seed=42)
    r2 = reg.get(64, seed=42)
    assert mx.allclose(r1, r2, atol=0.0)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_different_seeds_different_matrices() -> None:
    reg = RotationRegistry()
    r1 = reg.get(64, seed=42)
    r2 = reg.get(64, seed=43)
    assert not mx.allclose(r1, r2, atol=1e-6)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_orthogonality_gate() -> None:
    reg = RotationRegistry()
    reg.validate(64, seed=42)
    reg.validate(128, seed=42)
    err_64 = reg.orthogonality_error(64, seed=42)
    err_128 = reg.orthogonality_error(128, seed=42)
    assert err_64 < 1e-5
    assert err_128 < 1e-5


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_checksum_consistency() -> None:
    reg = RotationRegistry()
    c1 = reg.checksum(64, seed=42)
    c2 = reg.checksum(64, seed=42)
    assert c1 == c2
    assert len(c1) == 64  # SHA-256 hex


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_transpose_cached() -> None:
    reg = RotationRegistry()
    r = reg.get(64, seed=42)
    rt = reg.get_transpose(64, seed=42)
    assert mx.allclose(r.T, rt, atol=1e-6)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_different_dims() -> None:
    reg = RotationRegistry()
    r64 = reg.get(64, seed=42)
    r128 = reg.get(128, seed=42)
    assert r64.shape == (64, 64)
    assert r128.shape == (128, 128)
