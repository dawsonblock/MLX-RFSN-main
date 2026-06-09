#!/usr/bin/env python3
"""Main12 Metal WHT64 correctness tests."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kernels import wht64_metal
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager


def test_wht64_metal_matches_reference_small(tmp_path) -> None:
    manager = RFSNTurboQuantKVManager(cache_dir=str(tmp_path))
    mx.random.seed(42)

    x = mx.random.normal((1, 1, 128, 64)).astype(mx.float32)
    y_metal = wht64_metal(x)
    y_ref = manager._apply_wht_pretransform(x)

    mx.eval(y_metal, y_ref)
    diff = mx.max(mx.abs(y_metal - y_ref)).item()
    assert diff < 1e-4


def test_wht64_metal_matches_reference_large(tmp_path) -> None:
    manager = RFSNTurboQuantKVManager(cache_dir=str(tmp_path))
    mx.random.seed(7)

    x = mx.random.normal((1, 8, 512, 64)).astype(mx.float32)
    y_metal = wht64_metal(x)
    y_ref = manager._apply_wht_pretransform(x)

    mx.eval(y_metal, y_ref)
    diff = mx.max(mx.abs(y_metal - y_ref)).item()
    assert diff < 1e-4


def test_wht64_metal_is_self_inverse() -> None:
    mx.random.seed(123)
    x = mx.random.normal((1, 4, 256, 64)).astype(mx.float32)

    y = wht64_metal(x)
    z = wht64_metal(y)

    mx.eval(z)
    diff = mx.max(mx.abs(x - z)).item()
    assert diff < 1e-4


def test_wht64_metal_rejects_invalid_last_dim() -> None:
    x = mx.random.normal((1, 1, 128, 80)).astype(mx.float32)
    with pytest.raises(ValueError, match="multiple of 64"):
        wht64_metal(x)


def test_wht64_metal_rejects_empty_tensor() -> None:
    x = mx.zeros((0,), dtype=mx.float32)
    with pytest.raises(ValueError, match="empty"):
        wht64_metal(x)
