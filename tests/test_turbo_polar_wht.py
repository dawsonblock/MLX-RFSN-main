#!/usr/bin/env python3
"""TurboPolar WHT roundtrip safety test."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.quantization.turbo_polar_quant import TurboPolarQuantizer


def test_turbo_polar_wht_shape_and_finite():
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))
    q = TurboPolarQuantizer(feature_dim=64)
    packed = q.quantize(k, v)
    rk, rv = q.dequantize(packed)
    mx.eval(rk, rv)
    assert rk.shape == k.shape
    assert rv.shape == v.shape
    assert bool(mx.all(mx.isfinite(rk)).item())
    assert bool(mx.all(mx.isfinite(rv)).item())
