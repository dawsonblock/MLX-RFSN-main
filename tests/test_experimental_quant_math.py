#!/usr/bin/env python3
"""
Reference math tests for experimental IsoQuant + Polar + QJL.
These should pass before any Metal kernel is written.
"""
import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.quantization.hybrid_polar_cartesian import (
    HybridPolarCartesianQuantizer,
)
from rfsn_v10.quantization.isoquant_precondition import (
    IsoQuantPreconditioner,
)
from rfsn_v10.quantization.kv_quant_manager import QuantizedKVManager
from rfsn_v10.quantization.polar_quant import (
    PolarQuantizer,
    iterative_hierarchical_polar_forward,
    iterative_hierarchical_polar_inverse,
)


def _cos(a: mx.array, b: mx.array) -> float:
    a = a.reshape(-1).astype(mx.float32)
    b = b.reshape(-1).astype(mx.float32)
    return float(
        mx.sum(a * b)
        / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)) + 1e-8)
    )


def test_isoquant_roundtrip_exactish():
    x = mx.random.normal((1, 4, 128, 64))
    iso = IsoQuantPreconditioner(feature_dim=64)
    y, meta = iso.forward(x)
    x_rec = iso.inverse(y, meta)
    mx.eval(x_rec)
    assert x_rec.shape == x.shape
    assert _cos(x, x_rec) > 0.999999


def test_unquantized_polar_roundtrip_exactish():
    x = mx.random.normal((2, 128, 64))
    angles, radii = iterative_hierarchical_polar_forward(x, levels=4)
    x_rec = iterative_hierarchical_polar_inverse(angles, radii)
    mx.eval(x_rec)
    assert x_rec.shape == x.shape
    assert _cos(x, x_rec) > 0.999999


def test_quantized_polar_roundtrip_reasonable():
    x = mx.random.normal((2, 128, 64))
    q = PolarQuantizer(levels=4, angle_bits=5, radius_bits=8)
    packed = q.quantize(x)
    x_rec = q.dequantize(packed)
    mx.eval(x_rec)
    assert x_rec.shape == x.shape
    assert _cos(x, x_rec) > 0.98
    assert q.estimate_bytes(packed) > 0


def test_hybrid_roundtrip_shape_and_memory():
    x = mx.random.normal((1, 8, 128, 64))
    q = HybridPolarCartesianQuantizer(
        feature_dim=64,
        polar_ratio=0.65,
        polar_levels=4,
        polar_angle_bits=5,
        polar_radius_bits=8,
        cartesian_bits=6,
    )
    packed = q.quantize(x)
    x_rec = q.dequantize(packed)
    fp16_bytes = int(x.size) * 2
    compressed_bytes = q.estimate_bytes(packed)
    mx.eval(x_rec)
    assert x_rec.shape == x.shape
    assert compressed_bytes < fp16_bytes
    assert _cos(x, x_rec) > 0.97


def test_quantized_kv_manager_shape_and_ratio():
    k = mx.random.normal((1, 8, 128, 64))
    v = mx.random.normal((1, 8, 128, 64))
    manager = QuantizedKVManager(
        mode="hybrid_polar_cartesian",
        feature_dim=64,
        use_qjl_score_correction=False,
    )
    packet = manager.quantize(k, v)
    k_rec, v_rec = manager.dequantize(packet)
    mx.eval(k_rec, v_rec)
    assert k_rec.shape == k.shape
    assert v_rec.shape == v.shape
    assert manager.compression_ratio(packet) > 1.0
    assert _cos(k, k_rec) > 0.95
    assert _cos(v, v_rec) > 0.90


def test_qjl_score_correction_shape():
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))
    q = mx.random.normal((1, 4, 1, 64))
    manager = QuantizedKVManager(
        mode="hybrid_polar_cartesian",
        feature_dim=64,
        use_qjl_score_correction=True,
        qjl_proj_dim=64,
    )
    packet = manager.quantize(k, v)
    scores = manager.corrected_key_attention_scores(q, packet)
    mx.eval(scores)
    assert scores.shape == (1, 4, 1, 128)
    assert packet.uses_qjl is True
