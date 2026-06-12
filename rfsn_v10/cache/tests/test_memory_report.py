"""Tests for detailed memory measurement."""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_memory_report_after_appends() -> None:
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    session = GenerationCacheSession("test", 2, k_codec, v_codec, staging_capacity=32)

    # Append tokens
    for _ in range(4):
        keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        lc = session.get_layer_cache(0)
        lc.append(keys, values)

    report = session.memory_report()

    # Should have payload bytes
    assert report.payload_bytes > 0
    assert report.total_tokens > 0

    # Should have accounted for all categories
    assert report.total_accounted_bytes >= report.payload_bytes

    # Dict roundtrip
    d = report.to_dict()
    assert "payload_bytes" in d
    assert "compression_ratio" in d
