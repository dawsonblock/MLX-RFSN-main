"""Tests for GenerationCacheSession — request-local isolation."""
from __future__ import annotations

import pytest

from rfsn_v10.cache.cartesian_codec import CartesianCodec
from rfsn_v10.cache.session import GenerationCacheSession

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def test_session_isolation() -> None:
    """Two sessions with same params must not share cache state."""
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)

    session_a = GenerationCacheSession("test-model", 4, k_codec, v_codec)
    session_b = GenerationCacheSession("test-model", 4, k_codec, v_codec)

    assert session_a.session_id != session_b.session_id
    assert session_a.total_memory_bytes() == 0
    assert session_b.total_memory_bytes() == 0

    # Write to A
    if HAS_MLX:
        keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
        session_a.get_layer_cache(0).append(keys, values)

    assert session_a.total_memory_bytes() > 0
    assert session_b.total_memory_bytes() == 0


def test_session_counters() -> None:
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec)

    session.increment("new_tokens_received", 10)
    session.increment("new_tokens_encoded", 8)

    assert session.get_counter("new_tokens_received") == 10
    assert session.get_counter("new_tokens_encoded") == 8
    assert session.get_counter("nonexistent") == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_session_position_ownership() -> None:
    """Each session must own independent, monotonically increasing positions."""
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.cache.cartesian_codec import CartesianCodec

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)

    session_a = GenerationCacheSession("test", 2, k_codec, v_codec)
    session_b = GenerationCacheSession("test", 2, k_codec, v_codec)

    # Append 80 tokens to A → one 64-token block (pos 0) + 16 staged
    keys = mx.random.normal(shape=(1, 2, 80, 64)).astype(mx.float32)
    values = mx.random.normal(shape=(1, 2, 80, 64)).astype(mx.float32)
    session_a.get_layer_cache(0).append(keys, values)

    # A owns logical_start 0
    a_starts = [b.logical_start for b in session_a.get_layer_cache(0).iter_key_blocks()]
    assert a_starts == [0], f"Expected [0], got {a_starts}"

    # B must have no blocks yet
    b_starts = [b.logical_start for b in session_b.get_layer_cache(0).iter_key_blocks()]
    assert b_starts == []

    # A's total must be exactly 80; B's must be 0
    assert session_a.get_layer_cache(0).total_token_count() == 80
    assert session_b.get_layer_cache(0).total_token_count() == 0


def test_session_context_manager() -> None:
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)

    with GenerationCacheSession("test-model", 2, k_codec, v_codec) as session:
        if HAS_MLX:
            keys = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
            values = mx.random.normal(shape=(1, 2, 10, 64)).astype(mx.float32)
            session.get_layer_cache(0).append(keys, values)
        assert len(session.all_layer_caches()) == 2

    # After exit, caches are destroyed
    assert len(session.all_layer_caches()) == 0
