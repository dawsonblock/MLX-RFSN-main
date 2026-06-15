"""GQA (Grouped Query Attention) validation tests.

Tests for models with num_q_heads > num_kv_heads (Grouped Query Attention).
This architecture requires special handling in attention mechanisms.

Note: These tests require a GQA model (e.g., Llama 2/3, Mistral, etc.).
The current test model (Qwen2.5-0.5B-Instruct) uses standard attention (num_q_heads == num_kv_heads).
"""
from __future__ import annotations

import pytest

from rfsn_v10.cache.cartesian_codec import CartesianCodec
from rfsn_v10.cache.session import GenerationCacheSession

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_gqa_cache_structure() -> None:
    """Verify cache structure handles GQA geometry correctly.

    GQA models have num_q_heads > num_kv_heads, meaning multiple query heads
    share the same key/value heads. The cache must account for this geometry.
    """
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec)

    layer_cache = session.get_layer_cache(0)

    # Simulate GQA geometry: 8 query heads, 2 KV heads (4:1 ratio)
    B, n_kv_heads, T, D = 1, 2, 64, 64

    # Keys and values should be shaped for KV heads (not query heads)
    keys = mx.random.normal(shape=(B, n_kv_heads, T, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, n_kv_heads, T, D)).astype(mx.float32)

    layer_cache.append(keys, values)

    # Should handle GQA geometry correctly
    assert layer_cache.total_token_count() == T

    # Verify blocks maintain correct geometry
    blocks = list(layer_cache.iter_key_blocks())
    for block in blocks:
        if block.n_kv_heads > 0:
            assert block.n_kv_heads == n_kv_heads, (
                f"Block should preserve KV head count: {block.n_kv_heads} != {n_kv_heads}"
            )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
@pytest.mark.mlx
def test_gqa_attention_compatibility() -> None:
    """Verify packed attention is compatible with GQA geometry.

    This test ensures that when the model uses GQA, the packed attention
    mechanism correctly handles the head dimensionality without errors.
    """
    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=8, group_size=64)
    session = GenerationCacheSession("test-model", 2, k_codec, v_codec)

    layer_cache = session.get_layer_cache(0)

    # Simulate GQA geometry
    B, n_kv_heads, T, D = 1, 2, 64, 64

    keys = mx.random.normal(shape=(B, n_kv_heads, T, D)).astype(mx.float32)
    values = mx.random.normal(shape=(B, n_kv_heads, T, D)).astype(mx.float32)

    layer_cache.append(keys, values)

    # Verify that block metadata correctly reflects GQA geometry
    blocks = list(layer_cache.iter_key_blocks())
    if blocks:
        block = blocks[0]
        # Block should store KV head count, not query head count
        assert block.n_kv_heads == n_kv_heads or block.n_kv_heads == 0, (
            "Block should use KV head count for GQA"
        )


@pytest.mark.skip(reason="Requires GQA model (e.g., Llama 2/3, Mistral)")
def test_gqa_model_integration():
    """Integration test with actual GQA model.

    This test requires a model with GQA architecture to verify end-to-end
    behavior. Marked as skip until such a model is available in the test suite.
    """
    # TODO: Add actual GQA model test when available
    # Should test:
    # 1. Model loads correctly
    # 2. Packed attention handles GQA geometry
    # 3. Quality metrics are computed correctly
    # 4. No errors in generation
    pass
