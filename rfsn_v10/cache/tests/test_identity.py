"""Identity tests for QuantizedLayerCache and adapter.

These tests use deterministic values where each position encodes
layer × 1_000_000 + head × 10_000 + token × 100 + dimension.
This lets us verify that every reconstructed vector belongs to the
correct head and token after multiple appends, flushes, and trims.
"""
from __future__ import annotations

import pytest

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False


def _make_identity_tensor(B: int, Hkv: int, T: int, D: int, layer_id: int = 0) -> mx.array:  # noqa: N803
    """Create a tensor where each head/token has a unique, broadcast value.

    Every element for a given (head, token) has the same value:
        layer_id * 1000 + head * 100 + token

    This guarantees that within any quantization group of 64 elements,
    all values are identical, yielding EXACT round-trip reconstruction
    regardless of bit width.  The test therefore verifies tensor
    ordering, not quantization quality.
    """
    base = layer_id * 1000
    vals = []
    for b in range(B):
        for h in range(Hkv):
            for t in range(T):
                val = float(base + h * 100 + t)
                for _d in range(D):
                    vals.append(val)
    arr = mx.array(vals, dtype=mx.float32).reshape(B, Hkv, T, D)
    return arr


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_single_append_exact_reconstruction() -> None:
    """One append of 10 tokens with 2 heads; exact after decode."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    B, Hkv, T, D = 1, 2, 10, 64
    keys = _make_identity_tensor(B, Hkv, T, D, layer_id=0)
    values = _make_identity_tensor(B, Hkv, T, D, layer_id=0)

    cache.append(keys, values)

    # Not flushed yet
    assert cache.stats().staged_tokens == T
    assert cache.stats().sealed_blocks == 0

    # Reconstruct from staging
    stage_k, stage_v, stage_n = cache.get_staging()
    assert stage_k is not None
    assert stage_v is not None
    assert stage_n == T

    # Verify exact shape
    assert stage_k.shape == (B, Hkv, T, D)
    assert stage_v.shape == (B, Hkv, T, D)

    # Verify each head has correct token-major ordering
    for h in range(Hkv):
        for t in range(T):
            for d in range(D):
                expected = keys[0, h, t, d].item()
                actual = stage_k[0, h, t, d].item()
                assert actual == pytest.approx(expected, abs=1.5), (
                    f"head={h} token={t} dim={d}: expected {expected}, got {actual}"
                )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_multiple_appends_and_flush_preserves_identity() -> None:
    """Multiple appends trigger flush; verify sealed block reconstruction."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32)

    B, Hkv, D = 1, 4, 64
    all_keys = []
    all_values = []

    # Append 5 batches of 10 tokens each = 50 tokens
    for i in range(5):
        keys = _make_identity_tensor(B, Hkv, 10, D, layer_id=i)
        values = _make_identity_tensor(B, Hkv, 10, D, layer_id=i)
        all_keys.append(keys)
        all_values.append(values)
        cache.append(keys, values)

    # 5 x 10 = 50 tokens.  Fixed-size flush encodes one 32-token block,
    # leaving 18 tokens in staging.
    stats = cache.stats()
    assert stats.staged_tokens == 18
    assert stats.sealed_blocks == 1
    assert stats.tokens_encoded == 32
    assert stats.tokens_requantized == 0

    # Reconstruct sealed blocks + staging via adapter-like logic
    key_parts = []
    value_parts = []
    for kb in cache.iter_key_blocks():
        k_flat = k_codec.decode(kb)
        block_T = kb.token_count
        k_reshaped = k_flat.reshape(B, Hkv, block_T, D)
        key_parts.append(k_reshaped)
    for vb in cache.iter_value_blocks():
        v_flat = v_codec.decode(vb)
        block_T = vb.token_count
        v_reshaped = v_flat.reshape(B, Hkv, block_T, D)
        value_parts.append(v_reshaped)

    # Include staging
    stage_k, stage_v, stage_n = cache.get_staging()
    if stage_k is not None:
        key_parts.append(stage_k)
        value_parts.append(stage_v)

    full_k = mx.concatenate(key_parts, axis=2)
    full_v = mx.concatenate(value_parts, axis=2)

    assert full_k.shape == (B, Hkv, 50, D)
    assert full_v.shape == (B, Hkv, 50, D)

    # Verify ordering: head-major within the concatenated batch
    expected = mx.concatenate(all_keys, axis=2)
    for h in range(Hkv):
        for t in range(50):
            for d in range(D):
                exp = expected[0, h, t, d].item()
                act = full_k[0, h, t, d].item()
                assert act == pytest.approx(exp, abs=1.5), (
                    f"head={h} token={t} dim={d}: expected {exp}, got {act}"
                )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_append_flush_append_no_requantize() -> None:
    """Append, flush, append more — second batch must not trigger recompression."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32)

    B, Hkv, D = 1, 2, 64

    # First batch: 32 tokens → flush
    keys1 = _make_identity_tensor(B, Hkv, 32, D, layer_id=0)
    values1 = _make_identity_tensor(B, Hkv, 32, D, layer_id=0)
    cache.append(keys1, values1)

    assert cache.stats().tokens_encoded == 32
    assert cache.stats().staged_tokens == 0

    # Second batch: 16 tokens → stays in staging
    keys2 = _make_identity_tensor(B, Hkv, 16, D, layer_id=1)
    values2 = _make_identity_tensor(B, Hkv, 16, D, layer_id=1)
    cache.append(keys2, values2)

    assert cache.stats().tokens_encoded == 32
    assert cache.stats().staged_tokens == 16
    assert cache.stats().tokens_requantized == 0

    # Reconstruct both sealed and staging
    key_parts = []
    for kb in cache.iter_key_blocks():
        k_flat = k_codec.decode(kb)
        k_reshaped = k_flat.reshape(B, Hkv, kb.token_count, D)
        key_parts.append(k_reshaped)
    stage_k, _stage_v, _stage_n = cache.get_staging()
    if stage_k is not None:
        key_parts.append(stage_k)

    full_k = mx.concatenate(key_parts, axis=2)
    assert full_k.shape == (B, Hkv, 48, D)

    # Verify first 32 tokens
    for h in range(Hkv):
        for t in range(32):
            for d in range(D):
                exp = keys1[0, h, t, d].item()
                act = full_k[0, h, t, d].item()
                assert act == pytest.approx(exp, abs=1.5)

    # Verify next 16 tokens (starting at position 32 on token axis)
    for h in range(Hkv):
        for t in range(16):
            for d in range(D):
                exp = keys2[0, h, t, d].item()
                act = full_k[0, h, 32 + t, d].item()
                assert act == pytest.approx(exp, abs=1.5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_dense_residual_mutual_exclusion() -> None:
    """With dense residual enabled, tokens must not appear in both dense and staging."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64, dense_residual_window=8)

    B, Hkv, D = 1, 2, 64

    # Append 12 tokens
    keys = _make_identity_tensor(B, Hkv, 12, D, layer_id=0)
    values = _make_identity_tensor(B, Hkv, 12, D, layer_id=0)
    cache.append(keys, values)

    stats = cache.stats()
    # 8 in dense, 4 evicted to staging
    assert stats.dense_residual_tokens == 8
    assert stats.staged_tokens == 4
    assert stats.tokens_encoded == 0

    total = cache.total_token_count()
    assert total == 12, f"Expected 12 total tokens, got {total}"

    # Verify dense contains the LAST 8 tokens
    dense_k, dense_v = cache.get_dense_residual()
    assert dense_k is not None
    assert dense_k.shape == (B, Hkv, 8, D)
    for h in range(Hkv):
        for t in range(8):
            for d in range(D):
                exp = keys[0, h, 4 + t, d].item()
                act = dense_k[0, h, t, d].item()
                assert act == pytest.approx(exp, abs=1.5)

    # Verify staging contains the FIRST 4 tokens
    stage_k, stage_v, stage_n = cache.get_staging()
    assert stage_k is not None
    assert stage_k.shape == (B, Hkv, 4, D)
    for h in range(Hkv):
        for t in range(4):
            for d in range(D):
                exp = keys[0, h, t, d].item()
                act = stage_k[0, h, t, d].item()
                assert act == pytest.approx(exp, abs=1.5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_trim_across_regions() -> None:
    """Trim must correctly drop tokens from sealed, staged, and dense."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32, dense_residual_window=0)

    B, Hkv, D = 1, 2, 64

    # Build: 32 sealed + 8 staged = 40 total
    keys1 = _make_identity_tensor(B, Hkv, 32, D, layer_id=0)
    cache.append(keys1, keys1)  # use keys as values for simplicity

    keys2 = _make_identity_tensor(B, Hkv, 8, D, layer_id=1)
    cache.append(keys2, keys2)

    assert cache.total_token_count() == 40

    # Trim is disabled in this release
    with pytest.raises(NotImplementedError):
        cache.trim(32)

    # Reset still works
    cache.reset()
    assert cache.total_token_count() == 0


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_partial_staging_trim() -> None:
    """Partial trim of staging must retain exactly the requested token count."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64, dense_residual_window=0)

    B, Hkv, D = 1, 2, 64

    # Build: 64 sealed (flush) + 12 staged = 76 total
    keys1 = _make_identity_tensor(B, Hkv, 64, D, layer_id=0)
    cache.append(keys1, keys1)

    keys2 = _make_identity_tensor(B, Hkv, 12, D, layer_id=1)
    cache.append(keys2, keys2)

    assert cache.total_token_count() == 76
    assert cache.stats().tokens_encoded == 64
    assert cache.stats().staged_tokens == 12

    # Trim is disabled in this release
    with pytest.raises(NotImplementedError):
        cache.trim(68)

    # Verify staging still intact (12 tokens)
    stage_k, stage_v, stage_n = cache.get_staging()
    assert stage_k is not None
    assert stage_k.shape == (B, Hkv, 12, D)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_many_kv_heads() -> None:
    """Verify correct ordering with 8 KV heads."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=64)

    B, Hkv, T, D = 1, 8, 10, 64
    keys = _make_identity_tensor(B, Hkv, T, D, layer_id=0)
    values = _make_identity_tensor(B, Hkv, T, D, layer_id=0)
    cache.append(keys, values)

    stage_k, _stage_v, stage_n = cache.get_staging()
    assert stage_k is not None
    assert stage_k.shape == (B, Hkv, T, D)

    for h in range(Hkv):
        for t in range(T):
            for d in range(D):
                exp = keys[0, h, t, d].item()
                act = stage_k[0, h, t, d].item()
                assert act == pytest.approx(exp, abs=1.5)


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_adapter_multiple_updates_preserves_identity() -> None:
    """Adapter's update_and_fetch must preserve head/token identity across calls."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.session import GenerationCacheSession
    from rfsn_v10.integrations.mlx_lm_adapter.adapter import RfsnQuantizedKVCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    session = GenerationCacheSession("test", 2, k_codec, v_codec)

    cache = RfsnQuantizedKVCache(
        layer_cache=session.get_layer_cache(0),
        session=session,
    )

    B, Hkv, D = 1, 4, 64
    all_keys = []

    # Three updates of 10 tokens each
    for i in range(3):
        keys = _make_identity_tensor(B, Hkv, 10, D, layer_id=i)
        values = _make_identity_tensor(B, Hkv, 10, D, layer_id=i)
        all_keys.append(keys)
        full_k, full_v = cache.update_and_fetch(keys, values)

    assert full_k.shape == (B, Hkv, 30, D)

    expected = mx.concatenate(all_keys, axis=2)
    for h in range(Hkv):
        for t in range(30):
            for d in range(D):
                exp = expected[0, h, t, d].item()
                act = full_k[0, h, t, d].item()
                assert act == pytest.approx(exp, abs=1.5), (
                    f"head={h} token={t} dim={d}: expected {exp}, got {act}"
                )


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_blockwise_attention_matches_dense() -> None:
    """Blockwise attention must match dense reference exactly (within quant error)."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.incremental_layer_cache import QuantizedLayerCache

    k_codec = CartesianCodec(bits=8, group_size=64)
    v_codec = CartesianCodec(bits=5, group_size=64)
    cache = QuantizedLayerCache(k_codec, v_codec, staging_capacity=32)

    # Append 40 tokens in two batches
    keys1 = mx.random.normal(shape=(1, 2, 20, 64)).astype(mx.float32)
    values1 = mx.random.normal(shape=(1, 2, 20, 64)).astype(mx.float32)
    cache.append(keys1, values1)

    keys2 = mx.random.normal(shape=(1, 2, 20, 64)).astype(mx.float32)
    values2 = mx.random.normal(shape=(1, 2, 20, 64)).astype(mx.float32)
    cache.append(keys2, values2)

    # Decode: single query token
    queries = mx.random.normal(shape=(1, 2, 1, 64)).astype(mx.float32)
    scale = 1.0 / (64 ** 0.5)

    # Blockwise
    out_bw = cache.blockwise_attention(queries, scale=scale)

    # Dense reference (sealed blocks + staging)
    parts_k = []
    parts_v = []
    for kb in cache.iter_key_blocks():
        k_flat = k_codec.decode(kb)
        parts_k.append(k_flat.reshape(1, 2, kb.token_count, 64))
    for vb in cache.iter_value_blocks():
        v_flat = v_codec.decode(vb)
        parts_v.append(v_flat.reshape(1, 2, vb.token_count, 64))
    stage_k, stage_v, _stage_n = cache.get_staging()
    if stage_k is not None:
        parts_k.append(stage_k)
        parts_v.append(stage_v)
    full_k = mx.concatenate(parts_k, axis=2)
    full_v = mx.concatenate(parts_v, axis=2)

    scores = mx.matmul(queries, full_k.swapaxes(2, 3)) * scale
    max_score = mx.max(scores, axis=-1, keepdims=True)
    exp_scores = mx.exp(scores - max_score)
    weights = exp_scores / mx.sum(exp_scores, axis=-1, keepdims=True)
    out_dense = mx.matmul(weights, full_v)

    max_err = mx.max(mx.abs(out_bw - out_dense)).item()
    assert max_err < 0.01, f"blockwise vs dense max_err={max_err}"


@pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
def test_decode_rejects_signature_mismatch() -> None:
    """decode_bhtd must raise when block codec_signature does not match codec."""
    from rfsn_v10.cache.cartesian_codec import CartesianCodec
    from rfsn_v10.cache.contracts import PackedBlockV4

    k_codec = CartesianCodec(bits=8, group_size=64)

    # Build a valid block manually but with a deliberately wrong signature
    block = PackedBlockV4(
        packed_codes=mx.zeros((1, 2, 4, 8), dtype=mx.uint32),
        scales=mx.ones((1, 2, 4, 1), dtype=mx.float32),
        format_version=4,
        tensor_layout="BHTD",
        packing_layout="VECTOR_ALIGNED_UINT32_V4",
        scale_layout="BHTG_V4",
        preconditioner="WHT64_HASH_SIGN_V1",
        batch_size=1,
        n_kv_heads=2,
        token_count=4,
        head_dim=64,
        logical_start=0,
        logical_end=4,
        bits=8,
        group_size=64,
        groups_per_vector=1,
        codes_per_word=4,
        words_per_vector=8,
        original_value_count=512,
        padded_value_count=512,
        original_dtype="float16",
        sign_seed=42,
        sign_algorithm="murmur32-avalanche-v1",
        layer_id=0,
        stream_id="K",
        codec_signature="rfsn-v4-8-64-WRONG",
    )

    with pytest.raises(ValueError, match="Codec signature mismatch"):
        k_codec.decode_bhtd(block)
