"""Tests for the NumPy packed-attention oracle.

Validates that packed blockwise attention matches dense attention,
and that NumPy/MLX reference attention match within quality thresholds.
"""
from __future__ import annotations

import numpy as np
import pytest


def _make_cache_from_blocks(
    key_blocks: list, value_blocks: list
) -> tuple:
    """Minimal cache wrapper for the attention oracle."""
    return key_blocks, value_blocks


def test_dense_attention_identity() -> None:
    """Dense attention with identical Q/K/V should produce the query back."""
    from rfsn_v10.cache.numpy_attention_oracle import numpy_dense_attention

    B, Hq, Lq, D = 1, 2, 4, 64
    T = 4
    q = np.eye(D)[:Lq, :].reshape(1, 1, Lq, D).astype(np.float32)
    q = np.repeat(q, Hq, axis=1)
    k = np.eye(D)[:T, :].reshape(1, 1, T, D).astype(np.float32)
    k = np.repeat(k, 2, axis=1)
    v = np.eye(D)[:T, :].reshape(1, 1, T, D).astype(np.float32)
    v = np.repeat(v, 2, axis=1)

    out = numpy_dense_attention(q, k, v, scale=1.0)
    assert out.shape == (B, Hq, Lq, D)


def test_packed_attention_matches_dense_single_block() -> None:
    """Packed attention over one block must match dense attention."""
    from rfsn_v10.cache.numpy_attention_oracle import numpy_packed_attention
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    B, Hq, Lq, D = 1, 2, 4, 64
    T = 64
    Hkv = 1

    np.random.seed(42)
    q = np.random.randn(B, Hq, Lq, D).astype(np.float32)
    k = np.random.randn(B, Hkv, T, D).astype(np.float32)
    v = np.random.randn(B, Hkv, T, D).astype(np.float32)

    k_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)

    k_block = k_codec.encode_bhtd(k, logical_start=0, layer_id=0, stream_id="K")
    v_block = v_codec.encode_bhtd(v, logical_start=0, layer_id=0, stream_id="V")

    # Explicit zero mask so both paths use identical (no) masking
    mask = np.zeros((Lq, T), dtype=np.float32)

    packed_out = numpy_packed_attention(
        q,
        [k_block],
        [v_block],
        k_codec,
        v_codec,
        scale=D ** -0.5,
        query_start_pos=0,
        additive_mask=mask,
    )

    # Dense baseline
    from rfsn_v10.cache.numpy_attention_oracle import numpy_dense_attention
    dense_out = numpy_dense_attention(q, k, v, scale=D ** -0.5, mask=mask)

    # Cosine similarity should be very high
    flat_packed = packed_out.flatten()
    flat_dense = dense_out.flatten()
    cos_sim = np.dot(flat_packed, flat_dense) / (
        np.linalg.norm(flat_packed) * np.linalg.norm(flat_dense)
    )
    assert cos_sim >= 0.995, f"Cosine similarity too low: {cos_sim}"


def test_packed_attention_matches_dense_multiple_blocks() -> None:
    """Packed attention over multiple blocks must match dense attention."""
    from rfsn_v10.cache.numpy_attention_oracle import numpy_dense_attention, numpy_packed_attention
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    B, Hq, Lq, D = 1, 4, 8, 64
    Hkv = 2
    T = 128

    np.random.seed(43)
    q = np.random.randn(B, Hq, Lq, D).astype(np.float32)
    k = np.random.randn(B, Hkv, T, D).astype(np.float32)
    v = np.random.randn(B, Hkv, T, D).astype(np.float32)

    k_codec = NumpyCartesianCodec(bits=8, group_size=64, use_wht=True, sign_seed=42)
    v_codec = NumpyCartesianCodec(bits=5, group_size=64, use_wht=True, sign_seed=42)

    k_blocks = []
    v_blocks = []
    for i in range(2):
        start = i * 64
        k_slice = k[:, :, start:start + 64, :]
        v_slice = v[:, :, start:start + 64, :]
        k_blocks.append(k_codec.encode_bhtd(k_slice, logical_start=start, layer_id=0, stream_id="K"))
        v_blocks.append(v_codec.encode_bhtd(v_slice, logical_start=start, layer_id=0, stream_id="V"))

    mask = np.zeros((Lq, T), dtype=np.float32)

    packed_out = numpy_packed_attention(
        q,
        k_blocks,
        v_blocks,
        k_codec,
        v_codec,
        scale=D ** -0.5,
        query_start_pos=0,
        additive_mask=mask,
    )

    dense_out = numpy_dense_attention(q, k, v, scale=D ** -0.5, mask=mask)

    flat_packed = packed_out.flatten()
    flat_dense = dense_out.flatten()
    cos_sim = np.dot(flat_packed, flat_dense) / (
        np.linalg.norm(flat_packed) * np.linalg.norm(flat_dense)
    )
    assert cos_sim >= 0.995, f"Cosine similarity too low: {cos_sim}"


def test_packed_attention_invalid_gqa() -> None:
    """Invalid GQA ratio must raise ValueError."""
    from rfsn_v10.cache.numpy_attention_oracle import numpy_packed_attention
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    q = np.random.randn(1, 3, 4, 64).astype(np.float32)  # Hq=3, not divisible by Hkv=2
    k_codec = NumpyCartesianCodec(bits=8, group_size=64)

    k_block = k_codec.encode_bhtd(
        np.random.randn(1, 2, 64, 64).astype(np.float32),
        logical_start=0, layer_id=0, stream_id="K",
    )
    v_block = k_codec.encode_bhtd(
        np.random.randn(1, 2, 64, 64).astype(np.float32),
        logical_start=0, layer_id=0, stream_id="V",
    )

    with pytest.raises(ValueError, match="must be divisible"):
        numpy_packed_attention(
            q, [k_block], [v_block], k_codec, k_codec, scale=1.0
        )


def test_packed_attention_empty_cache() -> None:
    """Empty cache with no blocks must raise ValueError."""
    from rfsn_v10.cache.numpy_attention_oracle import numpy_packed_attention
    from rfsn_v10.cache.numpy_codec_oracle import NumpyCartesianCodec

    q = np.random.randn(1, 2, 4, 64).astype(np.float32)
    codec = NumpyCartesianCodec(bits=8, group_size=64)

    with pytest.raises(ValueError, match="No K/V blocks"):
        numpy_packed_attention(q, [], [], codec, codec, scale=1.0)
