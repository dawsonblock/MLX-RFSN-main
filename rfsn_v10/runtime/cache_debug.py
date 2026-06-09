"""Cache alignment and invariant validation utilities for RFSN v10.

Provides helpers to validate cache shapes, RoPE positions, and attention
mask consistency between FP16 and compressed generation paths.
"""
from __future__ import annotations

from typing import Any


def validate_cache_state(cache, expected_seq_len: int, layer_id: int) -> dict[str, Any]:
    """Validate that a cache object's K/V tensors have the expected sequence length.

    Args:
        cache: A cache object with ``k`` and ``v`` attributes (tensors),
            or a ``DynamicCache`` with ``key_cache``/``value_cache`` lists.
        expected_seq_len: Expected sequence dimension.
        layer_id: Layer identifier for reporting.

    Returns:
        Dict with shape metadata and a ``passes`` boolean.
    """
    # Handle DynamicCache (key_cache/value_cache lists)
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        k = cache.key_cache[layer_id]
        v = cache.value_cache[layer_id]
    elif hasattr(cache, "k") and hasattr(cache, "v"):
        k = cache.k
        v = cache.v
    else:
        return {
            "layer_id": layer_id,
            "expected_seq_len": expected_seq_len,
            "error": "cache object has no recognizable k/v attributes",
            "passes": False,
        }
    k_shape = tuple(k.shape)
    v_shape = tuple(v.shape)
    actual_k_len = int(k.shape[-2])
    actual_v_len = int(v.shape[-2])
    return {
        "layer_id": layer_id,
        "expected_seq_len": expected_seq_len,
        "actual_k_len": actual_k_len,
        "actual_v_len": actual_v_len,
        "k_shape": k_shape,
        "v_shape": v_shape,
        "passes": (actual_k_len == expected_seq_len and actual_v_len == expected_seq_len),
    }


def validate_quantized_cache_packet(cache, expected_seq_len: int, layer_id: int) -> dict[str, Any]:
    """Validate a quantized cache packet including packed shapes and block metadata.

    Args:
        cache: A quantized cache object with packed/reconstructed shapes.
        expected_seq_len: Expected sequence dimension after reconstruction.
        layer_id: Layer identifier for reporting.

    Returns:
        Dict with packed metadata and a ``passes`` boolean.
    """
    result = {
        "layer_id": layer_id,
        "expected_seq_len": expected_seq_len,
        "passes": True,
    }
    if hasattr(cache, "original_shape"):
        result["original_shape"] = tuple(cache.original_shape)
    if hasattr(cache, "packed_shape"):
        result["packed_shape"] = tuple(cache.packed_shape)
    if hasattr(cache, "reconstructed_shape"):
        result["reconstructed_shape"] = tuple(cache.reconstructed_shape)
        rs = result["reconstructed_shape"]
        if len(rs) >= 3:
            actual_seq = rs[-2]
            result["actual_seq_len"] = actual_seq
            result["passes"] = (actual_seq == expected_seq_len)
    if hasattr(cache, "block_size"):
        result["block_size"] = int(cache.block_size)
    if hasattr(cache, "group_size"):
        result["group_size"] = int(cache.group_size)
    if hasattr(cache, "seq_len"):
        result["seq_len"] = int(cache.seq_len)
    if hasattr(cache, "head_dim"):
        result["head_dim"] = int(cache.head_dim)
    return result


def assert_cache_invariants(
    fp16_position_ids,
    quant_position_ids,
    fp16_attention_mask,
    quant_attention_mask,
    fp16_cache_len: int,
    quant_cache_len: int,
) -> None:
    """Hard assert that FP16 and compressed paths share identical position/mask state.

    Raises:
        AssertionError: If any invariant is violated.
    """
    if fp16_position_ids is not None and quant_position_ids is not None:
        assert fp16_position_ids.tolist() == quant_position_ids.tolist(), (
            f"position_ids mismatch: FP16 {fp16_position_ids.tolist()} != "
            f"quant {quant_position_ids.tolist()}"
        )
    if fp16_attention_mask is not None and quant_attention_mask is not None:
        assert fp16_attention_mask.shape == quant_attention_mask.shape, (
            f"attention_mask shape mismatch: FP16 {fp16_attention_mask.shape} != "
            f"quant {quant_attention_mask.shape}"
        )
    assert fp16_cache_len == quant_cache_len, (
        f"cache length mismatch: FP16 {fp16_cache_len} != quant {quant_cache_len}"
    )
