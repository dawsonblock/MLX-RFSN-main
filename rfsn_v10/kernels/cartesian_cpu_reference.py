"""CPU reference kernels for packed QK and SV.

These implement the exact same bit-extraction and scale-indexing equations
as the Metal shaders in ``kernels/metal/cartesian_qk.metal`` and
``kernels/metal/cartesian_sv.metal``.  They serve as:

  * A ground-truth specification that NumPy and Metal must both match.
  * A fallback path when Metal is unavailable.
  * A debugging tool to isolate Metal-vs-reference mismatches.

All indexing uses the same flat buffer layout as the Metal kernels.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def _extract_code(
    packed_codes: np.ndarray,
    b: int,
    hkv: int,
    k_pos: int,
    d: int,
    bits: int,
    D: int,
    Hkv: int,
    Lkv: int,
) -> int:
    """Extract one quantized code from packed_codes (exact Metal indexing).

    Parameters match the Metal shader:
      packed_codes: (B, Hkv, Lkv, words_per_vec)
      codes_per_word = 32 // bits
      words_per_vec = ceil(D / codes_per_word)
    """
    codes_per_word = 32 // bits
    words_per_vec = math.ceil(D / codes_per_word)
    mask = (1 << bits) - 1

    word_idx = d // codes_per_word
    bit_offset = (d % codes_per_word) * bits

    kv_offset = (b * Hkv + hkv) * Lkv + k_pos
    packed_idx = kv_offset * words_per_vec + word_idx
    word = int(packed_codes[b, hkv, k_pos, word_idx])
    code = int((word >> bit_offset) & mask)
    return code


def _dequantize_code(code: int, bits: int, scale: float) -> float:
    """Convert quantized code back to float (exact Metal arithmetic)."""
    qmax = (1 << (bits - 1)) - 1
    return (float(code) - float(qmax)) * scale


def cartesian_qk_cpu_reference(
    queries: np.ndarray,
    packed_codes: np.ndarray,
    scales: np.ndarray,
    bits: int,
    group_size: int,
    scale_factor: float,
) -> np.ndarray:
    """Compute QK scores on CPU using exact Metal indexing.

    Parameters
    ----------
    queries
        (B, Hq, Lq, D)
    packed_codes
        (B, Hkv, Lkv, words_per_vec)
    scales
        (B, Hkv, Lkv, n_groups) — per-token scales
    bits
        Quantization bit width.
    group_size
        Group size for scale indexing.
    scale_factor
        Attention scale (e.g. D ** -0.5).

    Returns
    -------
    scores
        (B, Hq, Lq, Lkv)
    """
    B, Hq, Lq, D = queries.shape
    _, Hkv, Lkv, _ = packed_codes.shape
    n_groups = math.ceil(D / group_size)

    scores = np.zeros((B, Hq, Lq, Lkv), dtype=np.float32)

    for b in range(B):
        for hq in range(Hq):
            hkv = hq * Hkv // Hq  # GQA mapping
            for k_pos in range(Lkv):
                scale_base = ((b * Hkv + hkv) * Lkv + k_pos) * n_groups
                for q_pos in range(Lq):
                    q_offset = queries[b, hq, q_pos]
                    score = 0.0
                    for d in range(D):
                        code = _extract_code(
                            packed_codes, b, hkv, k_pos, d,
                            bits, D, Hkv, Lkv,
                        )
                        group_idx = d // group_size
                        scale = scales[b, hkv, k_pos, group_idx]
                        k_val = _dequantize_code(code, bits, scale)
                        q_val = q_offset[d]
                        score += q_val * k_val
                    score *= scale_factor
                    scores[b, hq, q_pos, k_pos] = score

    return scores


def cartesian_sv_cpu_reference(
    weights: np.ndarray,
    packed_codes: np.ndarray,
    scales: np.ndarray,
    bits: int,
    group_size: int,
    head_dim: int,
) -> np.ndarray:
    """Compute weighted value sum on CPU using exact Metal indexing.

    Parameters
    ----------
    weights
        (B, Hq, Lq, Lkv)
    packed_codes
        (B, Hkv, Lkv, words_per_vec)
    scales
        (B, Hkv, Lkv, n_groups) — per-token scales
    bits
        Quantization bit width.
    group_size
        Group size for scale indexing.
    head_dim
        Head dimension D.

    Returns
    -------
    output
        (B, Hq, Lq, D)
    """
    B, Hq, Lq, Lkv = weights.shape
    _, Hkv, _, _ = packed_codes.shape
    D = head_dim
    n_groups = math.ceil(D / group_size)

    output = np.zeros((B, Hq, Lq, D), dtype=np.float32)

    for b in range(B):
        for hq in range(Hq):
            hkv = hq * Hkv // Hq
            for q_pos in range(Lq):
                for d in range(D):
                    result = 0.0
                    for k_pos in range(Lkv):
                        code = _extract_code(
                            packed_codes, b, hkv, k_pos, d,
                            bits, D, Hkv, Lkv,
                        )
                        group_idx = d // group_size
                        scale = scales[b, hkv, k_pos, group_idx]
                        v_val = _dequantize_code(code, bits, scale)
                        w = weights[b, hq, q_pos, k_pos]
                        result += w * v_val
                    output[b, hq, q_pos, d] = result

    return output
