"""CPU reference kernels for packed QK and SV.

These implement the exact same bit-extraction and scale-indexing equations
as the Metal shaders in ``kernels/metal/cartesian_qk.metal`` and
``kernels/metal/cartesian_sv.metal``.  They serve as:

  * A ground-truth specification that NumPy and Metal must both match.
  * A fallback path when Metal is unavailable.
  * A debugging tool to isolate Metal-vs-reference mismatches.

When *use_wht* or *sign_seed* is set, the kernels inverse-transform K/V
back to the original domain before the dot-product / weighted-sum so
that the result matches canonical attention mathematics.
"""
from __future__ import annotations

import math

import numpy as np


def _token_hash_signs(
    token_flat_start: int,
    length: int,
    seed: int,
    layer_id: int = 0,
    stream_id: str = "",
) -> np.ndarray:
    """Generate hash signs for a single token at a specific global flat offset.

    Matches the sign pattern that ``_numpy_hash_signs`` produces after
    mixing layer_id and stream_id into the seed.
    """
    # Mix layer_id and stream_id into the seed (same algorithm as codec)
    stream_hash = 0
    for ch in stream_id:
        stream_hash = (stream_hash * 31 + ord(ch)) & 0xFFFFFFFF
    mixed = np.uint32(seed)
    mixed = np.uint32(mixed ^ np.uint32((layer_id * 0x9E3779B9) & 0xFFFFFFFF))
    mixed = np.uint32(mixed ^ np.uint32(stream_hash & 0xFFFFFFFF))
    seed_val = int(mixed) & 0xFFFFFFFF

    indices = np.arange(token_flat_start, token_flat_start + length, dtype=np.uint32)
    state = (indices ^ seed_val) & 0xFFFFFFFF
    state = (state + 0x9E3779B9) & 0xFFFFFFFF
    state = state ^ (state >> 16)
    state = (state * 0x85EBCA6B) & 0xFFFFFFFF
    state = state ^ (state >> 13)
    state = (state * 0xC2B2AE35) & 0xFFFFFFFF
    state = state ^ (state >> 16)
    signs = np.where((state & 1) == 1, -1.0, 1.0)
    return signs.astype(np.float32)


def _inverse_wht_signs_token(
    x: np.ndarray,
    use_wht: bool,
    sign_seed: int,
    token_flat_start: int,
    layer_id: int = 0,
    stream_id: str = "",
) -> np.ndarray:
    """Apply inverse WHT and/or hash signs to a single token's grouped values.

    *x* has shape ``(n_groups, group_size)``.  The signs are derived from
    the token's global flat position within the original block so they match
    the encode path exactly.
    """
    out = x.astype(np.float32)
    if sign_seed != 0:
        flat = out.reshape(-1)
        signs = _token_hash_signs(
            token_flat_start, flat.size, sign_seed, layer_id=layer_id, stream_id=stream_id
        )
        out = (flat * signs).reshape(out.shape)
    if use_wht:
        from rfsn_v10.cache.numpy_codec_oracle import _numpy_wht64
        out = _numpy_wht64(out)
    return out


def _extract_code(  # noqa: N803
    packed_codes: np.ndarray,
    b: int,
    hkv: int,
    k_pos: int,
    d: int,
    bits: int,
    D: int,  # noqa: N803
    Hkv: int,  # noqa: N803
    Lkv: int,  # noqa: N803
) -> int:
    """Extract one quantized code from packed_codes (exact Metal indexing).

    Parameters match the Metal shader:
      packed_codes: (B, Hkv, Lkv, words_per_vec)
      codes_per_word = 32 // bits
      words_per_vec = ceil(D / codes_per_word)
    """
    codes_per_word = 32 // bits
    mask = (1 << bits) - 1

    word_idx = d // codes_per_word
    bit_offset = (d % codes_per_word) * bits

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
    use_wht: bool = False,
    sign_seed: int = 0,
    layer_id: int = 0,
    stream_id: str = "",
) -> np.ndarray:
    """Compute QK scores on CPU using exact Metal indexing.

    When *use_wht* or *sign_seed* is set, K is inverse-transformed back to
    the original domain so the scores match canonical attention.

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
    use_wht
        Whether the packed codes were WHT-transformed.
    sign_seed
        Seed for deterministic hash signs (0 disables).
    layer_id
        Layer index for sign derivation (must match encode).
    stream_id
        Stream identifier for sign derivation (must match encode).

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
            if Hq % Hkv != 0:
                raise ValueError(
                    f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
                )
            hkv = hq * Hkv // Hq  # GQA mapping
            for k_pos in range(Lkv):
                # Dequantize the full K vector for this token
                k_vals = np.zeros(D, dtype=np.float32)
                for d in range(D):
                    code = _extract_code(
                        packed_codes, b, hkv, k_pos, d,
                        bits, D, Hkv, Lkv,
                    )
                    group_idx = d // group_size
                    scale = scales[b, hkv, k_pos, group_idx]
                    k_vals[d] = _dequantize_code(code, bits, scale)

                # Inverse-transform back to original domain if needed
                if use_wht or sign_seed != 0:
                    token_flat_start = (
                        (b * Hkv + hkv) * Lkv + k_pos
                    ) * (n_groups * group_size)
                    k_grouped = k_vals.reshape(n_groups, group_size)
                    k_grouped = _inverse_wht_signs_token(
                        k_grouped, use_wht, sign_seed, token_flat_start,
                        layer_id=layer_id, stream_id=stream_id,
                    )
                    k_vals = k_grouped.reshape(D)

                for q_pos in range(Lq):
                    q_offset = queries[b, hq, q_pos]
                    score = float(np.dot(q_offset.astype(np.float32), k_vals))
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
    use_wht: bool = False,
    sign_seed: int = 0,
    layer_id: int = 0,
    stream_id: str = "",
) -> np.ndarray:
    """Compute weighted value sum on CPU using exact Metal indexing.

    When *use_wht* or *sign_seed* is set, the accumulated result is
    inverse-transformed back to the original domain so the output
    matches canonical attention.

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
    use_wht
        Whether the packed codes were WHT-transformed.
    sign_seed
        Seed for deterministic hash signs (0 disables).
    layer_id
        Layer index for sign derivation (must match encode).
    stream_id
        Stream identifier for sign derivation (must match encode).

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
            if Hq % Hkv != 0:
                raise ValueError(
                    f"Hq ({Hq}) must be divisible by Hkv ({Hkv})"
                )
            hkv = hq * Hkv // Hq
            for q_pos in range(Lq):
                # Accumulate weighted V in the ORIGINAL domain.
                # Each token's V must be inverse-transformed individually before
                # the weighted sum because sign patterns differ per token.
                sv = np.zeros(D, dtype=np.float32)
                for k_pos in range(Lkv):
                    v_vals = np.zeros(D, dtype=np.float32)
                    for d in range(D):
                        code = _extract_code(
                            packed_codes, b, hkv, k_pos, d,
                            bits, D, Hkv, Lkv,
                        )
                        group_idx = d // group_size
                        scale = scales[b, hkv, k_pos, group_idx]
                        v_vals[d] = _dequantize_code(code, bits, scale)

                    # Inverse-transform this token's V back to original domain
                    if use_wht or sign_seed != 0:
                        token_flat_start = (
                            (b * Hkv + hkv) * Lkv + k_pos
                        ) * (n_groups * group_size)
                        v_grouped = v_vals.reshape(n_groups, group_size)
                        v_grouped = _inverse_wht_signs_token(
                            v_grouped, use_wht, sign_seed, token_flat_start,
                            layer_id=layer_id, stream_id=stream_id,
                        )
                        v_vals = v_grouped.reshape(D)

                    w = weights[b, hq, q_pos, k_pos]
                    sv += w * v_vals

                output[b, hq, q_pos, :] = sv

    return output
