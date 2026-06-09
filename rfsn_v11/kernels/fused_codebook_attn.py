"""
Fused codebook attention Metal kernels for RFSN v11.

Lloyd-Max codebook variant of the fused sparse attention kernel.

Instead of affine dequantization (scale + bias per group), this kernel
dequantizes each key/value vector by:
  1. Extracting 4-bit centroid indices from a packed uint32 stream.
  2. Looking up the corresponding float16 centroid vector from a codebook
     loaded cooperatively into threadgroup memory at kernel launch.

The codebook has shape ``(2**bits, D)`` where ``bits=4`` and ``D`` is the
head dimension (128 or 256).  Only bits=4 is supported in this kernel; for
other bit-widths use ``fused_sparse_attn.py`` with affine dequant.

TWO Metal kernel variants are shipped (same approach as fused_sparse_attn.py):
  ``_fused_cb_attn_d128``: ``tg_acc[32 * 128]``  — for D = 128
  ``_fused_cb_attn_d256``: ``tg_acc[32 * 256]``  — for D = 256

Python dispatch function selects the variant from the actual head dimension D.
``KernelDimError`` (from ``rfsn_v11.errors``) is raised hard-assert style if D
is not in (128, 256).

Grid: ``(n_q_heads * 1024,)`` threads — threadgroup 1024 = 32 simdgroups × 32
lanes, identical to the affine kernel.
"""

from __future__ import annotations

import mlx.core as mx

from ..compat import ensure_mlx_available
from ..errors import KernelDimError

# ---------------------------------------------------------------------------
# Shared Metal header — 4-bit index extraction from packed uint32 stream.
# No word-boundary straddling needed for 4-bit (two indices per byte, eight
# per 32-bit word), but we reuse the generic extract_bits helper for clarity.
# ---------------------------------------------------------------------------

_CB_HEADER = """
#include <metal_simdgroup>
using namespace metal;

// Extract bits_n-bit integer at position dim from a packed uint32 stream.
// For 4-bit indices: each uint32 holds 8 consecutive indices.
inline uint extract_bits(const device uint32_t* data, uint dim, uint bits_n) {
    uint bit_start = dim * bits_n;
    uint word      = bit_start >> 5;
    uint bit_off   = bit_start & 31u;

    if (bit_off + bits_n <= 32u) {
        return (data[word] >> bit_off) & ((1u << bits_n) - 1u);
    }
    // Straddles word boundary (cannot happen for 4-bit but kept for safety)
    uint bits_lo = 32u - bit_off;
    return ((data[word] >> bit_off) | (data[word + 1] << bits_lo))
           & ((1u << bits_n) - 1u);
}
"""

# ---------------------------------------------------------------------------
# Metal kernel body — parameterised by TG_D (threadgroup accumulator capacity)
# and CB_SIZE (codebook entries = 2**bits, always 16 for bits=4).
# ---------------------------------------------------------------------------

_BITS: int = 4
_CB_SIZE: int = 1 << _BITS  # 16


def _make_cb_attn_source(tg_d: int) -> str:
    """Generate Metal kernel source for a given threadgroup accumulator size."""
    # CB_SIZE is always 16 (bits=4).  TG_D is 128 or 256.
    cb_size = _CB_SIZE  # 16
    cb_total = cb_size * tg_d  # floats needed to cache the full codebook in tg mem

    return f"""
    // -----------------------------------------------------------------------
    // Thread / simdgroup ids
    // -----------------------------------------------------------------------
    uint head     = threadgroup_position_in_grid.x;
    uint tid      = thread_position_in_threadgroup.x;   // 0..1023
    uint simd_id  = tid >> 5;                           // 0..31
    uint lane_id  = tid & 31u;                          // 0..31

    // -----------------------------------------------------------------------
    // Shape metadata
    // -----------------------------------------------------------------------
    uint n_q_heads  = q_shape[0];
    uint D          = q_shape[2];                       // head dim (128 or 256)
    uint n_kv_heads = k_indices_shape[0];
    uint T_kv       = k_indices_shape[1];
    uint PACKED_COLS = k_indices_shape[2];              // packed uint32 columns

    uint n_repeats  = n_q_heads / n_kv_heads;
    uint kv_head    = head / n_repeats;

    uint num_blocks      = block_mask_shape[1];
    uint block_size_val  = (T_kv + num_blocks - 1u) / num_blocks;

    if (head >= n_q_heads) return;

    // -----------------------------------------------------------------------
    // Load codebook into threadgroup memory.
    // Codebook shape: (CB_SIZE={cb_size}, D) float16 = {cb_total} float16 values.
    // We cache it as float32 in threadgroup memory to avoid repeated half→float
    // conversions in the inner loop.
    // Each of the 1024 threads cooperatively loads ceil({cb_total}/1024) floats.
    // -----------------------------------------------------------------------
    threadgroup float tg_codebook[{cb_total}];  // {cb_size} centroids × D floats

    uint cb_total_elems = {cb_size}u * D;
    for (uint idx = tid; idx < cb_total_elems; idx += 1024u) {{
        tg_codebook[idx] = (float)codebook[idx];
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Load the 4 query elements this lane owns.
    // q shape: (n_q_heads, 1, D)  — the seq-len dimension (1) is squeezed.
    // -----------------------------------------------------------------------
    float q_local[4];
    uint  q_off = head * D;
    uint  d0    = lane_id * 4u;                        // first dim index for this lane
    for (uint i = 0u; i < 4u; i++) {{
        q_local[i] = (float)q[q_off + d0 + i];
    }}

    // -----------------------------------------------------------------------
    // Per-lane online softmax accumulators
    // -----------------------------------------------------------------------
    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_acc[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};

    uint kd_base = kv_head * T_kv * PACKED_COLS;
    uint bm_base = kv_head * num_blocks;

    // -----------------------------------------------------------------------
    // Inner loop: each simdgroup processes a disjoint subset of KV tokens
    // -----------------------------------------------------------------------
    for (uint k = simd_id; k < T_kv; k += 32u) {{

        // Block-sparse gate — skip inactive blocks.
        uint blk = (num_blocks > 0u) ? (k / block_size_val) : 0u;
        if (block_mask[bm_base + blk] == 0u) continue;

        // ----------------------------------------------------------------
        // Dequantize key vector for token k.
        // k_indices row:  k_indices[kv_head, k, :]  (packed uint32)
        // Each 4-bit index selects a row from tg_codebook.
        // Lane d0..d0+3 extracts 4 consecutive indices → 4 floats.
        // ----------------------------------------------------------------
        const device uint32_t* k_ptr = k_indices + kd_base + k * PACKED_COLS;

        float dot_partial = 0.0f;
        for (uint i = 0u; i < 4u; i++) {{
            uint idx = extract_bits(k_ptr, d0 + i, 4u);
            float k_val = tg_codebook[idx * D + d0 + i];
            dot_partial += q_local[i] * k_val;
        }}
        float score = simd_sum(dot_partial);

        // ----------------------------------------------------------------
        // Online softmax update
        // ----------------------------------------------------------------
        float new_max = max(local_max, score);
        float scale   = metal::fast::exp(local_max - new_max);
        float exp_s   = metal::fast::exp(score - new_max);
        local_max  = new_max;
        local_sum  = local_sum * scale + exp_s;

        // ----------------------------------------------------------------
        // Dequantize value vector for token k and accumulate.
        // Value indices are stored in the same k_indices tensor
        // (keys and values share the codebook quantisation), packed in
        // the second half of the packed_cols dimension.
        // v_indices row offset = k_indices row offset + PACKED_COLS/2 words.
        // ----------------------------------------------------------------
        uint v_col_offset = PACKED_COLS / 2u;
        const device uint32_t* v_ptr = k_ptr + v_col_offset;

        for (uint i = 0u; i < 4u; i++) {{
            uint idx   = extract_bits(v_ptr, d0 + i, 4u);
            float v_val = tg_codebook[idx * D + d0 + i];
            local_acc[i] = local_acc[i] * scale + exp_s * v_val;
        }}
    }}

    // -----------------------------------------------------------------------
    // Cross-simdgroup reduction.
    // Fixed-size threadgroup buffer for TG_D = {tg_d}.
    // -----------------------------------------------------------------------
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    threadgroup float tg_acc[32 * {tg_d}];

    if (lane_id == 0u) {{
        tg_max[simd_id] = local_max;
        tg_sum[simd_id] = local_sum;
    }}
    for (uint i = 0u; i < 4u; i++) {{
        tg_acc[simd_id * D + d0 + i] = local_acc[i];
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // -----------------------------------------------------------------------
    // Final reduction and output scatter — only lane 0 of simd 0 writes.
    // (All lanes in all simdgroups participate to avoid divergence in the
    // loop, but only lane_id == 0 writes the final output element it owns.)
    // -----------------------------------------------------------------------
    float global_max = -1e10f;
    for (uint s = 0u; s < 32u; s++) {{
        global_max = max(global_max, tg_max[s]);
    }}

    float global_sum = 0.0f;
    float result[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for (uint s = 0u; s < 32u; s++) {{
        float f = metal::fast::exp(tg_max[s] - global_max);
        global_sum += tg_sum[s] * f;
        for (uint i = 0u; i < 4u; i++) {{
            result[i] += tg_acc[s * D + d0 + i] * f;
        }}
    }}

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint i = 0u; i < 4u; i++) {{
        output[head * D + d0 + i] = (half)(result[i] * inv);
    }}
"""


# ---------------------------------------------------------------------------
# Kernel input / output metadata
# ---------------------------------------------------------------------------

_CB_INPUT_NAMES: list[str] = [
    "q",           # float16 (n_q_heads, 1, D)
    "k_indices",   # uint32  (n_kv_heads, T_kv, packed_cols) — keys+values packed
    "codebook",    # float16 (2**bits, D)
    "block_mask",  # uint8   (n_kv_heads, num_blocks) — 1=active, 0=skip
]
_CB_OUTPUT_NAMES: list[str] = ["output"]

_SUPPORTED_HEAD_DIMS: tuple[int, ...] = (128, 256)


# ---------------------------------------------------------------------------
# Lazy kernel builder / cache
# ---------------------------------------------------------------------------

def _get_or_build_cb_kernel(D: int):
    """Return (and lazily compile) the codebook attention kernel for head dim D.

    Raises:
        KernelDimError: if D is not in (128, 256).
        RuntimeError:   if ``mx.fast.metal_kernel`` is not available.
    """
    if D not in _SUPPORTED_HEAD_DIMS:
        raise KernelDimError(
            f"fused_codebook_attn: unsupported head_dim D={D}. "
            f"Supported values: {_SUPPORTED_HEAD_DIMS}. "
            "To add a new head dimension, ship a new kernel variant with "
            f"tg_acc[32 * D] and tg_codebook[{_CB_SIZE} * D]."
        )

    ensure_mlx_available()
    if not hasattr(mx.fast, "metal_kernel"):
        raise RuntimeError(
            "mx.fast.metal_kernel is not available on this platform or MLX version. "
            "Metal kernels require Apple Silicon with MLX >= 0.5."
        )

    cache_attr = f"_rfsn_v11_cb_attn_d{D}"
    kernel = getattr(_get_or_build_cb_kernel, cache_attr, None)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"rfsn_v11_cb_attn_d{D}",
            input_names=_CB_INPUT_NAMES,
            output_names=_CB_OUTPUT_NAMES,
            source=_make_cb_attn_source(D),
            header=_CB_HEADER,
        )
        setattr(_get_or_build_cb_kernel, cache_attr, kernel)
    return kernel


# ---------------------------------------------------------------------------
# Public dispatch function
# ---------------------------------------------------------------------------

def fused_codebook_attn(
    q: mx.array,
    k_indices: mx.array,
    codebook: mx.array,
    block_mask: mx.array | None = None,
    bits: int = 4,
) -> mx.array:
    """Fused Lloyd-Max codebook attention: index lookup + online softmax.

    Dequantizes key and value vectors by looking up 4-bit centroid indices in a
    shared codebook (loaded into threadgroup memory once per dispatch), then
    computes scaled dot-product attention with an online softmax accumulator.
    An optional block-sparse gate (``block_mask``) allows entire KV blocks to
    be skipped, enabling sparse decode paths.

    Args:
        q:          float16, shape ``(n_q_heads, 1, D)`` — queries.
        k_indices:  uint32,  shape ``(n_kv_heads, T_kv, packed_cols)`` —
                    packed 4-bit centroid indices.  The first half of
                    ``packed_cols`` encodes key indices; the second half
                    encodes value indices.  ``packed_cols`` must therefore be
                    even, with each half holding ``D * bits / 32`` uint32 words.
        codebook:   float16, shape ``(2**bits, D)`` — Lloyd-Max centroids.
                    Shared across keys and values.
        block_mask: uint8,   shape ``(n_kv_heads, num_blocks)`` — ``1`` means
                    the block is active, ``0`` means skip.  Pass ``None`` for
                    dense (no blocks skipped).
        bits:       Quantisation bits.  **Must be 4**; other values are not
                    supported by this kernel.

    Returns:
        output: float16, shape ``(n_q_heads, D)`` — attention output.

    Raises:
        KernelDimError: if the head dimension ``D`` inferred from ``q`` is not
            128 or 256.
        ValueError:     if ``bits != 4``.
    """
    if bits != _BITS:
        raise ValueError(
            f"fused_codebook_attn only supports bits=4; got bits={bits}. "
            "Use fused_sparse_attn for affine dequantization with other bit-widths."
        )

    # q shape: (n_q_heads, 1, D)
    n_q_heads: int = int(q.shape[0])
    D: int = int(q.shape[2])

    # Hard-assert D before attempting kernel compilation.
    if D not in _SUPPORTED_HEAD_DIMS:
        raise KernelDimError(
            f"fused_codebook_attn: unsupported head_dim D={D}. "
            f"q.shape={q.shape}. Supported head dims: {_SUPPORTED_HEAD_DIMS}."
        )

    n_kv_heads: int = int(k_indices.shape[0])
    T_kv: int = int(k_indices.shape[1])

    # Build a dense all-ones block_mask when none is provided so the Metal
    # kernel always has a valid pointer; num_blocks = T_kv means each token
    # is its own block and none are skipped.
    if block_mask is None:
        block_mask = mx.ones((n_kv_heads, T_kv), dtype=mx.uint8)

    kernel = _get_or_build_cb_kernel(D)

    outputs = kernel(
        inputs=[q, k_indices, codebook, block_mask],
        grid=(n_q_heads * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_q_heads, D)],
        output_dtypes=[mx.float16],
    )
    return outputs[0]
