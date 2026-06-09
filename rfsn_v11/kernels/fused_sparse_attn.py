"""
Fused sparse attention Metal kernels for RFSN v11.

Ported from turboquant-mlx-main/turboquant/fused_v2_attn.py.

TWO BUGS FIXED vs the upstream source:
  1. D=128 hardcoded in tg_acc allocation:
       BUG:  threadgroup float tg_acc[32 * 128]  ← corrupts memory for D > 128
       FIX:  Ship two kernel variants:
               _fused_attn_d128: tg_acc[32 * 128]  for D ≤ 128
               _fused_attn_d256: tg_acc[32 * 256]  for D ≤ 256
             The Python dispatch function selects the variant from actual D.
             Hard assert D in (128, 256) before any kernel call.

  2. Block-sparse gate: optional block_mask input (n_kv_heads, num_blocks) uint8.
     When block_mask[kv_head * num_blocks + k / block_size] == 0 the key/value
     is skipped in the inner loop. Enables block-sparse decode paths.

Grid: (n_q_heads * 1024) threads, threadgroup 1024 = 32 simdgroups × 32 lanes.
"""

from __future__ import annotations

import mlx.core as mx

from ..compat import ensure_mlx_available

_FUSED_V2_HEADER = """
#include <metal_simdgroup>
using namespace metal;

// Extract B-bit integer from packed uint32 stream.
// Handles word-boundary straddling for 3-bit.
inline uint extract_bits(const device uint32_t* data, uint dim, uint bits) {
    uint bit_start = dim * bits;
    uint word = bit_start >> 5;
    uint bit_offset = bit_start & 31;

    if (bit_offset + bits <= 32) {
        return (data[word] >> bit_offset) & ((1u << bits) - 1u);
    }
    // Straddles word boundary
    uint bits_lo = 32 - bit_offset;
    return ((data[word] >> bit_offset) | (data[word + 1] << bits_lo)) & ((1u << bits) - 1u);
}
"""

# ---------------------------------------------------------------------------
# Kernel body generators — one per supported D
# ---------------------------------------------------------------------------

def _make_attn_source(tg_d: int) -> str:
    """Dispatch to the correct kernel source generator for the given D.

    tg_d == 128  →  _make_attn_source_d128()  (4 registers per lane)
    tg_d == 256  →  _make_attn_source_d256()  (8 registers per lane, half tg_acc)
    """
    if tg_d <= 128:
        return _make_attn_source_d128()
    return _make_attn_source_d256()


def _make_attn_source_d128() -> str:
    """Metal kernel body for D=128.

    Each lane covers exactly 4 elements at positions (lane_id*4)..(lane_id*4+3).
    threadgroup memory: tg_max[32] + tg_sum[32] + tg_acc[32*128] = 16 640 bytes.
    """
    return """
    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_id = tid >> 5;
    uint lane_id = tid & 31;

    uint n_q_heads = q_rot_shape[0];
    uint D = q_rot_shape[1];
    uint n_kv_heads = key_scales_shape[0];
    uint T_kv = key_scales_shape[1];
    uint N_GROUPS = key_scales_shape[2];
    uint PACKED_DIM = key_data_shape[2];
    uint SIGN_WORDS = sign_bits_shape[2];
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;
    uint group_size = D / N_GROUPS;

    uint num_blocks = block_mask_shape[1];
    uint block_size_val = (T_kv + num_blocks - 1u) / num_blocks;

    if (head >= n_q_heads) return;

    // Load only the MSE query into registers; read q_sketch inline to save 4 registers.
    float q[4];
    uint q_off = head * D;
    uint d0 = lane_id * 4;
    for (uint i = 0; i < 4; i++)
        q[i] = q_rot[q_off + d0 + i];

    uint our_group     = d0 / group_size;
    uint sign_word     = d0 >> 5;
    uint sign_bit_base = d0 & 31;

    uint kv_data_stride  = T_kv * PACKED_DIM;
    uint kv_scale_stride = T_kv * N_GROUPS;
    uint kv_sign_stride  = T_kv * SIGN_WORDS;
    uint kv_norm_stride  = T_kv;

    uint kd_base = kv_head * kv_data_stride;
    uint ks_base = kv_head * kv_scale_stride;
    uint sb_base = kv_head * kv_sign_stride;
    uint rn_base = kv_head * kv_norm_stride;
    uint bm_base = kv_head * num_blocks;

    float local_max = -1e10f;
    float local_sum = 0.0f;
    float local_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint bits = (uint)bits_arr[0];
    float qjl_s = qjl_scale[0];   // cache scalar, avoid repeated device reads

    for (uint k = simd_id; k < T_kv; k += 32) {
        uint blk = (num_blocks > 0u) ? (k / block_size_val) : 0u;
        if (block_mask[bm_base + blk] == 0) continue;

        const device uint32_t* k_data_ptr = key_data + kd_base + k * PACKED_DIM;
        float k_scale = key_scales[ks_base + k * N_GROUPS + our_group];
        float k_bias  = key_biases[ks_base + k * N_GROUPS + our_group];

        // MSE and QJL partials in one pass over the 4 elements.
        float mse_partial = 0.0f;
        float qjl_partial = 0.0f;
        uint32_t signs = sign_bits[sb_base + k * SIGN_WORDS + sign_word];
        for (uint i = 0; i < 4; i++) {
            uint quant_int = extract_bits(k_data_ptr, d0 + i, bits);
            float k_val = (float)quant_int * k_scale + k_bias;
            mse_partial += q[i] * k_val;
            float sign_val = ((signs >> (sign_bit_base + i)) & 1u) ? 1.0f : -1.0f;
            qjl_partial += q_sketch[q_off + d0 + i] * sign_val;
        }
        float mse_score = simd_sum(mse_partial);
        float qjl_score = simd_sum(qjl_partial) * qjl_s * residual_norms[rn_base + k];

        float score = mse_score + qjl_score;
        float new_max = max(local_max, score);
        float factor  = exp(local_max - new_max);
        float exp_s   = exp(score - new_max);
        local_max = new_max;
        local_sum = local_sum * factor + exp_s;

        const device uint32_t* v_data_ptr = value_data + kd_base + k * PACKED_DIM;
        float v_scale = value_scales[ks_base + k * N_GROUPS + our_group];
        float v_bias  = value_biases[ks_base + k * N_GROUPS + our_group];

        for (uint i = 0; i < 4; i++) {
            uint v_int = extract_bits(v_data_ptr, d0 + i, bits);
            float v_val = (float)v_int * v_scale + v_bias;
            local_acc[i] = local_acc[i] * factor + exp_s * v_val;
        }
    }

    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    threadgroup float tg_acc[32 * 128];

    if (lane_id == 0) {
        tg_max[simd_id] = local_max;
        tg_sum[simd_id] = local_sum;
    }
    for (uint i = 0; i < 4; i++)
        tg_acc[simd_id * D + d0 + i] = local_acc[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e10f;
    for (uint s = 0; s < 32; s++)
        global_max = max(global_max, tg_max[s]);

    float global_sum = 0.0f;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint s = 0; s < 32; s++) {
        float rfactor = exp(tg_max[s] - global_max);
        global_sum += tg_sum[s] * rfactor;
        for (uint i = 0; i < 4; i++)
            result[i] += tg_acc[s * D + d0 + i] * rfactor;
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint i = 0; i < 4; i++)
        output_rot[head * D + d0 + i] = result[i] * inv;
"""


def _make_attn_source_d256() -> str:
    """Metal kernel body for D=256.

    Each lane covers 8 elements across two 4-element windows separated by 128:
      window 0: positions (lane_id*4)..(lane_id*4+3)
      window 1: positions (128+lane_id*4)..(128+lane_id*4+3)

    Uses separate named registers (q0/q1, acc0/acc1) rather than indexed arrays
    to avoid Metal compiler miscompilation of non-constant array indices.

    Uses float16 (half) threadgroup accumulator to stay within the 32 KB limit:
      tg_max[32]          =   128 bytes  (float32)
      tg_sum[32]          =   128 bytes  (float32)
      tg_acc[32*256 half] = 16 384 bytes (float16)
      Total               = 16 640 bytes  ✓
    """
    return """
    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint simd_id = tid >> 5;
    uint lane_id = tid & 31;

    uint n_q_heads = q_rot_shape[0];
    uint D = q_rot_shape[1];
    uint n_kv_heads = key_scales_shape[0];
    uint T_kv = key_scales_shape[1];
    uint N_GROUPS = key_scales_shape[2];
    uint PACKED_DIM = key_data_shape[2];
    uint SIGN_WORDS = sign_bits_shape[2];
    uint n_repeats = n_q_heads / n_kv_heads;
    uint kv_head = head / n_repeats;
    uint group_size = D / N_GROUPS;

    uint num_blocks = block_mask_shape[1];
    uint block_size_val = (T_kv + num_blocks - 1u) / num_blocks;

    if (head >= n_q_heads) return;

    // Load MSE query registers only; read q_sketch inline to save 8 registers.
    uint q_off = head * D;
    uint d0 = lane_id * 4;
    uint d1 = 128u + lane_id * 4;

    float q0[4], q1[4];
    for (uint i = 0; i < 4; i++) {
        q0[i] = q_rot[q_off + d0 + i];
        q1[i] = q_rot[q_off + d1 + i];
    }

    uint our_group0    = d0 / group_size;
    uint sign_word0    = d0 >> 5;
    uint sign_bit0     = d0 & 31;
    uint our_group1    = d1 / group_size;
    uint sign_word1    = d1 >> 5;
    uint sign_bit1     = d1 & 31;

    uint kv_data_stride  = T_kv * PACKED_DIM;
    uint kv_scale_stride = T_kv * N_GROUPS;
    uint kv_sign_stride  = T_kv * SIGN_WORDS;
    uint kv_norm_stride  = T_kv;

    uint kd_base = kv_head * kv_data_stride;
    uint ks_base = kv_head * kv_scale_stride;
    uint sb_base = kv_head * kv_sign_stride;
    uint rn_base = kv_head * kv_norm_stride;
    uint bm_base = kv_head * num_blocks;

    float local_max = -1e10f;
    float local_sum = 0.0f;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    uint bits = (uint)bits_arr[0];
    float qjl_s = qjl_scale[0];   // cache scalar

    for (uint k = simd_id; k < T_kv; k += 32) {
        uint blk = (num_blocks > 0u) ? (k / block_size_val) : 0u;
        if (block_mask[bm_base + blk] == 0) continue;

        const device uint32_t* k_data_ptr = key_data + kd_base + k * PACKED_DIM;

        // Window 0 — MSE and QJL in one pass.
        float k_scale0  = key_scales[ks_base + k * N_GROUPS + our_group0];
        float k_bias0   = key_biases[ks_base + k * N_GROUPS + our_group0];
        uint32_t signs0 = sign_bits[sb_base + k * SIGN_WORDS + sign_word0];
        float mse_partial = 0.0f, qjl_partial = 0.0f;
        for (uint i = 0; i < 4; i++) {
            uint qk0 = extract_bits(k_data_ptr, d0 + i, bits);
            mse_partial += q0[i] * ((float)qk0 * k_scale0 + k_bias0);
            float sv0 = ((signs0 >> (sign_bit0 + i)) & 1u) ? 1.0f : -1.0f;
            qjl_partial += q_sketch[q_off + d0 + i] * sv0;
        }

        // Window 1.
        float k_scale1  = key_scales[ks_base + k * N_GROUPS + our_group1];
        float k_bias1   = key_biases[ks_base + k * N_GROUPS + our_group1];
        uint32_t signs1 = sign_bits[sb_base + k * SIGN_WORDS + sign_word1];
        for (uint i = 0; i < 4; i++) {
            uint qk1 = extract_bits(k_data_ptr, d1 + i, bits);
            mse_partial += q1[i] * ((float)qk1 * k_scale1 + k_bias1);
            float sv1 = ((signs1 >> (sign_bit1 + i)) & 1u) ? 1.0f : -1.0f;
            qjl_partial += q_sketch[q_off + d1 + i] * sv1;
        }

        float mse_score = simd_sum(mse_partial);
        float qjl_score = simd_sum(qjl_partial) * qjl_s * residual_norms[rn_base + k];
        float score     = mse_score + qjl_score;

        float new_max = max(local_max, score);
        float factor  = exp(local_max - new_max);
        float exp_s   = exp(score - new_max);
        local_max = new_max;
        local_sum = local_sum * factor + exp_s;

        const device uint32_t* v_data_ptr = value_data + kd_base + k * PACKED_DIM;

        // Value accumulation window 0
        float v_scale0 = value_scales[ks_base + k * N_GROUPS + our_group0];
        float v_bias0  = value_biases[ks_base + k * N_GROUPS + our_group0];
        for (uint i = 0; i < 4; i++) {
            uint vi0 = extract_bits(v_data_ptr, d0 + i, bits);
            acc0[i] = acc0[i] * factor + exp_s * ((float)vi0 * v_scale0 + v_bias0);
        }

        // Value accumulation window 1
        float v_scale1 = value_scales[ks_base + k * N_GROUPS + our_group1];
        float v_bias1  = value_biases[ks_base + k * N_GROUPS + our_group1];
        for (uint i = 0; i < 4; i++) {
            uint vi1 = extract_bits(v_data_ptr, d1 + i, bits);
            acc1[i] = acc1[i] * factor + exp_s * ((float)vi1 * v_scale1 + v_bias1);
        }
    }

    // Cross-simdgroup reduction — half tg_acc to stay within 32 KB.
    threadgroup float tg_max[32];
    threadgroup float tg_sum[32];
    threadgroup half  tg_acc[32 * 256];

    if (lane_id == 0) {
        tg_max[simd_id] = local_max;
        tg_sum[simd_id] = local_sum;
    }
    for (uint i = 0; i < 4; i++) {
        tg_acc[simd_id * D + d0 + i] = (half)acc0[i];
        tg_acc[simd_id * D + d1 + i] = (half)acc1[i];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float global_max = -1e10f;
    for (uint s = 0; s < 32; s++)
        global_max = max(global_max, tg_max[s]);

    float global_sum = 0.0f;
    float result0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float result1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint s = 0; s < 32; s++) {
        float rfactor = exp(tg_max[s] - global_max);
        global_sum += tg_sum[s] * rfactor;
        for (uint i = 0; i < 4; i++) {
            result0[i] += (float)tg_acc[s * D + d0 + i] * rfactor;
            result1[i] += (float)tg_acc[s * D + d1 + i] * rfactor;
        }
    }

    float inv = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
    for (uint i = 0; i < 4; i++) {
        output_rot[head * D + d0 + i] = result0[i] * inv;
        output_rot[head * D + d1 + i] = result1[i] * inv;
    }
"""


# Common input/output names for both variants
_INPUT_NAMES = [
    "q_rot", "q_sketch",
    "key_data", "key_scales", "key_biases",
    "sign_bits", "residual_norms", "qjl_scale",
    "value_data", "value_scales", "value_biases",
    "bits_arr",
    "block_mask",  # (n_kv_heads, num_blocks) uint8 — 1=active, 0=skip
]
_OUTPUT_NAMES = ["output_rot"]

_SUPPORTED_HEAD_DIMS = (128, 256)


def _get_or_build_kernel(D: int):
    """Return (and lazily build) the kernel for head dim D.

    Raises ValueError if D is not in _SUPPORTED_HEAD_DIMS.
    """
    if D not in _SUPPORTED_HEAD_DIMS:
        raise ValueError(
            f"Unsupported head_dim D={D}. Supported: {_SUPPORTED_HEAD_DIMS}. "
            "Add a new variant with tg_acc[32 * D] to support additional head dims."
        )
    ensure_mlx_available()
    if not hasattr(mx.fast, "metal_kernel"):
        raise RuntimeError("mx.fast.metal_kernel not available on this platform/MLX version.")

    cache_attr = f"_rfsn_v11_fused_attn_d{D}"
    kernel = getattr(_get_or_build_kernel, cache_attr, None)
    if kernel is None:
        kernel = mx.fast.metal_kernel(
            name=f"rfsn_v11_fused_attn_d{D}",
            input_names=_INPUT_NAMES,
            output_names=_OUTPUT_NAMES,
            source=_make_attn_source(D),
            header=_FUSED_V2_HEADER,
        )
        setattr(_get_or_build_kernel, cache_attr, kernel)
    return kernel


def fused_sparse_attention(
    q_rot: mx.array,
    q_sketch: mx.array,
    key_data: mx.array,
    key_scales: mx.array,
    key_biases: mx.array,
    sign_bits: mx.array,
    residual_norms: mx.array,
    value_data: mx.array,
    value_scales: mx.array,
    value_biases: mx.array,
    bits: int,
    qjl_scale: float,
    n_q_heads: int,
    D: int,
    block_mask: mx.array | None = None,
) -> mx.array:
    """Fused attention: affine dequant + QJL + online softmax + block-sparse gate.

    Args:
        q_rot:          (n_q_heads, D) float32 — scaled, rotated queries
        q_sketch:       (n_q_heads, D) float32 — JL sketch of queries
        key_data:       (n_kv_heads, T_kv, D*bits/32) uint32
        key_scales:     (n_kv_heads, T_kv, D/group_size) float32
        key_biases:     (n_kv_heads, T_kv, D/group_size) float32
        sign_bits:      (n_kv_heads, T_kv, D/32) uint32
        residual_norms: (n_kv_heads, T_kv) float32
        value_data:     (n_kv_heads, T_kv, D*bits/32) uint32
        value_scales:   (n_kv_heads, T_kv, D/group_size) float32
        value_biases:   (n_kv_heads, T_kv, D/group_size) float32
        bits:           Quantization bits (3 or 4)
        qjl_scale:      QJL correction scale = sqrt(pi/2) / D
        n_q_heads:      Total query heads
        D:              Head dimension — must be 128 or 256
        block_mask:     (n_kv_heads, num_blocks) uint8 — 1=active, 0=skip.
                        Pass None or all-ones for dense (no blocks skipped).

    Returns:
        output_rot: (n_q_heads, D) float32 — in rotated key space
    """
    assert D in _SUPPORTED_HEAD_DIMS, (
        f"Unsupported head_dim D={D}. Must be one of {_SUPPORTED_HEAD_DIMS}."
    )

    n_kv_heads = key_data.shape[0]
    T_kv = key_data.shape[1]
    num_blocks = T_kv  # default: one "block" per token (all active)
    if block_mask is None:
        block_mask = mx.ones((n_kv_heads, num_blocks), dtype=mx.uint8)
    else:
        num_blocks = block_mask.shape[1]

    kernel = _get_or_build_kernel(D)
    bits_arr = mx.array([bits], dtype=mx.int32)
    qjl_scale_arr = mx.array([qjl_scale], dtype=mx.float32)

    outputs = kernel(
        inputs=[
            q_rot, q_sketch,
            key_data, key_scales, key_biases,
            sign_bits, residual_norms, qjl_scale_arr,
            value_data, value_scales, value_biases,
            bits_arr,
            block_mask,
        ],
        grid=(n_q_heads * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(n_q_heads, D)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]
