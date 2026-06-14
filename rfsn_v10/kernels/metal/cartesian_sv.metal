// Cartesian SV kernel for grouped symmetric quantization.
// One thread computes one output coordinate (b, hq, q_pos, d).
//
// FIXME: This kernel computes a weighted sum of V_packed values, where
// V_packed is in the transform domain (WHT + hash signs).  The accumulated
// result must be inverse-transformed before writing to output.  See
// cartesian_sv_body.metal for details.
//
// Generated signature:
//   device const float*    weights       [[buffer(0)]]
//   device const uint32_t* packed_codes   [[buffer(1)]]
//   device const float*    scales         [[buffer(2)]]   // (B, Hkv, Lkv, n_groups)
//   device const int*      bits_buf      [[buffer(3)]]
//   device const int*      group_buf     [[buffer(4)]]
//   device const int*      b_buf         [[buffer(5)]]
//   device const int*      hq_buf        [[buffer(6)]]
//   device const int*      hkv_buf       [[buffer(7)]]
//   device const int*      lq_buf        [[buffer(8)]]
//   device const int*      lkv_buf       [[buffer(9)]]
//   device const int*      d_buf         [[buffer(10)]]
//   device float*          output         [[buffer(11)]]

#include <metal_stdlib>
using namespace metal;

uint3 thread_position_in_grid [[thread_position_in_grid]];

kernel void cartesian_sv(
    device const float*   weights,
    device const uint32_t* packed_codes,
    device const float*   scales,
    device const int*     bits_buf,
    device const int*     group_buf,
    device const int*     b_buf,
    device const int*     hq_buf,
    device const int*     hkv_buf,
    device const int*     lq_buf,
    device const int*     lkv_buf,
    device const int*     d_buf,
    device float*         output
)
{
    int bits = bits_buf[0];
    int group_size = group_buf[0];
    int B = b_buf[0];
    int Hq = hq_buf[0];
    int Hkv = hkv_buf[0];
    int Lq = lq_buf[0];
    int Lkv = lkv_buf[0];
    int D = d_buf[0];

    int d = thread_position_in_grid.x;
    int q_pos = thread_position_in_grid.y;
    int hq = (thread_position_in_grid.z % Hq);
    int b = (thread_position_in_grid.z / Hq);

    if (b >= B || hq >= Hq || q_pos >= Lq || d >= D) return;

    int hkv = hq * Hkv / Hq;

    int codes_per_word = 32 / bits;
    int words_per_vec = (D + codes_per_word - 1) / codes_per_word;
    int mask = (1 << bits) - 1;
    int qmax = (1 << (bits - 1)) - 1;
    int n_groups = (D + group_size - 1) / group_size;

    float result = 0.0f;

    for (int k_pos = 0; k_pos < Lkv; ++k_pos) {
        int kv_offset = ((b * Hkv + hkv) * Lkv + k_pos);

        int word_idx = d / codes_per_word;
        int bit_offset = (d % codes_per_word) * bits;

        int packed_idx = kv_offset * words_per_vec + word_idx;
        uint32_t word = packed_codes[packed_idx];
        int code = int((word >> bit_offset) & mask);

        int group_idx = d / group_size;
        // Per-token scale indexing: (b, hkv, k_pos, group)
        float scale = scales[(((b * Hkv + hkv) * Lkv + k_pos) * n_groups) + group_idx];
        float v_val = (float(code) - float(qmax)) * scale;

        int w_idx = ((b * Hq + hq) * Lq + q_pos) * Lkv + k_pos;
        float w = weights[w_idx];

        result += w * v_val;
    }

    int out_idx = ((b * Hq + hq) * Lq + q_pos) * D + d;
    output[out_idx] = result;
}
