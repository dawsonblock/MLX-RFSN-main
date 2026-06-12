// Cartesian QK kernel for grouped symmetric quantization.
// One thread computes one (b, hq, k_pos) score.
//
// Generated signature by mlx.core.fast.metal_kernel:
//   device const float*    queries       [[buffer(0)]]
//   device const uint32_t* packed_codes   [[buffer(1)]]
//   device const float*    scales         [[buffer(2)]]
//   device const int*      bits_buf      [[buffer(3)]]
//   device const int*      group_buf     [[buffer(4)]]
//   device const float*    scale_buf     [[buffer(5)]]
//   device const int*      b_buf         [[buffer(6)]]
//   device const int*      hq_buf        [[buffer(7)]]
//   device const int*      hkv_buf       [[buffer(8)]]
//   device const int*      lq_buf        [[buffer(9)]]
//   device const int*      lkv_buf       [[buffer(10)]]
//   device const int*      d_buf         [[buffer(11)]]
//   device float*          scores         [[buffer(12)]]

#include <metal_stdlib>
using namespace metal;

uint3 thread_position_in_grid [[thread_position_in_grid]];

kernel void cartesian_qk(
    device const float*   queries,
    device const uint32_t* packed_codes,
    device const float*   scales,
    device const int*     bits_buf,
    device const int*     group_buf,
    device const float*   scale_buf,
    device const int*     b_buf,
    device const int*     hq_buf,
    device const int*     hkv_buf,
    device const int*     lq_buf,
    device const int*     lkv_buf,
    device const int*     d_buf,
    device float*         scores
)
{
    int bits = bits_buf[0];
    int group_size = group_buf[0];
    float scale_factor = scale_buf[0];
    int B = b_buf[0];
    int Hq = hq_buf[0];
    int Hkv = hkv_buf[0];
    int Lq = lq_buf[0];
    int Lkv = lkv_buf[0];
    int D = d_buf[0];

    int b = thread_position_in_grid.z;
    int hq = thread_position_in_grid.y;
    int k_pos = thread_position_in_grid.x;

    if (b >= B || hq >= Hq || k_pos >= Lkv) return;

    int hkv = hq * Hkv / Hq;

    int q_offset = ((b * Hq + hq) * Lq) * D;
    int kv_offset = ((b * Hkv + hkv) * Lkv + k_pos);

    int codes_per_word = 32 / bits;
    int words_per_vec = (D + codes_per_word - 1) / codes_per_word;
    int mask = (1 << bits) - 1;
    int qmax = (1 << (bits - 1)) - 1;
    int n_groups = (D + group_size - 1) / group_size;

    float score = 0.0f;

    for (int d = 0; d < D; ++d) {
        int word_idx = d / codes_per_word;
        int bit_offset = (d % codes_per_word) * bits;

        int packed_idx = kv_offset * words_per_vec + word_idx;
        uint32_t word = packed_codes[packed_idx];
        int code = int((word >> bit_offset) & mask);

        int group_idx = d / group_size;
        float scale = scales[((b * Hkv + hkv) * n_groups) + group_idx];
        float k_val = (float(code) - float(qmax)) * scale;

        float q_val = queries[q_offset + d];
        score += q_val * k_val;
    }

    score *= scale_factor;

    int out_idx = ((b * Hq + hq) * Lq) * Lkv + k_pos;
    scores[out_idx] = score;
}
