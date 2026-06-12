// Scalar SV kernel for Polar fused attention.
// One thread computes one output element (b, query_head, q_pos, dimension).
//
// Inputs:
//   - attention_weights:  float32 (B, Hq, Lq, Lkv)
//   - packed_value_indices: uint32  (B, Hkv, Lkv, packed_dim)
//   - value_norms:        float32  (B, Hkv, Lkv)
//   - value_centroids:    float32  (n_centroids,)
//
// Output:
//   - output_rotated: float32 (B, Hq, Lq, D)

#include <metal_stdlib>
using namespace metal;

kernel void polar_sv_scalar(
    device const float*   attention_weights   [[buffer(0)]],
    device const uint32_t* packed_value_indices [[buffer(1)]],
    device const float*   value_norms          [[buffer(2)]],
    device const float*   value_centroids      [[buffer(3)]],
    device float*         output_rotated       [[buffer(4)]],
    constant int&         bits               [[buffer(5)]],
    constant int&         B                  [[buffer(6)]],
    constant int&         Hq                 [[buffer(7)]],
    constant int&         Hkv                [[buffer(8)]],
    constant int&         Lq                 [[buffer(9)]],
    constant int&         Lkv                [[buffer(10)]],
    constant int&         D                  [[buffer(11)]],
    constant int&         values_per_word    [[buffer(12)]],
    uint3                 tid                [[thread_position_in_grid]]
)
{
    int b = tid.z;
    int hq = tid.y;
    int d = tid.x;

    if (b >= B || hq >= Hq || d >= D) return;

    int hkv = hq * Hkv / Hq;

    int mask = (1 << bits) - 1;
    int words_per_vec = (D + values_per_word - 1) / values_per_word;

    float accum = 0.0f;
    int w_offset = ((b * Hq + hq) * Lq) * Lkv;

    for (int k = 0; k < Lkv; ++k) {
        int word_idx = d / values_per_word;
        int bit_offset = (d % values_per_word) * bits;

        int v_idx = ((b * Hkv + hkv) * Lkv + k) * words_per_vec + word_idx;
        uint32_t word = packed_value_indices[v_idx];
        int code = int((word >> bit_offset) & mask);

        float centroid = value_centroids[code];
        float weight = attention_weights[w_offset + k];
        float norm = value_norms[((b * Hkv + hkv) * Lkv) + k];

        accum += weight * norm * centroid;
    }

    int out_idx = ((b * Hq + hq) * Lq) * D + d;
    output_rotated[out_idx] = accum;
}
