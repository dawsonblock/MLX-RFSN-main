// Scalar QK kernel for Polar fused attention.
// One thread computes one (query_head, key_position) score.
//
// Inputs:
//   - rotated_queries:  float/bfloat (B, Hq, Lq, D)
//   - packed_key_indices: uint32    (B, Hkv, Lkv, packed_dim)
//   - key_norms:        float32    (B, Hkv, Lkv)
//   - key_centroids:    float32    (n_centroids,)
//   - scale:            float32
//
// Output:
//   - scores: float32 (B, Hq, Lq, Lkv)

#include <metal_stdlib>
using namespace metal;

kernel void polar_qk_scalar(
    device const float*   rotated_queries    [[buffer(0)]],
    device const uint32_t* packed_key_indices [[buffer(1)]],
    device const float*   key_norms          [[buffer(2)]],
    device const float*   key_centroids      [[buffer(3)]],
    device float*         scores             [[buffer(4)]],
    constant int&         bits               [[buffer(5)]],
    constant float&       scale              [[buffer(6)]],
    constant int&         B                  [[buffer(7)]],
    constant int&         Hq                 [[buffer(8)]],
    constant int&         Hkv                [[buffer(9)]],
    constant int&         Lq                 [[buffer(10)]],
    constant int&         Lkv                [[buffer(11)]],
    constant int&         D                  [[buffer(12)]],
    constant int&         values_per_word    [[buffer(13)]],
    uint3                 tid                [[thread_position_in_grid]]
)
{
    int b = tid.z;
    int hq = tid.y;
    int k_pos = tid.x;

    if (b >= B || hq >= Hq || k_pos >= Lkv) return;

    // GQA: map query head to KV head
    int hkv = hq * Hkv / Hq;

    // Load query
    int q_offset = ((b * Hq + hq) * Lq) * D;
    int k_offset = ((b * Hkv + hkv) * Lkv + k_pos);

    float score = 0.0f;
    int mask = (1 << bits) - 1;

    // Packed words per key vector
    int words_per_vec = (D + values_per_word - 1) / values_per_word;

    for (int d = 0; d < D; ++d) {
        int word_idx = d / values_per_word;
        int bit_offset = (d % values_per_word) * bits;

        int packed_idx = k_offset * words_per_vec + word_idx;
        uint32_t word = packed_key_indices[packed_idx];
        int code = int((word >> bit_offset) & mask);

        float centroid = key_centroids[code];
        float q_val = rotated_queries[q_offset + d];
        score += q_val * centroid;
    }

    score *= key_norms[k_offset] * scale;

    int out_idx = ((b * Hq + hq) * Lq) * Lkv + k_pos;
    scores[out_idx] = score;
}
