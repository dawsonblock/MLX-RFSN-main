// TurboPolar online softmax attention with dense V
// Inputs per block:
//   q (float, D)
//   radii (half, block_size * D/2)
//   angle_codes (uchar, block_size * D/2)
//   values (half, block_size * D)   // dense V
//   prev_m (float, 1)   // previous max
//   prev_l (float, 1)   // previous sum-exp
//   prev_acc (float, D) // previous accumulator
// Output:
//   out_scores (float, block_size)  // for debugging
//   new_m (float, 1)
//   new_l (float, 1)
//   new_acc (float, D)
//
// One threadgroup processes one (batch, head, block).
// Threads cooperate to reconstruct K, compute scores, and update softmax.
//
// NOTE: This is the Phase 8 target. Initial implementation uses Python reference
// with per-block kernel calls; full fused version requires careful workgroup sync.

#include <metal_stdlib>
using namespace metal;

kernel void tqpolar_online_attention_dense_v(
    device const float* q,
    device const half* radii,
    device const uchar* angle_codes,
    device const half* values,
    device float* out_scores,
    device float* new_m,
    device float* new_l,
    device float* new_acc,
    constant int& head_dim,
    constant int& block_size,
    constant int& angle_bits,
    float prev_m,
    float prev_l,
    device const float* prev_acc,
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= uint(block_size)) { return; }

    const float TWO_PI = 6.28318530718f;
    int pairs = head_dim / 2;
    float bin_width = TWO_PI / float(1 << angle_bits);
    float half_bin = bin_width * 0.5f;

    // Reconstruct K for this token
    float k_vec[128]; // head_dim must be <= 128 for this static array
    for (int p = 0; p < pairs; ++p) {
        int idx = int(tid) * pairs + p;
        float radius = float(radii[idx]);
        uint code = uint(angle_codes[idx]);
        float angle = float(code) * bin_width + half_bin;
        k_vec[p * 2]     = radius * cos(angle);
        k_vec[p * 2 + 1] = radius * sin(angle);
    }

    // Compute QK score
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        score += q[d] * k_vec[d];
    }
    out_scores[int(tid)] = score;

    // Online softmax update (per-thread for now; workgroup reduction needed)
    float m_new = max(prev_m, score);
    float alpha = exp(prev_m - m_new);
    float p = exp(score - m_new);

    // Update accumulator
    for (int d = 0; d < head_dim; ++d) {
        new_acc[d] = prev_acc[d] * alpha + p * float(values[int(tid) * head_dim + d]);
    }
    new_l[0] = prev_l * alpha + p;
    new_m[0] = m_new;
}
