// TurboPolar fused dequant-QK kernel
// Inputs: q (float, D), radii (half, block_size * D/2), angle_codes (uchar, block_size * D/2)
// Output: scores (float, block_size)
//
// One thread per token in the block. Each thread reconstructs its K vector
// from polar coordinates and accumulates the dot product with Q.

#include <metal_stdlib>
using namespace metal;

kernel void tqpolar_fused_dequant_qk(
    device const float* q,
    device const half* radii,
    device const uchar* angle_codes,
    device float* scores,
    constant int& head_dim,
    constant int& block_size,
    constant int& angle_bits,
    uint tid [[thread_position_in_grid]]
)
{
    if (tid >= uint(block_size)) { return; }

    const float TWO_PI = 6.28318530718f;
    int pairs = head_dim / 2;
    float bin_width = TWO_PI / float(1 << angle_bits);
    float half_bin = bin_width * 0.5f;

    float score = 0.0f;
    for (int p = 0; p < pairs; ++p) {
        int idx = int(tid) * pairs + p;
        float radius = float(radii[idx]);
        uint code = uint(angle_codes[idx]);
        float angle = float(code) * bin_width + half_bin;

        float x = radius * cos(angle);
        float y = radius * sin(angle);

        score += q[p * 2] * x;
        score += q[p * 2 + 1] * y;
    }
    scores[int(tid)] = score;
}
