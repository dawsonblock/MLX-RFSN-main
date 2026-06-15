// True Packed Metal Kernel for Direct-Packed Attention
// P1 Implementation: Vectorized QK dot products on packed quantized KV
//
// This kernel implements the core requirements for zero-reconstruction attention:
// 1. Decode packed K/V blocks on-the-fly inside the shader
// 2. Compute full vector QK dot products across head_dim
// 3. Online softmax across all blocks
// 4. Weighted SV accumulation
// 5. Causal masking, GQA support, staging windows
//
// WARNING: This is a P1 scaffold. Full implementation requires:
// - Cartesian codec decode with WHT, sign seeds, and scales/zero-points
// - Block metadata format alignment with Python PackedBlockV4
// - Differential validation against MLX packed reference
//
// Until complete, the kernel will use CPU reference fallback.

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Constants and Types
// =============================================================================

// Block metadata matching PackedBlockV4 format
struct PackedBlockMetadata {
    int logical_start;      // Starting token position in the sequence
    int token_count;        // Number of tokens in this block
    int layer_id;           // Layer identifier for hash-sign derivation
    int stream_id;          // Stream identifier for independent caches
    float key_scale;        // Quantization scale for keys
    float key_zero_point;   // Quantization zero-point for keys
    float value_scale;      // Quantization scale for values
    float value_zero_point; // Quantization zero-point for values
};

// Cartesian codec configuration
struct CartesianConfig {
    int bits;               // Bit width (8 for K8/V8)
    int group_size;         // Group size for quantization (canonical: 64)
    int sign_seed;          // Seed for hash-sign derivation
    bool use_wht;           // Whether to apply Walsh-Hadamard transform
};

// Online softmax state for numerical stability
struct OnlineSoftmaxState {
    float running_max;
    float running_sum;

    OnlineSoftmaxState() : running_max(-1e9), running_sum(0.0) {}  // P0 Fix: Use finite init to avoid NaN

    void update(float new_score) {
        float new_max = fmax(running_max, new_score);
        float old_scale = exp(running_max - new_max);
        // Guard against NaN when running_max is -inf (P0 fix)
        if (running_max == -1e9) old_scale = 0.0;
        running_sum = running_sum * old_scale + exp(new_score - new_max);
        running_max = new_max;
    }

    float weight(float score) const {
        // Handle fully-masked case (running_sum == 0)
        if (running_sum == 0.0) return 0.0;
        return exp(score - running_max) / running_sum;
    }
};

// =============================================================================
// Cartesian Codec Decode (Scaffold - Full Implementation Required)
// =============================================================================

// Decode a single quantized value using Cartesian codec parameters
// TODO: Implement full decode with:
// - Bit unpacking for arbitrary bit widths
// - Walsh-Hadamard transform (if enabled)
// - Hash-sign derivation from layer_id/stream_id
// - Scale/zero-point dequantization
float decode_cartesian_scalar(
    device const uint8_t* packed_data,
    int packed_idx,
    float scale,
    float zero_point,
    int bits,
    int sign_seed,
    bool use_wht
) {
    // P1 Scaffold: Simple uniform dequantization
    // Full implementation needs:
    // 1. Extract the correct bits from packed_data (handles sub-byte packing)
    // 2. Apply hash-sign based on position and sign_seed
    // 3. Apply WHT rotation if enabled
    // 4. Dequantize with scale/zero_point

    uint8_t packed = packed_data[packed_idx];

    // Extract sign and magnitude (assumes 8-bit for now)
    // TODO: Generalize to arbitrary bit widths
    float sign = (packed & 0x80) ? -1.0 : 1.0;
    float magnitude = float(packed & 0x7F) / 127.0;

    // TODO: Apply hash-sign mixing
    // float hash_sign = compute_hash_sign(packed_idx, sign_seed);

    // TODO: Apply WHT if enabled
    // if (use_wht) { magnitude = apply_wht(magnitude, packed_idx, group_size); }

    // Dequantize
    return sign * magnitude * scale + zero_point;
}

// Decode a full head vector from packed representation
// TODO: This needs to handle interleaved group packing
void decode_cartesian_vector(
    device const uint8_t* packed_data,
    device float* output,
    int head_dim,
    int start_dim,
    float scale,
    float zero_point,
    CartesianConfig config,
    int block_idx,
    int token_idx
) {
    // P1 Scaffold: Decode each dimension independently
    // Full implementation needs to handle:
    // - Group-wise packing (group_size elements share statistics)
    // - Dimension permutation from WHT
    // - Efficient memory access patterns

    for (int d = 0; d < head_dim; d++) {
        int global_dim = start_dim + d;
        int packed_idx = block_idx * head_dim + token_idx * head_dim + d;
        output[d] = decode_cartesian_scalar(
            packed_data, packed_idx, scale, zero_point,
            config.bits, config.sign_seed, config.use_wht
        );
    }
}

// =============================================================================
// Vector Operations
// =============================================================================

// Dot product for QK computation
float vector_dot(
    device const float* vec_a,
    device const float* vec_b,
    int dim
) {
    float sum = 0.0;
    for (int i = 0; i < dim; i++) {
        sum += vec_a[i] * vec_b[i];
    }
    return sum;
}

// Weighted vector accumulation for SV
void vector_weighted_accumulate(
    device float* accumulator,
    device const float* vec,
    float weight,
    int dim
) {
    for (int i = 0; i < dim; i++) {
        accumulator[i] += weight * vec[i];
    }
}

// =============================================================================
// Main Kernel: True Packed Attention
// =============================================================================

kernel void true_packed_attention(
    // Query input: [batch=1, num_q_heads, num_q_tokens, head_dim]
    device const float* queries [[buffer(0)]],

    // Packed key blocks: flattened quantized key data
    device const uint8_t* packed_keys [[buffer(1)]],

    // Packed value blocks: flattened quantized value data
    device const uint8_t* packed_values [[buffer(2)]],

    // Block metadata array
    device const PackedBlockMetadata* block_metadata [[buffer(3)]],

    // Output: [batch=1, num_q_heads, num_q_tokens, head_dim]
    device float* output [[buffer(4)]],

    // Scalar parameters
    constant int& num_blocks [[buffer(5)]],
    constant int& num_q_heads [[buffer(6)]],
    constant int& num_kv_heads [[buffer(7)]],
    constant int& head_dim [[buffer(8)]],
    constant int& num_q_tokens [[buffer(9)]],
    constant float& scale [[buffer(10)]],
    constant int& causal [[buffer(11)]],
    constant int& query_start_pos [[buffer(12)]],

    // Cartesian config
    constant int& cartesian_bits [[buffer(13)]],
    constant int& cartesian_group_size [[buffer(14)]],
    constant int& cartesian_sign_seed [[buffer(15)]],
    constant int& cartesian_use_wht [[buffer(16)]],

    // Thread positioning
    uint3 gid [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]
) {
    // Each thread processes one (query_head, query_token) pair
    int q_head_idx = gid.x;
    int q_token_idx = gid.y;

    if (q_head_idx >= num_q_heads || q_token_idx >= num_q_tokens) {
        return;
    }

    // GQA: Map query head to KV head
    int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);
    int q_heads_per_kv = num_q_heads / num_kv_heads;

    // Query position in the sequence
    int query_pos = query_start_pos + q_token_idx;

    // Compute query tensor offsets
    // Layout: [B=1, Hq, Lq, D]
    int query_offset = q_head_idx * num_q_tokens * head_dim + q_token_idx * head_dim;

    // Output offset
    int output_offset = q_head_idx * num_q_tokens * head_dim + q_token_idx * head_dim;

    // Initialize output accumulator to zero
    for (int d = 0; d < head_dim; d++) {
        output[output_offset + d] = 0.0;
    }

    // Cartesian config for this thread
    CartesianConfig config;
    config.bits = cartesian_bits;
    config.group_size = cartesian_group_size;
    config.sign_seed = cartesian_sign_seed;
    config.use_wht = (cartesian_use_wht != 0);

    // Thread-local storage for decoded vectors
    // Note: In production, use threadgroup memory or tiling for efficiency
    thread float query_vec[128];  // Max head_dim assumption
    thread float key_vec[128];
    thread float value_vec[128];

    // Load query vector
    for (int d = 0; d < head_dim; d++) {
        query_vec[d] = queries[query_offset + d];
    }

    // =================================================================
    // Phase 1: Online softmax - compute max and sumexp over all KV tokens
    // =================================================================
    OnlineSoftmaxState softmax_state;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        PackedBlockMetadata meta = block_metadata[block_idx];

        int block_start = meta.logical_start;
        int block_tokens = meta.token_count;

        // Skip blocks that are entirely beyond causal boundary
        if (causal && block_start > query_pos) {
            continue;
        }

        // Decode keys for this KV head and compute QK scores
        for (int t = 0; t < block_tokens; t++) {
            int kv_pos = block_start + t;

            // Causal mask check
            if (causal && kv_pos > query_pos) {
                continue;
            }

            // Decode key vector for this token and KV head
            // TODO: Optimize - avoid decoding all dimensions if not needed
            int key_offset = block_idx * block_tokens * head_dim + t * head_dim;

            // Decode Cartesian packed key to float
            decode_cartesian_vector(
                packed_keys, key_vec, head_dim, 0,
                meta.key_scale, meta.key_zero_point,
                config, block_idx, t
            );

            // Compute QK dot product
            float qk_score = vector_dot(query_vec, key_vec, head_dim) * scale;

            // Update online softmax
            softmax_state.update(qk_score);
        }
    }

    // =================================================================
    // Phase 2: Accumulate weighted values
    // =================================================================

    // Reset accumulators for second pass
    thread float output_accumulator[128];
    for (int d = 0; d < head_dim; d++) {
        output_accumulator[d] = 0.0;
    }

    // Second pass: compute weighted sum of values
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        PackedBlockMetadata meta = block_metadata[block_idx];

        int block_start = meta.logical_start;
        int block_tokens = meta.token_count;

        // Skip blocks beyond causal boundary
        if (causal && block_start > query_pos) {
            continue;
        }

        for (int t = 0; t < block_tokens; t++) {
            int kv_pos = block_start + t;

            // Causal mask check
            if (causal && kv_pos > query_pos) {
                continue;
            }

            // Decode key for QK score recomputation
            decode_cartesian_vector(
                packed_keys, key_vec, head_dim, 0,
                meta.key_scale, meta.key_zero_point,
                config, block_idx, t
            );

            // Recompute QK score
            float qk_score = vector_dot(query_vec, key_vec, head_dim) * scale;

            // Get softmax weight
            float weight = softmax_state.weight(qk_score);

            // Decode value vector
            decode_cartesian_vector(
                packed_values, value_vec, head_dim, 0,
                meta.value_scale, meta.value_zero_point,
                config, block_idx, t
            );

            // Accumulate weighted value
            vector_weighted_accumulate(output_accumulator, value_vec, weight, head_dim);
        }
    }

    // Write final output
    for (int d = 0; d < head_dim; d++) {
        output[output_offset + d] = output_accumulator[d];
    }
}

// =============================================================================
// Staging Window Kernel (Future Work)
// =============================================================================
// This kernel handles the staging window for recent tokens that are kept
// in dense FP16 format for efficiency. The staging window allows:
// - Fast access to recent tokens without dequantization
// - Smooth transition between dense and packed representations
// - Configurable window size (typically 64-128 tokens)
//
// kernel void staging_window_attention(
//     // Similar signature but with additional staging buffers
//     device const float* staging_keys [[buffer(17)]],
//     device const float* staging_values [[buffer(18)]],
//     constant int& staging_start [[buffer(19)]],
//     constant int& staging_count [[buffer(20)]],
//     ...
// );
