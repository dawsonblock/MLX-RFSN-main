// True Packed Metal Kernel v2 - Full Implementation
// Implements vectorized QK dot products with on-the-fly Cartesian codec decode

#include <metal_stdlib>
using namespace metal;

// Include Cartesian decode functions
// Note: In production, these would be in a separate .metal file and compiled together
// For now, we include the key functions inline

// =============================================================================
// Hash-Sign Derivation (Matching Python cartesian_codec.py)
// =============================================================================

inline int derive_hash_sign(int layer_id, int stream_id, int position, int dim_idx, int sign_seed) {
    int h = sign_seed;
    h = h * 31 + layer_id;
    h = h * 31 + stream_id;
    h = h * 31 + position;
    h = h * 31 + dim_idx;
    return (h & 1) ? 1 : -1;
}

// =============================================================================
// Bit Unpacking for Sub-Byte Precision
// =============================================================================

inline uint8_t unpack_bits(
    device const uint8_t* packed_data,
    int bit_offset,
    int num_bits
) {
    int byte_idx = bit_offset / 8;
    int bit_idx = bit_offset % 8;

    uint16_t buffer = (uint16_t(packed_data[byte_idx]) << 8);
    if (bit_idx + num_bits > 8) {
        buffer |= packed_data[byte_idx + 1];
    }

    int shift = 16 - bit_idx - num_bits;
    uint16_t mask = (1u << num_bits) - 1;
    return uint8_t((buffer >> shift) & mask);
}

// =============================================================================
// Cartesian Decode Scalar
// =============================================================================

inline float cartesian_decode_scalar(
    device const uint8_t* packed_data,
    int packed_offset,
    int bits,
    int dim_idx,
    int position,
    int layer_id,
    int stream_id,
    int sign_seed,
    float scale,
    float zero_point
) {
    // Unpack bits
    uint8_t packed = unpack_bits(packed_data, packed_offset, bits);

    // Map to [-1, 1]
    int max_val = (1 << bits) - 1;
    float normalized = (float(packed) / float(max_val)) * 2.0 - 1.0;

    // Apply hash-sign
    int hash_sign = derive_hash_sign(layer_id, stream_id, position, dim_idx, sign_seed);
    float signed_value = normalized * float(hash_sign);

    // Dequantize
    return signed_value * scale + zero_point;
}

// =============================================================================
// Block Metadata Structure
// =============================================================================

struct PackedBlockMetadata {
    int logical_start;
    int token_count;
    int layer_id;
    int stream_id;
    float key_scale;
    float key_zero_point;
    float value_scale;
    float value_zero_point;
    int bits;
    int sign_seed;
};

// =============================================================================
// Online Softmax State
// =============================================================================

struct OnlineSoftmaxState {
    float running_max;
    float running_sum;

    OnlineSoftmaxState() : running_max(-1e9), running_sum(0.0) {}

    void update(float new_score) {
        float new_max = fmax(running_max, new_score);
        float old_scale = (running_max == -1e9) ? 0.0 : exp(running_max - new_max);
        running_sum = running_sum * old_scale + exp(new_score - new_max);
        running_max = new_max;
    }

    float weight(float score) const {
        if (running_sum == 0.0) return 0.0;
        return exp(score - running_max) / running_sum;
    }
};

// =============================================================================
// QK Dot Product with On-the-Fly Decode
// =============================================================================

// Compute QK dot product for one query position against one KV token
// Decodes the key vector on-the-fly without materializing it
inline float qk_dot_ondemand(
    device const float* query,
    device const uint8_t* packed_keys,
    int key_packed_offset,
    int head_dim,
    int bits,
    int position,
    int layer_id,
    int stream_id,
    int sign_seed,
    float scale,
    float zero_point
) {
    float dot = 0.0;

    // Decode each dimension and accumulate dot product
    for (int d = 0; d < head_dim; d++) {
        float q_val = query[d];

        float k_val = cartesian_decode_scalar(
            packed_keys,
            key_packed_offset + d * bits,
            bits,
            d,
            position,
            layer_id,
            stream_id,
            sign_seed,
            scale,
            zero_point
        );

        dot += q_val * k_val;
    }

    return dot;
}

// =============================================================================
// SV Accumulation with On-the-Fly Decode
// =============================================================================

// Accumulate weighted value vectors
// Decodes value vectors on-the-fly without materializing full history
inline void sv_accumulate_ondemand(
    thread float* accumulator,
    device const uint8_t* packed_values,
    int value_packed_offset,
    float weight,
    int head_dim,
    int bits,
    int position,
    int layer_id,
    int stream_id,
    int sign_seed,
    float scale,
    float zero_point
) {
    for (int d = 0; d < head_dim; d++) {
        float v_val = cartesian_decode_scalar(
            packed_values,
            value_packed_offset + d * bits,
            bits,
            d,
            position,
            layer_id,
            stream_id,
            sign_seed,
            scale,
            zero_point
        );

        accumulator[d] += weight * v_val;
    }
}

// =============================================================================
// Main Kernel: True Packed Attention v2
// =============================================================================

kernel void true_packed_attention_v2(
    // Query input: [batch=1, num_q_heads, num_q_tokens, head_dim]
    device const float* queries [[buffer(0)]],

    // Packed key blocks: concatenated quantized key data
    device const uint8_t* packed_keys [[buffer(1)]],

    // Packed value blocks: concatenated quantized value data
    device const uint8_t* packed_values [[buffer(2)]],

    // Block metadata array
    device const PackedBlockMetadata* block_metadata [[buffer(3)]],

    // Block offset array: byte offset of each block in packed data
    device const int* key_block_offsets [[buffer(4)]],
    device const int* value_block_offsets [[buffer(5)]],

    // Output: [batch=1, num_q_heads, num_q_tokens, head_dim]
    device float* output [[buffer(6)]],

    // Causal mask lookup: prefix sum of tokens per block for position calculation
    device const int* prefix_sum_tokens [[buffer(7)]],

    // Scalar parameters
    constant int& num_blocks [[buffer(8)]],
    constant int& num_q_heads [[buffer(9)]],
    constant int& num_kv_heads [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    constant int& num_q_tokens [[buffer(12)]],
    constant float& scale_factor [[buffer(13)]],
    constant int& causal [[buffer(14)]],
    constant int& query_start_pos [[buffer(15)]],

    // Thread positioning
    uint3 gid [[thread_position_in_grid]]
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

    // Query tensor offset: [Hq, Lq, D]
    int query_offset = q_head_idx * num_q_tokens * head_dim + q_token_idx * head_dim;

    // Output offset
    int output_offset = q_head_idx * num_q_tokens * head_dim + q_token_idx * head_dim;

    // Thread-local output accumulator
    // Use threadgroup memory for larger head_dim if needed
    float output_acc[128];  // Max head_dim = 128
    for (int d = 0; d < head_dim; d++) {
        output_acc[d] = 0.0;
    }

    // Online softmax state for this query position
    OnlineSoftmaxState softmax_state;

    // Track if any tokens were processed (for fully-masked case)
    bool any_token_processed = false;

    // =============================================================================
    // First Pass: Compute All QK Scores and Online Softmax
    // =============================================================================

    // We iterate through all blocks and tokens, computing QK scores
    // and accumulating online softmax statistics

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Get block metadata
        PackedBlockMetadata meta = block_metadata[block_idx];

        // Get byte offsets for this block
        int key_offset_bytes = key_block_offsets[block_idx];
        int value_offset_bytes = value_block_offsets[block_idx];

        // Calculate bit offset for the key head we need
        // Within a block: [head, token, dim] layout
        int tokens_in_block = meta.token_count;
        int bits = meta.bits;

        // Key offset to the start of our KV head's data
        int kv_head_key_bit_offset = key_offset_bytes * 8 + kv_head_idx * tokens_in_block * head_dim * bits;
        int kv_head_value_bit_offset = value_offset_bytes * 8 + kv_head_idx * tokens_in_block * head_dim * bits;

        // Process each token in this block
        for (int t = 0; t < tokens_in_block; t++) {
            // Calculate absolute position of this KV token
            int kv_pos = meta.logical_start + t;

            // Causal masking: only attend to tokens before or at query position
            if (causal && kv_pos > query_pos) {
                continue;
            }

            any_token_processed = true;

            // Compute QK dot product with on-the-fly decode
            int key_token_bit_offset = kv_head_key_bit_offset + t * head_dim * bits;

            float qk_score = qk_dot_ondemand(
                queries + query_offset,
                packed_keys,
                key_token_bit_offset,
                head_dim,
                bits,
                kv_pos,
                meta.layer_id,
                meta.stream_id,
                meta.sign_seed,
                meta.key_scale,
                meta.key_zero_point
            );

            // Apply attention scale (1/sqrt(head_dim))
            qk_score *= scale_factor;

            // Update online softmax
            softmax_state.update(qk_score);
        }
    }

    // Handle fully-masked case (all positions masked out)
    if (!any_token_processed) {
        for (int d = 0; d < head_dim; d++) {
            output[output_offset + d] = 0.0;
        }
        return;
    }

    // =============================================================================
    // Second Pass: Weighted Accumulation of Values
    // =============================================================================

    // Reset tracking and accumulate weighted values
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        PackedBlockMetadata meta = block_metadata[block_idx];

        int key_offset_bytes = key_block_offsets[block_idx];
        int value_offset_bytes = value_block_offsets[block_idx];

        int tokens_in_block = meta.token_count;
        int bits = meta.bits;

        int kv_head_key_bit_offset = key_offset_bytes * 8 + kv_head_idx * tokens_in_block * head_dim * bits;
        int kv_head_value_bit_offset = value_offset_bytes * 8 + kv_head_idx * tokens_in_block * head_dim * bits;

        for (int t = 0; t < tokens_in_block; t++) {
            int kv_pos = meta.logical_start + t;

            if (causal && kv_pos > query_pos) {
                continue;
            }

            // Recompute QK score to get weight
            int key_token_bit_offset = kv_head_key_bit_offset + t * head_dim * bits;

            float qk_score = qk_dot_ondemand(
                queries + query_offset,
                packed_keys,
                key_token_bit_offset,
                head_dim,
                bits,
                kv_pos,
                meta.layer_id,
                meta.stream_id,
                meta.sign_seed,
                meta.key_scale,
                meta.key_zero_point
            );

            qk_score *= scale_factor;

            // Get softmax weight
            float weight = softmax_state.weight(qk_score);

            // Accumulate weighted value
            int value_token_bit_offset = kv_head_value_bit_offset + t * head_dim * bits;

            sv_accumulate_ondemand(
                output_acc,
                packed_values,
                value_token_bit_offset,
                weight,
                head_dim,
                bits,
                kv_pos,
                meta.layer_id,
                meta.stream_id,
                meta.sign_seed,
                meta.value_scale,
                meta.value_zero_point
            );
        }
    }

    // =============================================================================
    // Write Output
    // =============================================================================

    for (int d = 0; d < head_dim; d++) {
        output[output_offset + d] = output_acc[d];
    }
}

// =============================================================================
// Optimized Kernel: True Packed Attention with Shared Memory
// =============================================================================
// This version uses threadgroup memory for better performance on larger head_dim

kernel void true_packed_attention_shared(
    // Same buffers as v2
    device const float* queries [[buffer(0)]],
    device const uint8_t* packed_keys [[buffer(1)]],
    device const uint8_t* packed_values [[buffer(2)]],
    device const PackedBlockMetadata* block_metadata [[buffer(3)]],
    device const int* key_block_offsets [[buffer(4)]],
    device const int* value_block_offsets [[buffer(5)]],
    device float* output [[buffer(6)]],
    device const int* prefix_sum_tokens [[buffer(7)]],
    constant int& num_blocks [[buffer(8)]],
    constant int& num_q_heads [[buffer(9)]],
    constant int& num_kv_heads [[buffer(10)]],
    constant int& head_dim [[buffer(11)]],
    constant int& num_q_tokens [[buffer(12)]],
    constant float& scale_factor [[buffer(13)]],
    constant int& causal [[buffer(14)]],
    constant int& query_start_pos [[buffer(15)]],

    // Threadgroup memory for query caching
    threadgroup float* shared_query [[threadgroup(0)]],

    uint3 gid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group_size [[threads_per_threadgroup]]
) {
    int q_head_idx = gid.x;
    int q_token_idx = gid.y;

    if (q_head_idx >= num_q_heads || q_token_idx >= num_q_tokens) {
        return;
    }

    int kv_head_idx = q_head_idx / (num_q_heads / num_kv_heads);
    int query_pos = query_start_pos + q_token_idx;
    int query_offset = q_head_idx * num_q_tokens * head_dim + q_token_idx * head_dim;
    int output_offset = q_head_idx * num_q_tokens * head_dim + q_token_idx * head_dim;

    // Load query into threadgroup memory (cooperative load)
    int local_id = lid.x + lid.y * group_size.x;
    int num_threads = group_size.x * group_size.y;

    for (int d = local_id; d < head_dim; d += num_threads) {
        shared_query[d] = queries[query_offset + d];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Rest of computation same as v2, but reads from shared_query
    float output_acc[128];
    for (int d = 0; d < head_dim; d++) {
        output_acc[d] = 0.0;
    }

    OnlineSoftmaxState softmax_state;
    bool any_token_processed = false;

    // First pass: QK scores
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        PackedBlockMetadata meta = block_metadata[block_idx];
        int key_offset_bytes = key_block_offsets[block_idx];
        int tokens_in_block = meta.token_count;
        int bits = meta.bits;

        int kv_head_key_bit_offset = key_offset_bytes * 8 + kv_head_idx * tokens_in_block * head_dim * bits;

        for (int t = 0; t < tokens_in_block; t++) {
            int kv_pos = meta.logical_start + t;
            if (causal && kv_pos > query_pos) continue;

            any_token_processed = true;

            int key_token_bit_offset = kv_head_key_bit_offset + t * head_dim * bits;

            // Compute dot product reading from shared query
            float dot = 0.0;
            for (int d = 0; d < head_dim; d++) {
                float q_val = shared_query[d];
                float k_val = cartesian_decode_scalar(
                    packed_keys,
                    key_token_bit_offset + d * bits,
                    bits,
                    d,
                    kv_pos,
                    meta.layer_id,
                    meta.stream_id,
                    meta.sign_seed,
                    meta.key_scale,
                    meta.key_zero_point
                );
                dot += q_val * k_val;
            }

            dot *= scale_factor;
            softmax_state.update(dot);
        }
    }

    if (!any_token_processed) {
        for (int d = 0; d < head_dim; d++) {
            output[output_offset + d] = 0.0;
        }
        return;
    }

    // Second pass: weighted accumulation
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        PackedBlockMetadata meta = block_metadata[block_idx];
        int key_offset_bytes = key_block_offsets[block_idx];
        int value_offset_bytes = value_block_offsets[block_idx];
        int tokens_in_block = meta.token_count;
        int bits = meta.bits;

        int kv_head_key_bit_offset = key_offset_bytes * 8 + kv_head_idx * tokens_in_block * head_dim * bits;
        int kv_head_value_bit_offset = value_offset_bytes * 8 + kv_head_idx * tokens_in_block * head_dim * bits;

        for (int t = 0; t < tokens_in_block; t++) {
            int kv_pos = meta.logical_start + t;
            if (causal && kv_pos > query_pos) continue;

            int key_token_bit_offset = kv_head_key_bit_offset + t * head_dim * bits;

            float dot = 0.0;
            for (int d = 0; d < head_dim; d++) {
                float q_val = shared_query[d];
                float k_val = cartesian_decode_scalar(
                    packed_keys,
                    key_token_bit_offset + d * bits,
                    bits,
                    d,
                    kv_pos,
                    meta.layer_id,
                    meta.stream_id,
                    meta.sign_seed,
                    meta.key_scale,
                    meta.key_zero_point
                );
                dot += q_val * k_val;
            }

            dot *= scale_factor;
            float weight = softmax_state.weight(dot);

            int value_token_bit_offset = kv_head_value_bit_offset + t * head_dim * bits;
            for (int d = 0; d < head_dim; d++) {
                float v_val = cartesian_decode_scalar(
                    packed_values,
                    value_token_bit_offset + d * bits,
                    bits,
                    d,
                    kv_pos,
                    meta.layer_id,
                    meta.stream_id,
                    meta.sign_seed,
                    meta.value_scale,
                    meta.value_zero_point
                );
                output_acc[d] += weight * v_val;
            }
        }
    }

    for (int d = 0; d < head_dim; d++) {
        output[output_offset + d] = output_acc[d];
    }
}
