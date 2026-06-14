    // FIXME: This shader computes dot(Q, K_packed) where K_packed is in the
    // transform domain (WHT + hash signs).  For correctness when use_wht or
    // sign_seed is active, K must be inverse-transformed back to the original
    // domain before the dot product.
    //
    // Correct design:
    //   1. Decode the full K vector (D elements) for this token.
    //   2. Apply inverse hash signs (per-element sign flips from the codec seed).
    //   3. Apply inverse WHT per group (length = group_size).
    //   4. Then compute the dot product with the untransformed Q vector.
    //
    // Until this is implemented, the Python dispatch layer falls back to the
    // CPU reference path whenever use_wht or sign_seed is non-zero.
    // See: rfsn_v10/kernels/cartesian_cpu_reference.py
    //
    // The same issue exists in cartesian_sv_body.metal: the accumulated output
    // is in the transform domain and must be inverse-transformed once after
    // the weighted sum.

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

    int kv_offset = ((b * Hkv + hkv) * Lkv + k_pos);

    int codes_per_word = 32 / bits;
    int words_per_vec = (D + codes_per_word - 1) / codes_per_word;
    int mask = (1 << bits) - 1;
    int qmax = (1 << (bits - 1)) - 1;
    int n_groups = (D + group_size - 1) / group_size;

    // Precompute scale offset for this (b, hkv, k_pos) — per-token scales
    int scale_base = ((b * Hkv + hkv) * Lkv + k_pos) * n_groups;

    for (int q_pos = 0; q_pos < Lq; ++q_pos) {
        int q_offset = ((b * Hq + hq) * Lq + q_pos) * D;

        float score = 0.0f;
        for (int d = 0; d < D; ++d) {
            int word_idx = d / codes_per_word;
            int bit_offset = (d % codes_per_word) * bits;

            int packed_idx = kv_offset * words_per_vec + word_idx;
            uint32_t word = packed_codes[packed_idx];
            int code = int((word >> bit_offset) & mask);

            int group_idx = d / group_size;
            float scale = scales[scale_base + group_idx];
            float k_val = (float(code) - float(qmax)) * scale;

            float q_val = queries[q_offset + d];
            score += q_val * k_val;
        }

        score *= scale_factor;

        int out_idx = ((b * Hq + hq) * Lq + q_pos) * Lkv + k_pos;
        scores[out_idx] = score;
    }
