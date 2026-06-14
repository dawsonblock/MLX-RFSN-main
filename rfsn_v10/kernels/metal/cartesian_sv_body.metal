    // FIXME: This shader computes a weighted sum of V_packed values, where
    // V_packed is in the transform domain (WHT + hash signs).  The accumulated
    // result is therefore also in the transform domain.  For correctness when
    // use_wht or sign_seed is active, the final output vector must be
    // inverse-transformed (apply inverse hash signs, then inverse WHT per
    // group) before writing to the output buffer.
    //
    // Until this is implemented, the Python dispatch layer falls back to the
    // CPU reference path whenever use_wht or sign_seed is non-zero.
    // See: rfsn_v10/kernels/cartesian_cpu_reference.py

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
