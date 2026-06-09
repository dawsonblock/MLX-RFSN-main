"""MLX Metal kernel backend — fast path for Apple Silicon."""

from __future__ import annotations

import math

from ..compat import MLX_AVAILABLE, ensure_mlx_available, mx

from ._common import KernelRouteError


class MetalBackend:
    """MLX Metal kernel backend."""

    name = "metal"

    @classmethod
    def available(cls) -> bool:
        if not MLX_AVAILABLE:
            return False
        return hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel")

    # -- bit packing ---------------------------------------------------------

    @staticmethod
    def pack_bits(codes, bits: int) -> tuple:
        ensure_mlx_available()
        if not isinstance(bits, int) or bits < 2 or bits > 8:
            raise KernelRouteError(f"bits must be in [2, 8], got {bits}")
        if codes.size == 0:
            raise KernelRouteError("Cannot pack empty array")

        # Handle float inputs: check integer-valued
        FLOAT_DTYPES = {
            dt
            for dt in [
                getattr(mx, "float16", None),
                getattr(mx, "float32", None),
                getattr(mx, "bfloat16", None),
            ]
            if dt is not None
        }
        if codes.dtype in FLOAT_DTYPES:
            x_rounded = mx.round(codes)
            if not mx.all(codes == x_rounded).item():
                raise KernelRouteError(
                    "Float codes must be integer-valued"
                )
            codes = x_rounded.astype(mx.int32)

        if codes.dtype in (mx.int8, mx.int16, mx.int32, mx.int64):
            if mx.any(codes < 0).item():
                raise KernelRouteError("Codes cannot be negative")

        codes = codes.astype(mx.uint32)
        max_val = (1 << bits) - 1
        if mx.any(codes > max_val).item():
            raise KernelRouteError(
                f"Codes exceed max value {max_val}"
            )

        n_values = int(codes.size)
        codes_per_word = 32 // bits
        n_words = (n_values + codes_per_word - 1) // codes_per_word
        pad_len = (
            codes_per_word - (n_values % codes_per_word)
        ) % codes_per_word
        if pad_len > 0:
            codes = mx.concatenate(
                [codes, mx.zeros((pad_len,), dtype=mx.uint32)]
            )
        codes = codes.reshape(-1, codes_per_word)
        packed = mx.zeros((n_words,), dtype=mx.uint32)
        for i in range(codes_per_word):
            packed = packed | (codes[:, i] << (i * bits))
        return packed, n_values

    @staticmethod
    def unpack_bits(packed, n_values: int, bits: int):
        ensure_mlx_available()
        if not isinstance(bits, int) or bits < 2 or bits > 8:
            raise KernelRouteError(f"bits must be in [2, 8], got {bits}")
        if n_values <= 0:
            raise KernelRouteError(
                f"n_values must be positive, got {n_values}"
            )
        if packed.size == 0:
            raise KernelRouteError("Cannot unpack from empty buffer")

        codes_per_word = 32 // bits
        required_words = (n_values + codes_per_word - 1) // codes_per_word
        if packed.size < required_words:
            raise KernelRouteError(
                f"Packed buffer too small: "
                f"need {required_words} words"
            )

        packed_view = packed[:required_words].reshape(-1, 1).astype(mx.uint32)
        shifts = (mx.arange(codes_per_word, dtype=mx.uint32) * bits).reshape(
            1, -1
        )
        mask = mx.array((1 << bits) - 1, dtype=mx.uint32)
        codes = (packed_view >> shifts) & mask
        return codes.reshape(-1)[:n_values]

    # -- attention ----------------------------------------------------------

    @staticmethod
    def scaled_dot_product_attention(
        queries,
        keys,
        values,
        scale: float | None = None,
        causal: bool = False,
    ):
        ensure_mlx_available()
        if scale is None:
            scale = 1.0 / math.sqrt(queries.shape[-1])

        if causal:
            B, H, T_q, D = queries.shape
            T_k = keys.shape[2]
            scores = queries @ keys.transpose(0, 1, 3, 2) * scale
            q_pos = mx.arange(T_q, dtype=mx.int32).reshape(1, 1, T_q, 1)
            k_pos = mx.arange(T_k, dtype=mx.int32).reshape(1, 1, 1, T_k)
            offset = T_k - T_q
            causal_mask = (k_pos <= (q_pos + offset)).astype(scores.dtype)
            scores = scores * causal_mask + (1.0 - causal_mask) * mx.array(
                -1e9, dtype=scores.dtype
            )
            weights = mx.softmax(scores, axis=-1)
            return weights @ values

        return mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=scale
        )

    # -- dequant ------------------------------------------------------------

    @staticmethod
    def packed_dequant(
        packed,
        scales,
        n_values: int,
        bits: int,
        group_size: int = 64,
    ):
        ensure_mlx_available()
        if not MetalBackend.available():
            raise KernelRouteError("metal_kernel_api_unavailable")
        if bits < 2 or bits > 8:
            raise KernelRouteError(f"bits must be in [2, 8], got {bits}")
        if group_size <= 0:
            raise KernelRouteError(
                f"group_size must be positive, got {group_size}"
            )
        if n_values <= 0:
            raise KernelRouteError(f"n_values must be positive, got {n_values}")

        codes_per_word = 32 // bits
        required_words = math.ceil(int(n_values) / codes_per_word)
        if int(packed.size) < required_words:
            raise KernelRouteError(
                f"packed has insufficient words: have={int(packed.size)} "
                f"need={required_words}"
            )

        required_scales = math.ceil(int(n_values) / int(group_size))
        if int(scales.size) < required_scales:
            raise KernelRouteError(
                f"scales has insufficient groups: have={int(scales.size)} "
                f"need={required_scales}"
            )

        qmax = (1 << (bits - 1)) - 1
        out_dtype = mx.float32
        source = """
            uint gid = thread_position_in_grid.x;
            uint n = n_buf[0];
            if (gid >= n) { return; }

            uint bits = bits_buf[0];
            uint group_size = group_buf[0];
            uint qmax = qmax_buf[0];

            uint codes_per_word = 32u / bits;
            uint word_idx = gid / codes_per_word;
            uint offset = (gid % codes_per_word) * bits;
            uint mask = (1u << bits) - 1u;

            uint code = (packed[word_idx] >> offset) & mask;
            uint group_idx = gid / group_size;
            float scale = float(scales[group_idx]);

            float deq = (float(code) - float(qmax)) * scale;
            out[gid] = T(deq);
        """

        kernel = mx.fast.metal_kernel(
            name="rfsn_packed_dequant",
            input_names=[
                "packed",
                "scales",
                "n_buf",
                "bits_buf",
                "group_buf",
                "qmax_buf",
            ],
            output_names=["out"],
            source=source,
        )

        n_buf = mx.array([int(n_values)], dtype=mx.uint32)
        bits_buf = mx.array([int(bits)], dtype=mx.uint32)
        group_buf = mx.array([int(group_size)], dtype=mx.uint32)
        qmax_buf = mx.array([int(qmax)], dtype=mx.uint32)

        threadgroup_x = 256 if int(n_values) >= 256 else max(1, int(n_values))
        outputs = kernel(
            inputs=[
                packed,
                scales.astype(mx.float32),
                n_buf,
                bits_buf,
                group_buf,
                qmax_buf,
            ],
            template=[("T", out_dtype)],
            grid=(int(n_values), 1, 1),
            threadgroup=(threadgroup_x, 1, 1),
            output_shapes=[(int(n_values),)],
            output_dtypes=[out_dtype],
        )
        return outputs[0]

    # -- WHT ----------------------------------------------------------------

    @staticmethod
    def wht_transform(x):
        """Walsh-Hadamard transform over the last dimension in blocks of 64.

        The transform is self-inverse when normalised by 1/sqrt(64) = 1/8.

        Raises:
            ValueError: If the tensor is empty or last dim is not a multiple
                of 64.
            KernelRouteError: If the Metal kernel API is unavailable.
        """
        ensure_mlx_available()
        if not MetalBackend.available():
            raise KernelRouteError("metal_kernel_api_unavailable")
        if x.size == 0:
            raise ValueError("Cannot WHT-transform empty tensor")
        if x.shape[-1] % 64 != 0:
            raise ValueError(
                f"Last dimension must be a multiple of 64, got {x.shape[-1]}"
            )

        out_dtype = mx.float32
        # Each threadgroup processes exactly one 64-element WHT block.
        # lid  = lane within the block [0, 63]
        # tgid = block index
        # gid  = flat element index = tgid * 64 + lid
        source = """
            uint tgid = threadgroup_position_in_grid.x;
            uint lid  = thread_position_in_threadgroup.x;
            uint gid  = tgid * 64u + lid;
            uint n    = n_buf[0];

            threadgroup float smem[64];

            // Load — bounds guard for partial last block (shouldn't happen
            // since we validated last-dim % 64 == 0, but be safe).
            smem[lid] = (gid < n) ? float(x[gid]) : 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Butterfly WHT (Hadamard order, in-place on smem).
            for (uint step = 1u; step < 64u; step <<= 1u) {
                float a = smem[lid];
                float b = smem[lid ^ step];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                smem[lid] = ((lid & step) == 0u) ? (a + b) : (b - a);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // Normalise by 1/sqrt(64) = 0.125 so the transform is
            // self-inverse (applying it twice returns the original).
            if (gid < n) {
                out[gid] = T(smem[lid] * 0.125f);
            }
        """

        kernel = mx.fast.metal_kernel(
            name="rfsn_wht64",
            input_names=["x", "n_buf"],
            output_names=["out"],
            source=source,
        )

        shape = x.shape
        flat = x.reshape(-1).astype(mx.float32)
        n = int(flat.size)
        n_blocks = n // 64  # guaranteed integer division after shape check
        n_buf = mx.array([n], dtype=mx.uint32)

        outputs = kernel(
            inputs=[flat, n_buf],
            template=[("T", out_dtype)],
            # One threadgroup per 64-element block; 64 threads per group.
            grid=(n_blocks * 64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(n,)],
            output_dtypes=[out_dtype],
        )
        return outputs[0].reshape(shape)

    # -- hash signs ---------------------------------------------------------

    @staticmethod
    def apply_hash_signs(x, seed: int):
        ensure_mlx_available()
        if not MetalBackend.available():
            raise KernelRouteError("metal_kernel_api_unavailable")

        source = """
            uint gid = thread_position_in_grid.x;
            uint n = n_buf[0];
            uint seed_val = seed_buf[0];

            if (gid >= n) { return; }

            uint state = gid ^ seed_val;
            state += 0x9E3779B9u;
            state ^= state >> 16;
            state *= 0x85ebca6bu;
            state ^= state >> 13;
            state *= 0xc2b2ae35u;
            state ^= state >> 16;

            T sign = (state & 1u) ? T(-1.0f) : T(1.0f);
            out[gid] = x[gid] * sign;
        """

        kernel = mx.fast.metal_kernel(
            name="rfsn_hash_sign",
            input_names=["x", "seed_buf", "n_buf"],
            output_names=["out"],
            source=source,
        )

        flat = mx.array(x.reshape(-1))
        n = int(flat.size)
        seed_buf = mx.array([seed & 0xFFFFFFFF], dtype=mx.uint32)
        n_buf = mx.array([n], dtype=mx.uint32)
        threadgroup_x = 256 if n >= 256 else max(1, n)
        outputs = kernel(
            inputs=[flat, seed_buf, n_buf],
            template=[("T", flat.dtype)],
            grid=(n, 1, 1),
            threadgroup=(threadgroup_x, 1, 1),
            output_shapes=[(n,)],
            output_dtypes=[flat.dtype],
        )
        return outputs[0].reshape(x.shape)

    # -- quantized attention decode -----------------------------------------

    @staticmethod
    def quantized_attention_decode(
        queries,
        packed_k,
        packed_v,
        scales_k,
        scales_v,
        n_keys: int,
        bits: int,
        group_size: int = 64,
        scale: float | None = None,
    ):
        ensure_mlx_available()
        if scale is None:
            n_h, d_head = queries.shape
            scale = 1.0 / math.sqrt(d_head)

        n_h, d_head = queries.shape
        n_values = n_h * n_keys * d_head

        k_deq = MetalBackend.packed_dequant(
            packed_k, scales_k, n_values, bits, group_size
        )
        v_deq = MetalBackend.packed_dequant(
            packed_v, scales_v, n_values, bits, group_size
        )

        k_deq = k_deq.reshape(n_h, n_keys, d_head)[:, None, :, :]
        v_deq = v_deq.reshape(n_h, n_keys, d_head)[:, None, :, :]
        queries = queries[:, None, None, :]

        out = mx.fast.scaled_dot_product_attention(
            queries, k_deq, v_deq, scale=scale
        )
        return out[:, 0, 0, :]
