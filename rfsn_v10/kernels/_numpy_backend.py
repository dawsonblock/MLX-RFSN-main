"""Pure NumPy kernel backend — slow but universal."""

from __future__ import annotations

import math

import numpy as np

from ._common import KernelRouteError


class NumpyBackend:
    """Pure NumPy reference backend."""

    name = "numpy"

    @classmethod
    def available(cls) -> bool:
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            return False

    # -- bit packing ---------------------------------------------------------

    @staticmethod
    def pack_bits(codes, bits: int) -> tuple:
        if not isinstance(bits, int) or bits < 2 or bits > 8:
            raise KernelRouteError(f"bits must be in [2, 8], got {bits}")
        arr = np.asarray(codes)
        if arr.size == 0:
            raise KernelRouteError("Cannot pack empty array")
        if np.issubdtype(arr.dtype, np.floating):
            rounded = np.rint(arr)
            if not np.allclose(arr, rounded):
                raise KernelRouteError("Float codes must be integer-valued")
            arr = rounded.astype(np.int32)
        if np.any(arr < 0):
            raise KernelRouteError("Codes cannot be negative")
        arr = arr.astype(np.uint32)
        max_val = (1 << bits) - 1
        if np.any(arr > max_val):
            raise KernelRouteError(f"Codes exceed max value {max_val}")

        n_values = int(arr.size)
        codes_per_word = 32 // bits
        n_words = (n_values + codes_per_word - 1) // codes_per_word
        pad_len = (
            codes_per_word - (n_values % codes_per_word)
        ) % codes_per_word
        if pad_len:
            arr = np.concatenate([arr, np.zeros(pad_len, dtype=np.uint32)])
        arr = arr.reshape(-1, codes_per_word)
        packed = np.zeros(n_words, dtype=np.uint32)
        for i in range(codes_per_word):
            packed |= arr[:, i] << (i * bits)
        return packed, n_values

    @staticmethod
    def unpack_bits(packed, n_values: int, bits: int):
        if not isinstance(bits, int) or bits < 2 or bits > 8:
            raise KernelRouteError(f"bits must be in [2, 8], got {bits}")
        if n_values <= 0:
            raise KernelRouteError(
                f"n_values must be positive, got {n_values}"
            )
        packed_arr = np.asarray(packed)
        if packed_arr.size == 0:
            raise KernelRouteError("Cannot unpack from empty buffer")
        codes_per_word = 32 // bits
        required_words = (n_values + codes_per_word - 1) // codes_per_word
        if packed_arr.size < required_words:
            raise KernelRouteError(
                f"Packed buffer too small: need {required_words} words"
            )
        view = packed_arr[:required_words].reshape(-1, 1).astype(np.uint32)
        shifts = (np.arange(codes_per_word, dtype=np.uint32) * bits).reshape(
            1, -1
        )
        mask = (1 << bits) - 1
        codes = (view >> shifts) & mask
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
        q = np.asarray(queries)
        k = np.asarray(keys)
        v = np.asarray(values)
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        if causal:
            B, H, T_q, _ = q.shape
            T_k = k.shape[2]
            q_pos = np.arange(T_q).reshape(1, 1, T_q, 1)
            k_pos = np.arange(T_k).reshape(1, 1, 1, T_k)
            offset = T_k - T_q
            mask = k_pos <= (q_pos + offset)
            scores = np.where(mask, scores, -1e9)
        max_score = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - max_score)
        weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        return np.matmul(weights, v)

    # -- dequant ------------------------------------------------------------

    @staticmethod
    def packed_dequant(
        packed,
        scales,
        n_values: int,
        bits: int,
        group_size: int = 64,
    ):
        if bits < 2 or bits > 8:
            raise KernelRouteError(f"bits must be in [2, 8], got {bits}")
        if group_size <= 0:
            raise KernelRouteError(
                f"group_size must be positive, got {group_size}"
            )
        if n_values <= 0:
            raise KernelRouteError(
                f"n_values must be positive, got {n_values}"
            )
        codes_per_word = 32 // bits
        required_words = math.ceil(int(n_values) / codes_per_word)
        packed_arr = np.asarray(packed)
        if packed_arr.size < required_words:
            raise KernelRouteError(
                f"packed has insufficient words: have={packed_arr.size} "
                f"need={required_words}"
            )
        scales_arr = np.asarray(scales).astype(np.float32)
        required_scales = math.ceil(int(n_values) / int(group_size))
        if scales_arr.size < required_scales:
            raise KernelRouteError(
                f"scales has insufficient groups: "
                f"have={scales_arr.size} need={required_scales}"
            )

        qmax = (1 << (bits - 1)) - 1
        words = packed_arr[:required_words].astype(np.uint32)
        # Extract all codes
        shifts = np.arange(codes_per_word, dtype=np.uint32) * bits
        mask = (1 << bits) - 1
        all_codes = ((words.reshape(-1, 1) >> shifts) & mask).reshape(-1)
        codes = all_codes[:n_values].astype(np.float32)
        group_idx = np.arange(n_values, dtype=np.int32) // group_size
        scale_vals = scales_arr[group_idx]
        return (codes - qmax) * scale_vals

    # -- WHT ----------------------------------------------------------------

    @staticmethod
    def wht_transform(x):
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            raise KernelRouteError("Cannot WHT-transform empty tensor")
        if arr.shape[-1] % 64 != 0:
            raise KernelRouteError(
                "Last dimension must be a multiple of 64"
            )
        original_shape = arr.shape
        flat = arr.reshape(-1, 64)
        out = np.zeros_like(flat)
        for i in range(flat.shape[0]):
            vec = flat[i].copy()
            step = 1
            while step < 64:
                partner = np.arange(64) ^ step
                a = vec.copy()
                b = vec[partner]
                vec = np.where(
                    (np.arange(64) & step) == 0, a + b, b - a
                )
                step *= 2
            out[i] = vec / 8.0
        return out.reshape(original_shape)

    # -- hash signs ---------------------------------------------------------

    @staticmethod
    def apply_hash_signs(x, seed: int):
        arr = np.asarray(x)
        flat = arr.reshape(-1)
        n = flat.size
        signs = np.ones(n, dtype=flat.dtype)
        seed_val = seed & 0xFFFFFFFF
        for gid in range(n):
            state = (gid ^ seed_val) & 0xFFFFFFFF
            state += 0x9E3779B9
            state &= 0xFFFFFFFF
            state ^= state >> 16
            state &= 0xFFFFFFFF
            state *= 0x85EBCA6B
            state &= 0xFFFFFFFF
            state ^= state >> 13
            state &= 0xFFFFFFFF
            state *= 0xC2B2AE35
            state &= 0xFFFFFFFF
            state ^= state >> 16
            state &= 0xFFFFFFFF
            signs[gid] = -1.0 if (state & 1) else 1.0
        return (flat * signs).reshape(arr.shape)

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
        q = np.asarray(queries)
        n_h, d_head = q.shape
        if scale is None:
            scale = 1.0 / math.sqrt(d_head)
        n_values = n_h * n_keys * d_head

        k_deq = NumpyBackend.packed_dequant(
            packed_k, scales_k, n_values, bits, group_size
        )
        v_deq = NumpyBackend.packed_dequant(
            packed_v, scales_v, n_values, bits, group_size
        )

        k_deq = k_deq.reshape(n_h, n_keys, d_head)[:, None, :, :]
        v_deq = v_deq.reshape(n_h, n_keys, d_head)[:, None, :, :]
        q = q[:, None, None, :]

        out = NumpyBackend.scaled_dot_product_attention(
            q, k_deq, v_deq, scale=scale
        )
        return out[:, 0, 0, :]
