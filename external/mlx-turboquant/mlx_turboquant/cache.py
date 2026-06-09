"""
TurboQuantKVCache: Drop-in replacement for mlx-lm's KVCache.

Stores compressed KV cache using bit-packed indices for real memory savings.
Dequantizes on fetch for compatibility with standard attention.
"""

import math

import mlx.core as mx

from .turbo_quant import TurboQuantCompressor, CompressedKeys, CompressedValues
from .packing import pack_indices, unpack_indices, pack_signs, unpack_signs


class TurboQuantKVCache:
    """TurboQuant-compressed KV cache for mlx-lm.

    Args:
        bits: Bits per coordinate (2-4, supports 0.5 increments like 3.5).
        head_dim: Dimension of each attention head.
        key_seed: Random seed for key rotation.
        value_seed: Random seed for value rotation.
    """

    step = 256

    def __init__(
        self,
        bits: float = 3,
        head_dim: int = 128,
        key_seed: int = 42,
        value_seed: int = 43,
    ):
        self.turbo_bits = bits
        self.head_dim = head_dim
        self.offset = 0

        self.compressor = TurboQuantCompressor(
            bits=bits, dim=head_dim,
            key_seed=key_seed, value_seed=value_seed,
        )

        # Determine packing parameters
        self._fractional = self.compressor.fractional
        if self._fractional:
            # For fractional bits, store as uint8 (mixed bit widths per channel)
            self._pack_indices = False
        else:
            self._pack_indices = True
            self._key_bits = self.compressor.key_pq.bits
            self._val_bits = self.compressor.value_pq.bits

        # Compressed storage (initialized on first update)
        self._key_indices = None
        self._key_norms = None
        self._value_indices = None
        self._value_norms = None

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Compress, store, and return dequantized KV for attention."""
        B, n_kv_heads, S, D = keys.shape
        prev = self.offset

        # Compress new entries
        ck = self.compressor.compress_keys(keys)
        cv = self.compressor.compress_values(values)

        # Allocate or expand storage
        if self._key_indices is None or (prev + S) > self._stored_capacity:
            self._expand_storage(B, n_kv_heads, S, D, keys.dtype)

        # Pack and store
        if self._pack_indices:
            pk = pack_indices(ck.indices, self._key_bits)
            pv = pack_indices(cv.indices, self._val_bits)
            self._key_indices[..., prev:prev + S, :] = pk
            self._value_indices[..., prev:prev + S, :] = pv
        else:
            self._key_indices[..., prev:prev + S, :] = ck.indices
            self._value_indices[..., prev:prev + S, :] = cv.indices

        norms_dim = ck.norms.shape[-1]
        self._key_norms[..., prev:prev + S, :norms_dim] = ck.norms
        self._value_norms[..., prev:prev + S, :norms_dim] = cv.norms

        self.offset += S

        # Dequantize full cache for attention
        if self._pack_indices:
            all_k_idx = unpack_indices(
                self._key_indices[..., :self.offset, :], self._key_bits, D
            )
            all_v_idx = unpack_indices(
                self._value_indices[..., :self.offset, :], self._val_bits, D
            )
        else:
            all_k_idx = self._key_indices[..., :self.offset, :]
            all_v_idx = self._value_indices[..., :self.offset, :]

        all_k_norms = self._key_norms[..., :self.offset, :norms_dim]
        all_v_norms = self._value_norms[..., :self.offset, :norms_dim]

        deq_keys = self.compressor.key_pq.dequantize(all_k_idx, all_k_norms)
        deq_values = self.compressor.value_pq.dequantize(all_v_idx, all_v_norms)

        return deq_keys, deq_values

    def _expand_storage(self, B, n_kv_heads, new_steps, D, dtype):
        """Allocate or expand packed compressed storage."""
        alloc_steps = ((self.step + new_steps - 1) // self.step) * self.step

        if self._pack_indices:
            k_vals_per_int = 32 // self._key_bits
            v_vals_per_int = 32 // self._val_bits
            k_packed_dim = (D + k_vals_per_int - 1) // k_vals_per_int
            v_packed_dim = (D + v_vals_per_int - 1) // v_vals_per_int
            idx_dtype = mx.uint32
        else:
            k_packed_dim = D
            v_packed_dim = D
            idx_dtype = mx.uint8

        # Norms: 1 for integer bits, 2 for fractional (hi + lo)
        norms_dim = 2 if self._fractional else 1
        shape = (B, n_kv_heads, alloc_steps)

        if self._key_indices is not None and self.offset > 0:
            old_k = self._key_indices[..., :self.offset, :]
            old_kn = self._key_norms[..., :self.offset, :]
            old_v = self._value_indices[..., :self.offset, :]
            old_vn = self._value_norms[..., :self.offset, :]

            new_k = mx.zeros((*shape, k_packed_dim), dtype=idx_dtype)
            new_kn = mx.zeros((*shape, norms_dim), dtype=dtype)
            new_v = mx.zeros((*shape, v_packed_dim), dtype=idx_dtype)
            new_vn = mx.zeros((*shape, norms_dim), dtype=dtype)

            self._key_indices = mx.concatenate([old_k, new_k], axis=2)
            self._key_norms = mx.concatenate([old_kn, new_kn], axis=2)
            self._value_indices = mx.concatenate([old_v, new_v], axis=2)
            self._value_norms = mx.concatenate([old_vn, new_vn], axis=2)
        else:
            self._key_indices = mx.zeros((*shape, k_packed_dim), dtype=idx_dtype)
            self._key_norms = mx.zeros((*shape, norms_dim), dtype=dtype)
            self._value_indices = mx.zeros((*shape, v_packed_dim), dtype=idx_dtype)
            self._value_norms = mx.zeros((*shape, norms_dim), dtype=dtype)

        self._stored_capacity = self._key_indices.shape[2]

    def size(self):
        return self.offset

    def empty(self):
        return self._key_indices is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, N, return_array=False, window_size=None):
        from mlx_lm.models.base import create_causal_mask
        offset = self.offset
        if window_size is not None:
            return create_causal_mask(N, offset, window_size=window_size)
        elif N == 1:
            return None
        elif return_array:
            return create_causal_mask(N, offset, window_size=window_size)
        else:
            return "causal"

    @property
    def state(self):
        if self._key_indices is None:
            return []
        return [
            self._key_indices[..., :self.offset, :],
            self._key_norms[..., :self.offset, :],
            self._value_indices[..., :self.offset, :],
            self._value_norms[..., :self.offset, :],
        ]

    @state.setter
    def state(self, v):
        if v is not None and v:
            (self._key_indices, self._key_norms,
             self._value_indices, self._value_norms) = v
            self.offset = self._key_indices.shape[2]
            self._stored_capacity = self._key_indices.shape[2]

    @property
    def meta_state(self):
        return tuple(map(str, (self.offset, self.turbo_bits, self.head_dim)))

    @meta_state.setter
    def meta_state(self, v):
        self.offset = int(v[0])
        self.turbo_bits = float(v[1])
        self.head_dim = int(v[2])

    @property
    def nbytes(self):
        if self._key_indices is None:
            return 0
        total = 0
        for arr in [self._key_indices, self._key_norms,
                     self._value_indices, self._value_norms]:
            total += arr[..., :self.offset, :].nbytes
        return total
