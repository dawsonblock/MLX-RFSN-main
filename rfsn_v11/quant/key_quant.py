"""
Key quantization for RFSN v11 KV cache.

Ported from rfsn_v10/kv_manager.py:
  - _apply_wht_pretransform / _wht_block_recursive
  - _quantize / _dequantize_unsigned (grouped symmetric)
  - _apply_signs_on_the_fly (SplitMix hash-sign preconditioning, 128-LRU)

Metal kernel paths (WHT + hash signs) ported from rfsn_v10/kernels.py.

NOTE ON MLX METAL KERNEL GRID API:
  In MLX's mx.fast.metal_kernel, `grid` specifies TOTAL THREADS (not
  threadgroups, unlike raw Metal). So for n elements with threadgroup=64:
    grid=(n, 1, 1), threadgroup=(64, 1, 1)  → n total threads, n/64 groups
  Inside the kernel:
    tgid = threadgroup_position_in_grid.x  (0..n/64-1)
    lid  = thread_position_in_threadgroup.x  (0..63)
    gid  = tgid * 64 + lid  (0..n-1)
  This is the same as rfsn_v10/kernels.py line 112.

Step-256 pre-allocated buffer: allocate (H, T_allocated, D) where
  T_allocated = ceil(T / 256) * 256
and expand on demand.
"""

from __future__ import annotations

import math
import threading
from threading import get_ident

import mlx.core as mx
import numpy as np

from ..compat import MLX_AVAILABLE, ensure_mlx_available


class KernelRouteError(RuntimeError):
    """Raised when a requested Metal kernel route cannot run."""


def maybe_supports_metal_kernels() -> bool:
    """Return True if mx.fast.metal_kernel is available."""
    if not MLX_AVAILABLE:
        return False
    try:
        return hasattr(mx.fast, "metal_kernel")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Metal kernel: WHT64 (grid dispatch bug FIXED)
# ---------------------------------------------------------------------------

def wht64_metal(x: mx.array, out_dtype=None) -> mx.array:
    """Apply normalized WHT over contiguous 64-value blocks with Metal.

    In MLX's metal_kernel API, grid specifies TOTAL THREADS (not threadgroups).
    So grid=(n, 1, 1) with threadgroup=(64, 1, 1) correctly dispatches n/64
    threadgroups of 64 threads each.

    Args:
        x: Input array with total size divisible by 64.
        out_dtype: Output dtype (default mx.float32).

    Raises:
        KernelRouteError: If Metal kernel API is unavailable.
        ValueError: If total size is not a multiple of 64 or x is empty.
    """
    ensure_mlx_available()
    if out_dtype is None:
        out_dtype = mx.float32
    if not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")
    if x.size == 0:
        raise ValueError("Cannot WHT-transform empty tensor.")
    if x.shape[-1] % 64 != 0:
        raise ValueError("Last dimension must be a multiple of 64.")

    source = """
        uint tgid = threadgroup_position_in_grid.x;
        uint lid = thread_position_in_threadgroup.x;
        uint gid = tgid * 64u + lid;
        uint n = n_buf[0];

        threadgroup float smem[64];
        float val = 0.0f;
        if (gid < n) {
            val = float(x[gid]);
        }

        smem[lid] = val;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint step = 1u; step < 64u; step *= 2u) {
            uint partner = lid ^ step;
            float a = smem[lid];
            float b = smem[partner];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            smem[lid] = ((lid & step) == 0u) ? (a + b) : (b - a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (gid < n) {
            out[gid] = T(smem[lid] / 8.0f);
        }
    """

    kernel = mx.fast.metal_kernel(
        name="rfsn_v11_wht64",
        input_names=["x", "n_buf"],
        output_names=["out"],
        source=source,
    )

    shape = x.shape
    flat = mx.array(x.reshape(-1))
    n = int(flat.size)
    assert n % 64 == 0, f"Total elements {n} must be divisible by 64 for WHT dispatch"
    n_buf = mx.array([n], dtype=mx.uint32)

    # grid = (n, 1, 1) — MLX grid means TOTAL threads, not threadgroups.
    # With threadgroup=(64, 1, 1) this launches n/64 threadgroups of 64 threads.
    outputs = kernel(
        inputs=[flat, n_buf],
        template=[("T", out_dtype)],
        grid=(n, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[out_dtype],
    )

    return outputs[0].reshape(shape)


def apply_hash_signs_metal(x: mx.array, seed: int) -> mx.array:
    """Apply deterministic +/-1 signs with an MLX metal kernel."""
    ensure_mlx_available()
    if not hasattr(mx.fast, "metal_kernel"):
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
        name="rfsn_v11_hash_sign",
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


def apply_hash_signs_with_indices_metal(
    x: mx.array, indices: mx.array, seed: int
) -> mx.array:
    """Apply deterministic +/-1 signs with custom global indices via Metal."""
    ensure_mlx_available()
    if not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")

    source = """
        uint gid = thread_position_in_grid.x;
        uint n = n_buf[0];
        uint seed_val = seed_buf[0];

        if (gid >= n) { return; }

        uint state = uint(indices[gid]) ^ seed_val;
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
        name="rfsn_v11_hash_sign_indices",
        input_names=["x", "indices", "seed_buf", "n_buf"],
        output_names=["out"],
        source=source,
    )

    flat = mx.array(x.reshape(-1))
    idx_flat = mx.array(indices.reshape(-1).astype(mx.uint32))
    n = int(flat.size)
    if int(idx_flat.size) != n:
        raise KernelRouteError(
            f"indices size {idx_flat.size} does not match x size {n}"
        )
    seed_buf = mx.array([seed & 0xFFFFFFFF], dtype=mx.uint32)
    n_buf = mx.array([n], dtype=mx.uint32)

    threadgroup_x = 256 if n >= 256 else max(1, n)
    outputs = kernel(
        inputs=[flat, idx_flat, seed_buf, n_buf],
        template=[("T", flat.dtype)],
        grid=(n, 1, 1),
        threadgroup=(threadgroup_x, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[flat.dtype],
    )
    return outputs[0].reshape(x.shape)


# ---------------------------------------------------------------------------
# KeyQuant class
# ---------------------------------------------------------------------------

class KeyQuant:
    """WHT + grouped symmetric quantization for key vectors.

    Pipeline:
        compress:  [optional sign preconditioning] → [optional WHT] → symmetric quant
        decompress: symmetric dequant → [optional inverse WHT] → [optional inverse signs]

    Args:
        bits: Bits per code (2-8), default 8.
        group_size: Elements per quantization group, default 64.
        use_wht: Apply Walsh-Hadamard pre-transform, default True.
        use_incoherent_signs: Apply SplitMix hash sign preconditioning, default True.
        sign_seed: Seed for deterministic sign generation, default 0.
        prefer_metal: Prefer Metal kernels when available, default True.
    """

    def __init__(
        self,
        bits: int = 8,
        group_size: int = 64,
        use_wht: bool = True,
        use_incoherent_signs: bool = True,
        sign_seed: int = 0,
        prefer_metal: bool = True,
    ):
        self.bits = bits
        self.group_size = group_size
        self.use_wht = use_wht
        self.use_incoherent_signs = use_incoherent_signs
        self.sign_seed = sign_seed
        self.prefer_metal = prefer_metal

        # 128-entry LRU sign cache (shape, seed, dtype, thread_id) → signs tensor
        self._sign_cache: dict = {}
        self._sign_cache_lock = threading.Lock()

        # Step-256 pre-allocated buffer: (H, T_allocated, D)
        self._buf: mx.array | None = None
        self._buf_shape: tuple | None = None  # (H, T_alloc, D)

    # ------------------------------------------------------------------
    # Sign preconditioning (SplitMix hash, self-inverse)
    # ------------------------------------------------------------------

    def _apply_signs_on_the_fly(
        self,
        x: mx.array,
        seed: int,
        indices: mx.array | None = None,
    ) -> mx.array:
        """Apply deterministic ±1 sign preconditioning (self-inverse).

        Calling with the same seed twice restores the original tensor.
        Uses Metal when available; falls back to vectorized NumPy-backed MLX.
        """
        if self.prefer_metal and maybe_supports_metal_kernels():
            if indices is None:
                try:
                    return apply_hash_signs_metal(x, seed)
                except Exception:
                    pass
            else:
                try:
                    return apply_hash_signs_with_indices_metal(x, indices, seed)
                except Exception:
                    pass

        shape = x.shape
        n = x.size

        # Check sign cache (only for default indices)
        if indices is None:
            cache_key = (shape, seed, x.dtype, get_ident())
            with self._sign_cache_lock:
                signs = self._sign_cache.get(cache_key)
            if signs is not None:
                return x * signs

        # Vectorized SplitMix-style hash
        seed_u32 = mx.array(np.uint32(int(seed) & 0xFFFFFFFF))
        if indices is not None:
            idx = indices.reshape(-1).astype(mx.uint32)
        else:
            idx = mx.arange(n, dtype=mx.uint32)
        z = mx.bitwise_xor(idx, seed_u32)
        z = z + mx.array(np.uint32(0x9E3779B9))
        z = mx.bitwise_xor(z, z >> 16) * mx.array(np.uint32(0x85EBCA6B))
        z = mx.bitwise_xor(z, z >> 13) * mx.array(np.uint32(0xC2B2AE35))
        z = mx.bitwise_xor(z, z >> 16)
        parity = z & mx.array(np.uint32(1))
        signs = mx.where(
            parity == 0,
            mx.array(1.0, dtype=x.dtype),
            mx.array(-1.0, dtype=x.dtype),
        ).reshape(shape)

        if indices is None:
            with self._sign_cache_lock:
                if (
                    cache_key not in self._sign_cache
                    and len(self._sign_cache) < 128
                ):
                    self._sign_cache[cache_key] = signs
                else:
                    signs = self._sign_cache.get(cache_key, signs)

        return x * signs

    # ------------------------------------------------------------------
    # WHT (Python fallback — used when Metal unavailable)
    # ------------------------------------------------------------------

    def _wht_block_recursive(self, x: mx.array) -> mx.array:
        """Recursive Walsh-Hadamard transform along last dimension."""
        n = x.shape[-1]
        if n == 1:
            return x
        half = n // 2
        x0 = x[..., :half]
        x1 = x[..., half:]
        y0 = self._wht_block_recursive(x0)
        y1 = self._wht_block_recursive(x1)
        return mx.concatenate([y0 + y1, y0 - y1], axis=-1)

    def _apply_wht_pretransform(self, x: mx.array) -> mx.array:
        """Apply normalized WHT along last dimension. Self-inverse.

        Tries Metal kernel first; falls back to recursive Python path.
        Requires last dimension divisible by 64.
        """
        D = x.shape[-1]
        if D % 64 != 0:
            raise ValueError(f"Last dimension must be a multiple of 64, got {D}")

        shape = x.shape

        # Try Metal kernel (with fixed grid dispatch)
        if self.prefer_metal and maybe_supports_metal_kernels():
            try:
                return wht64_metal(x, out_dtype=x.dtype)
            except Exception:
                pass

        # Python fallback
        x = x.reshape(-1, 64)
        x = self._wht_block_recursive(x)
        x = x / math.sqrt(64)
        return x.reshape(shape)

    # ------------------------------------------------------------------
    # Grouped symmetric quantization
    # ------------------------------------------------------------------

    def _quantize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Grouped symmetric quantization. Returns (codes, scales).

        Codes are in [0, 2*qmax]; zero reconstructs as exactly zero.
        """
        if x.size == 0:
            raise ValueError("Cannot quantize empty tensor")

        original_size = x.size
        n_groups = (original_size + self.group_size - 1) // self.group_size

        pad_len = (self.group_size - (original_size % self.group_size)) % self.group_size
        if pad_len > 0:
            x = mx.concatenate([x, mx.zeros((pad_len,), dtype=x.dtype)])

        x = x.reshape(n_groups, self.group_size)

        qmax = (1 << (self.bits - 1)) - 1
        abs_max = mx.max(mx.abs(x), axis=-1)
        raw_scale = abs_max / float(qmax)
        scales = mx.maximum(raw_scale, mx.array(1e-8, dtype=x.dtype))

        codes = mx.round(x / scales.reshape(-1, 1)) + qmax
        codes = mx.clip(codes.astype(mx.int32), 0, 2 * qmax).astype(mx.uint32)
        codes = codes.reshape(-1)[:original_size]

        return codes, scales

    def _dequantize_unsigned(
        self, q: mx.array, scales: mx.array
    ) -> mx.array:
        """Dequantize symmetric codes back to float."""
        qmax = (1 << (self.bits - 1)) - 1
        max_code = 2 * qmax

        if mx.any(q > max_code).item():
            raise ValueError(
                f"Invalid symmetric quant code: max allowed is {max_code} for {self.bits} bits"
            )

        original_size = q.size
        n_groups = (original_size + self.group_size - 1) // self.group_size

        if scales.size != n_groups:
            raise ValueError(
                f"Scale count mismatch: expected {n_groups}, got {scales.size}"
            )

        pad_len = (self.group_size - (original_size % self.group_size)) % self.group_size
        if pad_len > 0:
            q = mx.concatenate([q, mx.zeros((pad_len,), dtype=mx.uint32)])

        q_f = q.reshape(n_groups, self.group_size).astype(mx.float32)
        q_f = q_f - qmax  # shift back to [-qmax, qmax]
        x = q_f * scales.reshape(-1, 1)

        return x.reshape(-1)[:original_size]

    # ------------------------------------------------------------------
    # Public compress / decompress API
    # ------------------------------------------------------------------

    def compress(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Compress key vectors.

        Pipeline: [signs] → [WHT] → symmetric quant

        Args:
            x: (..., D) float array of key vectors.

        Returns:
            (codes, scales): uint32 flat codes, float32 group scales.
        """
        flat = x.reshape(-1)
        working = x

        if self.use_incoherent_signs:
            working = self._apply_signs_on_the_fly(working, self.sign_seed)

        if self.use_wht:
            working = self._apply_wht_pretransform(working)

        flat_working = working.reshape(-1)
        return self._quantize(flat_working)

    def decompress(
        self, codes: mx.array, scales: mx.array, shape: tuple
    ) -> mx.array:
        """Decompress key vectors.

        Pipeline: symmetric dequant → [inverse WHT] → [inverse signs]

        Args:
            codes: uint32 flat codes.
            scales: float32 group scales.
            shape: Original tensor shape (..., D).

        Returns:
            Reconstructed float32 tensor of shape `shape`.
        """
        working = self._dequantize_unsigned(codes, scales)
        working = working.reshape(shape)

        # WHT is self-inverse (applying twice restores original)
        if self.use_wht:
            working = self._apply_wht_pretransform(working)

        # Signs are self-inverse (same seed)
        if self.use_incoherent_signs:
            working = self._apply_signs_on_the_fly(working, self.sign_seed)

        return working

    def estimate_bytes(self, shape: tuple) -> int:
        """Estimate compressed byte footprint for a tensor of the given shape."""
        n = 1
        for d in shape:
            n *= d
        bits_per_code = self.bits
        codes_per_word = 32 // bits_per_code
        n_words = (n + codes_per_word - 1) // codes_per_word
        n_groups = (n + self.group_size - 1) // self.group_size
        return n_words * 4 + n_groups * 4  # codes (uint32) + scales (float32)
