"""Kernel registry for polar_fused — dispatch to Metal or reference implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# MLX optional at import time
try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    HAS_MLX = False


@dataclass(frozen=True)
class KernelSignature:
    """Identifier for a compiled kernel variant."""
    name: str
    bits: int
    head_dim: int
    variant: str  # "scalar", "tiled", "simd"


class KernelRegistry:
    """Registry of Polar fused kernels.

    Attempts to load and cache compiled Metal kernels.  Falls back to
    pure-MLX reference implementations when Metal compilation is unavailable.
    """

    def __init__(self) -> None:
        self._cache: dict[KernelSignature, Callable[..., Any]] = {}
        self._metal_available = self._check_metal()

    # ------------------------------------------------------------------
    # Public dispatch API
    # ------------------------------------------------------------------

    def get_qk_kernel(
        self,
        bits: int,
        head_dim: int,
        query_length: int,
        key_length: int,
    ) -> Callable[..., Any]:
        """Return the best QK kernel for the given parameters."""
        sig = self._select_qk_variant(bits, head_dim, query_length, key_length)
        return self._get_or_compile(sig)

    def get_sv_kernel(
        self,
        bits: int,
        head_dim: int,
        query_length: int,
        key_length: int,
    ) -> Callable[..., Any]:
        """Return the best SV kernel for the given parameters."""
        sig = self._select_sv_variant(bits, head_dim, query_length, key_length)
        return self._get_or_compile(sig)

    # ------------------------------------------------------------------
    # Selection heuristics
    # ------------------------------------------------------------------

    def _select_qk_variant(
        self,
        bits: int,
        head_dim: int,
        query_length: int,
        key_length: int,
    ) -> KernelSignature:
        # Decode: Lq == 1 → scalar is fine
        # Prefill: larger Lq → tiled may be better
        if query_length == 1:
            return KernelSignature("qk", bits, head_dim, "scalar")
        return KernelSignature("qk", bits, head_dim, "scalar")  # start simple

    def _select_sv_variant(
        self,
        bits: int,
        head_dim: int,
        query_length: int,
        key_length: int,
    ) -> KernelSignature:
        if query_length == 1:
            return KernelSignature("sv", bits, head_dim, "scalar")
        return KernelSignature("sv", bits, head_dim, "scalar")

    # ------------------------------------------------------------------
    # Compilation / caching
    # ------------------------------------------------------------------

    def _get_or_compile(self, sig: KernelSignature) -> Callable[..., Any]:
        if sig in self._cache:
            return self._cache[sig]

        kernel = self._compile(sig)
        self._cache[sig] = kernel
        return kernel

    def _compile(self, sig: KernelSignature) -> Callable[..., Any]:
        if not self._metal_available:
            return self._reference_impl(sig)

        # TODO: load .metal source, compile via mlx.core.fast, wrap
        # For now, fall back to reference
        return self._reference_impl(sig)

    def _reference_impl(self, sig: KernelSignature) -> Callable[..., Any]:
        """Return a pure-MLX reference implementation for the kernel."""
        if sig.name == "qk":
            return self._reference_qk
        elif sig.name == "sv":
            return self._reference_sv
        else:
            raise ValueError(f"Unknown kernel: {sig.name}")

    # ------------------------------------------------------------------
    # Reference implementations (pure MLX, slow but correct)
    # ------------------------------------------------------------------

    def _reference_qk(
        self,
        rotated_queries: Any,      # (B, Hq, Lq, D)
        packed_key_indices: Any,     # (B, Hkv, Lkv, packed_dim)
        key_norms: Any,              # (B, Hkv, Lkv)
        key_centroids: Any,          # (n_centroids,)
        scale: float,
        bits: int,
        values_per_word: int,
    ) -> Any:
        """Reference QK: unpack keys, lookup centroids, dot with queries."""
        if mx is None:
            raise RuntimeError("MLX is not installed")

        B, Hq, Lq, D = rotated_queries.shape
        _, Hkv, Lkv, _ = packed_key_indices.shape
        mask = (1 << bits) - 1

        # Unpack all key indices
        # packed shape: (B, Hkv, Lkv, words_per_vec)
        # We need (B, Hkv, Lkv, D)
        from ..packing import unpack_indices
        key_indices = unpack_indices(packed_key_indices, bits, D)
        # key_indices: (B, Hkv, Lkv, D) uint8

        # Lookup centroids: (B, Hkv, Lkv, D)
        key_values = key_centroids[key_indices]

        # GQA: expand KV heads
        repeats = Hq // Hkv
        key_values = mx.repeat(key_values, repeats, axis=1)
        key_norms_rep = mx.repeat(key_norms, repeats, axis=1)

        # QK dot product
        # queries: (B, Hq, Lq, D)
        # keys:    (B, Hq, Lkv, D)
        scores = mx.matmul(
            rotated_queries,
            key_values.transpose(0, 1, 3, 2)
        )  # (B, Hq, Lq, Lkv)

        # Apply norms and scale
        scores = scores * key_norms_rep[:, :, None, :] * scale
        return scores

    def _reference_sv(
        self,
        attention_weights: Any,      # (B, Hq, Lq, Lkv)
        packed_value_indices: Any,   # (B, Hkv, Lkv, packed_dim)
        value_norms: Any,            # (B, Hkv, Lkv)
        value_centroids: Any,        # (n_centroids,)
        bits: int,
        values_per_word: int,
    ) -> Any:
        """Reference SV: unpack values, lookup centroids, weighted sum."""
        if mx is None:
            raise RuntimeError("MLX is not installed")

        B, Hq, Lq, Lkv = attention_weights.shape
        _, Hkv, _, _ = packed_value_indices.shape

        from ..packing import unpack_indices
        # Determine head_dim from packed shape and bits
        # packed_value_indices shape: (B, Hkv, Lkv, words_per_vec)
        values_per_word = {2: 16, 3: 10, 4: 8}[bits]
        words_per_vec = packed_value_indices.shape[-1]
        head_dim = words_per_vec * values_per_word  # may include padding
        value_indices = unpack_indices(packed_value_indices, bits, head_dim)
        # value_indices: (B, Hkv, Lkv, head_dim) uint8

        # Lookup centroids
        D = value_indices.shape[-1]
        value_values = value_centroids[value_indices]

        # GQA: expand
        repeats = Hq // Hkv
        value_values = mx.repeat(value_values, repeats, axis=1)
        value_norms_rep = mx.repeat(value_norms, repeats, axis=1)

        # Weighted sum: (B, Hq, Lq, Lkv) @ (B, Hq, Lkv, D) → (B, Hq, Lq, D)
        weighted = attention_weights[:, :, :, :, None] * value_values[:, :, None, :, :] * value_norms_rep[:, :, None, :, None]
        output = mx.sum(weighted, axis=3)
        return output

    # ------------------------------------------------------------------
    # Metal availability check
    # ------------------------------------------------------------------

    def _check_metal(self) -> bool:
        """Check if custom Metal kernel compilation is available."""
        if not HAS_MLX:
            return False
        # mlx.core.fast is the experimental JIT path; check presence
        try:
            import mlx.core.fast as fast  # type: ignore[import-not-found]
            return hasattr(fast, "metal_kernel")
        except Exception:
            return False


# Global singleton
_default_registry: KernelRegistry | None = None


def get_kernel_registry() -> KernelRegistry:
    """Return the process-global kernel registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = KernelRegistry()
    return _default_registry
