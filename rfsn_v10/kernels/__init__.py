"""RFSN v10 — Portable kernel backend dispatcher.

Selects the compute backend via the ``RFSN_BACKEND`` environment variable
or ``default_runtime.yaml`` config.  Available backends:

* ``metal`` — MLX Metal kernels (fast, Apple Silicon only).
* ``numpy`` — Pure NumPy reference (slow, universal).
* ``cuda``  — CUDA/Triton stub (not yet implemented).
"""

from __future__ import annotations

import os
import warnings
from typing import Protocol

from ._common import KernelRouteError


class _Backend(Protocol):
    """Protocol every kernel backend must satisfy."""

    name: str

    def scaled_dot_product_attention(
        self,
        queries,
        keys,
        values,
        scale: float | None = None,
        causal: bool = False,
    ):
        """Compute attention.  Returns array-like object."""
        ...

    def pack_bits(self, codes, bits: int) -> tuple:
        """Pack integer codes into compact representation.

        Returns ``(packed_array, n_values)``.
        """
        ...

    def unpack_bits(self, packed, n_values: int, bits: int):
        """Unpack codes from compact representation."""
        ...

    def packed_dequant(
        self,
        packed,
        scales,
        n_values: int,
        bits: int,
        group_size: int = 64,
    ):
        """Dequantize packed symmetric codes."""
        ...

    def wht_transform(self, x):
        """Walsh-Hadamard transform over contiguous 64-value blocks."""
        ...

    def apply_hash_signs(self, x, seed: int):
        """Apply deterministic +/-1 signs."""
        ...

    def quantized_attention_decode(
        self,
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
        """Quantized attention for decode step."""
        ...

    def available(self) -> bool:
        """Return True if this backend can be instantiated."""
        ...


def _resolve_backend_name() -> str:
    """Read RFSN_BACKEND from env, then config, defaulting to 'metal'."""
    name = os.environ.get("RFSN_BACKEND", "").lower().strip()
    if name in ("metal", "numpy", "cuda"):
        return name

    # Try config file (lightweight, no heavy imports)
    try:
        import yaml

        cfg_path = os.environ.get(
            "RFSN_CONFIG", "configs/default_runtime.yaml"
        )
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            name = str(cfg.get("backend", "metal")).lower().strip()
            if name in ("metal", "numpy", "cuda"):
                return name
    except Exception:
        pass

    return "metal"


def _get_backend() -> _Backend:
    """Lazy singleton — first call imports the chosen backend module."""
    global _BACKEND_INSTANCE  # noqa: PLW0603
    if "_BACKEND_INSTANCE" not in globals():
        name = _resolve_backend_name()

        if name == "metal":
            from ._metal_backend import MetalBackend

            if MetalBackend.available():
                _BACKEND_INSTANCE = MetalBackend()  # type: ignore[misc]
            else:
                warnings.warn(
                    "Metal backend requested but MLX/metal_kernel is not "
                    "available.  Falling back to NumPy.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                from ._numpy_backend import NumpyBackend

                _BACKEND_INSTANCE = NumpyBackend()  # type: ignore[misc]

        elif name == "numpy":
            from ._numpy_backend import NumpyBackend

            _BACKEND_INSTANCE = NumpyBackend()  # type: ignore[misc]

        elif name == "cuda":
            from ._cuda_backend import CudaBackend

            _BACKEND_INSTANCE = CudaBackend()  # type: ignore[misc]

        else:
            raise KernelRouteError(f"Unknown backend {name!r}")

    return _BACKEND_INSTANCE  # type: ignore[return-value]


# Public API — forwards to active backend
backend: _Backend = _get_backend()  # type: ignore[assignment]


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    scale: float | None = None,
    causal: bool = False,
):
    return backend.scaled_dot_product_attention(
        queries, keys, values, scale=scale, causal=causal
    )


def pack_bits(codes, bits: int) -> tuple:
    return backend.pack_bits(codes, bits)


def unpack_bits(packed, n_values: int, bits: int):
    return backend.unpack_bits(packed, n_values, bits)


def packed_dequant(
    packed,
    scales,
    n_values: int,
    bits: int,
    group_size: int = 64,
):
    return backend.packed_dequant(
        packed, scales, n_values, bits, group_size=group_size
    )


def wht_transform(x):
    return backend.wht_transform(x)


def apply_hash_signs(x, seed: int):
    return backend.apply_hash_signs(x, seed)


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
    return backend.quantized_attention_decode(
        queries,
        packed_k,
        packed_v,
        scales_k,
        scales_v,
        n_keys,
        bits,
        group_size=group_size,
        scale=scale,
    )


def maybe_supports_metal_kernels() -> bool:
    """Convenience helper for feature detection."""
    from ._metal_backend import MetalBackend

    return MetalBackend.available()


# ---------------------------------------------------------------------------
# Backward-compatible re-exports — old MLX-specific helpers
# ---------------------------------------------------------------------------


def sequential_reference_route_supported(
    *,
    shape: tuple,
    out_dtype,
    use_wht: bool,
    use_incoherent_signs: bool,
) -> tuple[bool, str]:
    """Check whether the sequential reference route is supported."""
    from ..compat import mx

    if not use_wht:
        return False, "sequential_reference_requires_wht"
    if not use_incoherent_signs:
        return False, "sequential_reference_requires_incoherent_signs"
    if out_dtype not in (mx.float32, mx.float16):
        return False, "sequential_reference_out_dtype_unsupported"
    if len(shape) != 4:
        return False, "sequential_reference_shape_rank_unsupported"
    if shape[-1] % 64 != 0:
        return False, "sequential_reference_head_dim_unsupported"
    return True, "sequential_reference_supported"


def wht64_metal(x, out_dtype=None):
    """Backward-compatible wrapper — delegates to Metal backend."""
    from ._metal_backend import MetalBackend

    return MetalBackend.wht_transform(x)


def apply_hash_signs_metal(x, seed: int):
    """Backward-compatible wrapper — delegates to Metal backend."""
    from ._metal_backend import MetalBackend

    return MetalBackend.apply_hash_signs(x, seed)


def apply_hash_signs_with_indices_metal(x, indices, seed: int):
    """Backward-compatible wrapper (indices ignored in new API)."""
    from ._metal_backend import MetalBackend

    return MetalBackend.apply_hash_signs(x, seed)


def packed_dequant_metal(
    packed,
    scales,
    n_values: int,
    bits: int,
    group_size: int = 64,
    out_dtype=None,
):
    """Backward-compatible wrapper — delegates to Metal backend."""
    from ._metal_backend import MetalBackend

    return MetalBackend.packed_dequant(
        packed, scales, n_values, bits, group_size=group_size
    )


def packed_dequant_wht_sign_metal(
    packed,
    scales,
    n_values: int,
    bits: int,
    group_size: int = 64,
    seed: int = 0,
    out_dtype=None,
):
    """Backward-compatible fused dequant+WHT+signs wrapper.

    Since the new backend API separates dequant, WHT, and signs,
    this wrapper chains the three MetalBackend calls for compatibility.
    """
    from ._metal_backend import MetalBackend

    deq = MetalBackend.packed_dequant(
        packed, scales, n_values, bits, group_size=group_size
    )
    deq = MetalBackend.wht_transform(deq)
    return MetalBackend.apply_hash_signs(deq, seed)


def quantized_attention_decode_metal(
    queries,
    packed_k,
    packed_v,
    scales_k,
    scales_v,
    n_keys: int,
    bits: int,
    group_size: int = 64,
    scale: float | None = None,
    out_dtype=None,
):
    """Backward-compatible wrapper — delegates to Metal backend."""
    from ._metal_backend import MetalBackend

    return MetalBackend.quantized_attention_decode(
        queries,
        packed_k,
        packed_v,
        scales_k,
        scales_v,
        n_keys,
        bits,
        group_size=group_size,
        scale=scale,
    )
