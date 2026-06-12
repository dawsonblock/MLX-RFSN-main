"""Metal kernel dispatch for Cartesian QK and SV.

Uses mlx.core.fast.metal_kernel to compile and dispatch the Cartesian
QK and SV shaders.  Falls back to reference blockwise attention if Metal
is unavailable or compilation fails.
"""
from __future__ import annotations

import functools
import importlib.resources
import os
from typing import Any

from rfsn_v10.compat import mx

from ._common import KernelRouteError


# ---------------------------------------------------------------------------
# Kernel source loading
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=2)
def _load_metal_source(filename: str) -> str:
    """Load a .metal shader file from the package."""
    pkg = "rfsn_v10.kernels.metal"
    try:
        files = importlib.resources.files(pkg)
        path = files / filename
        return path.read_text()
    except Exception:
        # Fallback for editable installs where package data may not be visible
        module_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(module_dir, "metal", filename)
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Cartesian QK kernel
# ---------------------------------------------------------------------------

def cartesian_qk_metal(
    queries: Any,          # (B, Hq, Lq, D)
    packed_codes: Any,     # (B, Hkv, Lkv, words_per_vec)
    scales: Any,           # (B, Hkv, n_groups)
    bits: int,
    group_size: int,
    scale_factor: float,
) -> Any:
    """Compute QK scores via Metal kernel."""
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")

    B, Hq, Lq, D = queries.shape
    _, Hkv, Lkv, _ = packed_codes.shape

    source = _load_metal_source("cartesian_qk.metal")

    kernel = mx.fast.metal_kernel(
        name="cartesian_qk",
        input_names=[
            "queries", "packed_codes", "scales",
            "bits_buf", "group_buf", "scale_buf",
            "b_buf", "hq_buf", "hkv_buf", "lq_buf", "lkv_buf", "d_buf",
        ],
        output_names=["scores"],
        source=source,
    )

    bits_buf = mx.array([bits], dtype=mx.int32)
    group_buf = mx.array([group_size], dtype=mx.int32)
    scale_buf = mx.array([scale_factor], dtype=mx.float32)
    b_buf = mx.array([B], dtype=mx.int32)
    hq_buf = mx.array([Hq], dtype=mx.int32)
    hkv_buf = mx.array([Hkv], dtype=mx.int32)
    lq_buf = mx.array([Lq], dtype=mx.int32)
    lkv_buf = mx.array([Lkv], dtype=mx.int32)
    d_buf = mx.array([D], dtype=mx.int32)

    grid = (Lkv, Hq, B)
    threadgroup = (min(256, Lkv), 1, 1)

    outputs = kernel(
        inputs=[
            queries.astype(mx.float32),
            packed_codes.astype(mx.uint32),
            scales.astype(mx.float32),
            bits_buf, group_buf, scale_buf,
            b_buf, hq_buf, hkv_buf, lq_buf, lkv_buf, d_buf,
        ],
        template=[],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, Hq, Lq, Lkv)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# Cartesian SV kernel
# ---------------------------------------------------------------------------

def cartesian_sv_metal(
    weights: Any,          # (B, Hq, Lq, Lkv)
    packed_codes: Any,     # (B, Hkv, Lkv, words_per_vec)
    scales: Any,           # (B, Hkv, n_groups)
    bits: int,
    group_size: int,
    head_dim: int,
) -> Any:
    """Compute weighted value sum via Metal kernel."""
    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")

    B, Hq, Lq, Lkv = weights.shape
    _, Hkv, _, _ = packed_codes.shape
    D = head_dim

    source = _load_metal_source("cartesian_sv.metal")

    kernel = mx.fast.metal_kernel(
        name="cartesian_sv",
        input_names=[
            "weights", "packed_codes", "scales",
            "bits_buf", "group_buf",
            "b_buf", "hq_buf", "hkv_buf", "lq_buf", "lkv_buf", "d_buf",
        ],
        output_names=["output"],
        source=source,
    )

    bits_buf = mx.array([bits], dtype=mx.int32)
    group_buf = mx.array([group_size], dtype=mx.int32)
    b_buf = mx.array([B], dtype=mx.int32)
    hq_buf = mx.array([Hq], dtype=mx.int32)
    hkv_buf = mx.array([Hkv], dtype=mx.int32)
    lq_buf = mx.array([Lq], dtype=mx.int32)
    lkv_buf = mx.array([Lkv], dtype=mx.int32)
    d_buf = mx.array([D], dtype=mx.int32)

    grid = (D, Lq, B * Hq)
    threadgroup = (min(256, D), 1, 1)

    outputs = kernel(
        inputs=[
            weights.astype(mx.float32),
            packed_codes.astype(mx.uint32),
            scales.astype(mx.float32),
            bits_buf, group_buf,
            b_buf, hq_buf, hkv_buf, lq_buf, lkv_buf, d_buf,
        ],
        template=[],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[(B, Hq, Lq, D)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


# ---------------------------------------------------------------------------
# Dispatch report
# ---------------------------------------------------------------------------

def dispatch_report(
    requested_backend: str,
    executed_backend: str,
    fallback_used: bool,
    kernel_name: str,
) -> dict[str, Any]:
    return {
        "requested_backend": requested_backend,
        "executed_backend": executed_backend,
        "fallback_used": fallback_used,
        "kernel_name": kernel_name,
        "metal_executed": executed_backend == "metal" and not fallback_used,
    }
