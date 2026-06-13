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

import numpy as np

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
        with open(filepath, encoding="utf-8") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Cartesian QK kernel
# ---------------------------------------------------------------------------

def cartesian_qk_metal(
    queries: Any,          # (B, Hq, Lq, D)
    packed_codes: Any,     # (B, Hkv, Lkv, words_per_vec)
    scales: Any,           # (B, Hkv, Lkv, n_groups) — per-token scales
    bits: int,
    group_size: int,
    scale_factor: float,
    use_wht: bool = False,
    sign_seed: int = 0,
    layer_id: int = 0,
    stream_id: str = "",
) -> Any:
    """Compute QK scores via Metal kernel.

    The shader loops over all query positions inside each thread, so it
    is valid for any ``Lq``.  For decode ``Lq == 1`` this is a single
    iteration; for prefill ``Lq > 1`` each thread computes all query
    positions for its assigned KV token.

    ``scales`` must have shape ``(B, Hkv, Lkv, n_groups)`` — one scale
    group per KV token.  The caller is responsible for concatenating
    per-block scales into this flat layout.

    When *use_wht* or *sign_seed* is set, the kernel falls back to the
    CPU reference path because the Metal shader does not yet implement
    inverse WHT / hash-sign transforms.
    """
    if use_wht or sign_seed != 0:
        # Metal shader does not yet implement inverse WHT / hash signs.
        from rfsn_v10.kernels.cartesian_cpu_reference import cartesian_qk_cpu_reference
        return mx.array(
            cartesian_qk_cpu_reference(
                np.array(queries),
                np.array(packed_codes),
                np.array(scales),
                bits=bits,
                group_size=group_size,
                scale_factor=scale_factor,
                use_wht=use_wht,
                sign_seed=sign_seed,
                layer_id=layer_id,
                stream_id=stream_id,
            )
        )

    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")

    B, Hq, Lq, D = queries.shape
    _, Hkv, Lkv, _ = packed_codes.shape

    source = _load_metal_source("cartesian_qk_body.metal")

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
    scales: Any,           # (B, Hkv, Lkv, n_groups) — per-token scales
    bits: int,
    group_size: int,
    head_dim: int,
    use_wht: bool = False,
    sign_seed: int = 0,
    layer_id: int = 0,
    stream_id: str = "",
) -> Any:
    """Compute weighted value sum via Metal kernel.

    ``scales`` must have shape ``(B, Hkv, Lkv, n_groups)`` — one scale
    group per KV token.  The caller is responsible for concatenating
    per-block scales into this flat layout.

    When *use_wht* or *sign_seed* is set, the kernel falls back to the
    CPU reference path because the Metal shader does not yet implement
    inverse WHT / hash-sign transforms.
    """
    if use_wht or sign_seed != 0:
        from rfsn_v10.kernels.cartesian_cpu_reference import cartesian_sv_cpu_reference
        return mx.array(
            cartesian_sv_cpu_reference(
                np.array(weights),
                np.array(packed_codes),
                np.array(scales),
                bits=bits,
                group_size=group_size,
                head_dim=head_dim,
                use_wht=use_wht,
                sign_seed=sign_seed,
                layer_id=layer_id,
                stream_id=stream_id,
            )
        )

    if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
        raise KernelRouteError("metal_kernel_api_unavailable")

    B, Hq, Lq, Lkv = weights.shape
    _, Hkv, _, _ = packed_codes.shape
    D = head_dim

    source = _load_metal_source("cartesian_sv_body.metal")

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
