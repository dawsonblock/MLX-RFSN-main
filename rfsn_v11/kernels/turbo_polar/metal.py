"""TurboPolar Metal kernel dispatcher with Python/MLX fallback.

Phases:
  6 — fused_dequant_qk: reconstruct Polar K tile → QK scores
  7 — qjl_correction inside the score kernel
  8 — online_attention_dense_v: block-wise softmax + dense V accumulation

All functions return (scores, used_metal) so callers can track fallback.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Tuple

import mlx.core as mx
import numpy as np

from rfsn_v11.quant.polar.payload import PolarKeyBlock
from rfsn_v11.quant.qjl.encoder import QJLPayload
from rfsn_v11.quant.qjl.score_estimate import qjl_dot_estimate

_KernelResult = Tuple[mx.array, bool]


def _has_metal_kernels() -> bool:
    try:
        return hasattr(mx.fast, "metal_kernel")
    except Exception:
        return False


def _read_metal_source(name: str) -> str:
    """Read a .metal file from the same directory."""
    here = Path(__file__).parent
    path = here / f"{name}.metal"
    if path.exists():
        return path.read_text()
    return ""


# ---------------------------------------------------------------------------
# Phase 6 — Fused dequant-QK (Python reference)
# ---------------------------------------------------------------------------

def fused_dequant_qk_python(
    q: mx.array,
    polar_block: PolarKeyBlock,
) -> mx.array:
    """Python/MLX reference: reconstruct K block and compute Q @ K.T.

    Args:
        q: (D,) float query vector.
        polar_block: PolarKeyBlock with K tile.

    Returns:
        scores: (block_size,) float array.
    """
    from rfsn_v11.quant.polar.decoder import PolarQuantDecoder

    decoder = PolarQuantDecoder(
        head_dim=polar_block.head_dim,
        use_rotation=polar_block.metadata.get("use_rotation", True),
        rotation_seed=polar_block.metadata.get("rotation_seed", 42),
    )
    k_recon = decoder.decode(polar_block)  # (..., block_size, D)
    k_flat = k_recon.reshape(-1, polar_block.head_dim)
    return q @ k_flat.T


# ---------------------------------------------------------------------------
# Phase 6 — Fused dequant-QK (Metal kernel)
# ---------------------------------------------------------------------------

def fused_dequant_qk_metal(
    q: mx.array,
    polar_block: PolarKeyBlock,
) -> mx.array | None:
    """Metal kernel path for dequant-QK.

    Returns None if Metal is unavailable or the kernel fails to compile.
    """
    if not _has_metal_kernels():
        return None

    try:
        source = _read_metal_source("tqpolar_fused_qk")
        if not source:
            # Fallback inline source
            source = """
                uint tid = thread_position_in_grid.x;
                if (tid >= block_size_buf[0]) { return; }
                int head_dim = head_dim_buf[0];
                int angle_bits = angle_bits_buf[0];
                int pairs = head_dim / 2;
                float bin_width = 6.28318530718f / float(1 << angle_bits);
                float half_bin = bin_width * 0.5f;
                float score = 0.0f;
                for (int p = 0; p < pairs; ++p) {
                    int idx = int(tid) * pairs + p;
                    float radius = float(radii[idx]);
                    uint code = uint(angle_codes[idx]);
                    float angle = float(code) * bin_width + half_bin;
                    float x = radius * cos(angle);
                    float y = radius * sin(angle);
                    score += q[p * 2] * x;
                    score += q[p * 2 + 1] * y;
                }
                scores[tid] = score;
            """

        kernel = mx.fast.metal_kernel(
            name="tqpolar_fused_dequant_qk",
            input_names=["q", "radii", "angle_codes", "block_size_buf", "head_dim_buf", "angle_bits_buf"],
            output_names=["scores"],
            source=source,
        )

        block_size = polar_block.radii.shape[-2] if polar_block.radii.ndim >= 2 else polar_block.radii.shape[0]
        pairs = polar_block.head_dim // 2

        # Flatten radii and angle_codes to 1D
        radii_flat = polar_block.radii.reshape(-1)
        codes_flat = polar_block.angle_codes_l1.reshape(-1)

        outputs = kernel(
            inputs=[
                q.astype(mx.float32),
                radii_flat,
                codes_flat,
                mx.array([block_size], dtype=mx.int32),
                mx.array([polar_block.head_dim], dtype=mx.int32),
                mx.array([polar_block.angle_bits_level1], dtype=mx.int32),
            ],
            template=[("T", mx.float32)],
            grid=(block_size, 1, 1),
            threadgroup=(min(block_size, 256), 1, 1),
            output_shapes=[(block_size,)],
            output_dtypes=[mx.float32],
        )
        return outputs[0]
    except Exception:
        return None


def fused_dequant_qk(
    q: mx.array,
    polar_block: PolarKeyBlock,
    force_metal: bool = False,
) -> _KernelResult:
    """Dispatch to Metal if possible and requested, else Python reference.

    Metal is OFF by default because the TurboPolar Metal kernels are
    EXPERIMENTAL and may trigger runtime issues on some MLX versions.
    Set force_metal=True or TURBOPOLAR_FORCE_METAL=1 to attempt Metal.

    Returns:
        (scores, used_metal)
    """
    import os
    if not force_metal and os.getenv("TURBOPOLAR_FORCE_METAL", "0") != "1":
        return fused_dequant_qk_python(q, polar_block), False
    result = fused_dequant_qk_metal(q, polar_block)
    if result is not None:
        return result, True
    return fused_dequant_qk_python(q, polar_block), False


# ---------------------------------------------------------------------------
# Phase 7 — QJL-corrected fused dequant-QK
# ---------------------------------------------------------------------------

def fused_dequant_qk_qjl(
    q: mx.array,
    polar_block: PolarKeyBlock,
    qjl_payload: QJLPayload | None,
) -> _KernelResult:
    """Compute QK scores with optional QJL residual correction.

    Args:
        q: (D,) query.
        polar_block: compressed K block.
        qjl_payload: QJL sketch of residual, or None.

    Returns:
        (scores, used_metal)
    """
    scores, used_metal = fused_dequant_qk(q, polar_block)
    if qjl_payload is not None:
        correction = qjl_dot_estimate(q, qjl_payload)
        scores = scores + correction
    return scores, used_metal


# ---------------------------------------------------------------------------
# Phase 8 — Online softmax attention with dense V (Python reference)
# ---------------------------------------------------------------------------

def online_attention_dense_v_python(
    q: mx.array,
    polar_blocks: list[PolarKeyBlock],
    value_blocks: list[mx.array],
    qjl_payloads: list[QJLPayload | None] | None = None,
) -> mx.array:
    """Python reference: online softmax attention over PolarQuant K + dense V.

    Args:
        q: (D,) query.
        polar_blocks: list of PolarKeyBlock, one per cache block.
        value_blocks: list of dense V blocks, same length.
        qjl_payloads: optional QJL sketches per block.

    Returns:
        attention output: (D,) float array.
    """
    if qjl_payloads is None:
        qjl_payloads = [None] * len(polar_blocks)

    m = -float("inf")
    l = 0.0
    acc = mx.zeros(q.shape, dtype=mx.float32)

    for polar_block, v_block, qjl in zip(polar_blocks, value_blocks, qjl_payloads):
        scores, _ = fused_dequant_qk(q, polar_block)
        if qjl is not None:
            scores = scores + qjl_dot_estimate(q, qjl)

        block_max = float(mx.max(scores).item())
        m_new = max(m, block_max)

        # Numerical stability: subtract m_new before exp
        p = mx.exp(scores - m_new)
        p_sum = float(mx.sum(p).item())

        alpha = math.exp(m - m_new) if m != -float("inf") else 0.0
        acc = acc * alpha + (p[:, None] * v_block.astype(mx.float32)).sum(axis=0)
        l = l * alpha + p_sum
        m = m_new

    return acc / l


# ---------------------------------------------------------------------------
# Phase 8 — Online softmax attention (Metal scaffolding)
# ---------------------------------------------------------------------------

def online_attention_dense_v_metal(
    q: mx.array,
    polar_blocks: list[PolarKeyBlock],
    value_blocks: list[mx.array],
    qjl_payloads: list[QJLPayload | None] | None = None,
) -> mx.array | None:
    """Metal scaffolding for online attention.

    Currently returns None (not yet implemented) so the caller falls back
    to the Python reference. The .metal source file documents the target.
    """
    # TODO: Phase 8 Metal implementation requires workgroup-level
    # reduction for online softmax across threads. This is left as
    # scaffolding until Phase 6/7 are fully validated.
    return None


def online_attention_dense_v(
    q: mx.array,
    polar_blocks: list[PolarKeyBlock],
    value_blocks: list[mx.array],
    qjl_payloads: list[QJLPayload | None] | None = None,
    force_metal: bool = False,
) -> Tuple[mx.array, bool]:
    """Dispatch to Metal if available and requested, else Python reference.

    Returns:
        (output, used_metal)
    """
    import os
    if not force_metal and os.getenv("TURBOPOLAR_FORCE_METAL", "0") != "1":
        return online_attention_dense_v_python(q, polar_blocks, value_blocks, qjl_payloads), False
    result = online_attention_dense_v_metal(q, polar_blocks, value_blocks, qjl_payloads)
    if result is not None:
        return result, True
    return online_attention_dense_v_python(q, polar_blocks, value_blocks, qjl_payloads), False
