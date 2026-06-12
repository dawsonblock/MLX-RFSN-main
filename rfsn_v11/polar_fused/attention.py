"""Naive reference attention using dequantized Polar cache.

This is the correctness oracle for every Metal kernel.  It is pure Python
+ MLX, slow, but mathematically exact relative to the quantization contract.
"""
from __future__ import annotations

from typing import Any

from .codebooks import get_default_codebook_registry
from .contracts import AttentionOutputResult, QuantizedVectors
from .quantize import PolarQuantizer
from .rotations import get_default_rotation_registry

# MLX optional at import time
try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class NaivePolarAttention:
    """Reference attention that dequantizes the full cache each step.

    Pipeline:
        queries
          → key-basis rotation
          → QK with dequantized keys
          → mask + softmax
          → SV with dequantized values
          → value inverse rotation
          → output
    """

    def __init__(
        self,
        key_quantizer: PolarQuantizer,
        value_quantizer: PolarQuantizer,
        scale: float | None = None,
    ) -> None:
        if mx is None:
            raise RuntimeError("MLX is not installed")
        self.key_q = key_quantizer
        self.value_q = value_quantizer
        self.scale = scale

        # Rotation matrices
        self.Rk_T = get_default_rotation_registry().get_transpose(
            key_quantizer.head_dim, key_quantizer.rotation_seed
        )
        self.Rk = get_default_rotation_registry().get(
            key_quantizer.head_dim, key_quantizer.rotation_seed
        )
        self.Rv_T = get_default_rotation_registry().get_transpose(
            value_quantizer.head_dim, value_quantizer.rotation_seed
        )
        self.Rv = get_default_rotation_registry().get(
            value_quantizer.head_dim, value_quantizer.rotation_seed
        )

    def attend(
        self,
        queries: Any,
        key_qv: QuantizedVectors,
        value_qv: QuantizedVectors,
        mask: Any | None = None,
    ) -> AttentionOutputResult:
        """Compute attention from quantized K/V and raw queries.

        Parameters
        ----------
        queries
            Shape ``(batch, n_q_heads, 1, head_dim)`` for decode step.
        key_qv, value_qv
            Quantized cache for all prior tokens.
        mask
            Optional causal mask.

        Returns
        -------
        AttentionOutputResult
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")

        # Dequantize full cache
        keys = self.key_q.dequantize(key_qv)   # (B, H_kv, L, D)
        values = self.value_q.dequantize(value_qv)  # (B, H_kv, L, D)

        # GQA: map query heads to KV heads
        n_q_heads = queries.shape[1]
        n_kv_heads = keys.shape[1]
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
            )
        repeats = n_q_heads // n_kv_heads

        # Expand KV to match query heads
        keys = mx.repeat(keys, repeats, axis=1)
        values = mx.repeat(values, repeats, axis=1)

        # Rotate queries into key basis
        q_rot = queries @ self.Rk_T  # (B, Hq, 1, D)

        # QK
        # scores[b, h, q_pos, k_pos] = q_rot[b, h, q_pos, :] · keys[b, h, k_pos, :]
        scores = mx.matmul(q_rot, keys.transpose(0, 1, 3, 2))

        # Scale
        head_dim = queries.shape[-1]
        scale = self.scale if self.scale is not None else (head_dim ** -0.5)
        scores = scores * scale

        # Mask
        if mask is not None:
            scores = scores + mask

        # Softmax
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)

        # SV in value basis
        out_rot = mx.matmul(weights, values)  # (B, Hq, 1, D)

        # Inverse rotation back to original basis
        output = out_rot @ self.Rv

        return AttentionOutputResult(
            output=output,
            backend="naive_polar",
            metrics={},
        )
