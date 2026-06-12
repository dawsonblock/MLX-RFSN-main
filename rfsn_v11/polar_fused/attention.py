"""Naive reference attention using dequantized Polar cache.

This is the correctness oracle for every Metal kernel.  It is pure Python
+ MLX, slow, but mathematically exact relative to the quantization contract.

Important: ``PolarQuantizer.dequantize()`` already applies the inverse
rotation, returning vectors in the original basis.  Therefore this class
performs standard attention on the dequantized K/V without any extra
rotations.  The rotations are an implementation detail of the quantizer.
"""
from __future__ import annotations

from typing import Any

from .contracts import AttentionOutputResult, QuantizedVectors
from .quantize import PolarQuantizer

# MLX optional at import time
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


class NaivePolarAttention:
    """Reference attention that dequantizes the full cache each step.

    Pipeline:
        keys, values = dequantize(quantized_cache)
        scores = softmax(scale * queries @ keys.T)
        output = scores @ values

    No extra rotations are applied — ``dequantize()`` already returns
    vectors in the original (unrotated) basis.
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
            Shape ``(batch, n_q_heads, Lq, head_dim)``.
        key_qv, value_qv
            Quantized cache for all prior tokens.
        mask
            Optional attention mask.

        Returns
        -------
        AttentionOutputResult
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")

        # Dequantize full cache (already in original basis)
        keys = self.key_q.dequantize(key_qv)     # (B, H_kv, L, D)
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

        # Standard attention with dequantized K/V
        scores = mx.matmul(
            queries,
            keys.transpose(0, 1, 3, 2)
        )

        # Scale
        head_dim = queries.shape[-1]
        scale = self.scale if self.scale is not None else (head_dim ** -0.5)
        scores = scores * scale

        # Mask
        if mask is not None:
            scores = scores + mask

        # Softmax
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)

        # SV
        output = mx.matmul(weights, values)

        return AttentionOutputResult(
            output=output,
            backend="naive_polar",
            metrics={},
        )
