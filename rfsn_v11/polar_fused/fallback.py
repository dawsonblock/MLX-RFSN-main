"""Explicit fallback to standard MLX attention.

This module provides a clean fallback path so unsupported layers or
configurations can degrade gracefully without global monkey-patching.
"""
from __future__ import annotations

from typing import Any

from .contracts import AttentionOutputResult

# MLX optional at import time
try:
    import mlx.core as mx
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]


class StandardMLXAttention:
    """Wrapper around standard MLX scaled_dot_product_attention."""

    def attend(
        self,
        queries: Any,
        keys: Any,
        values: Any,
        mask: Any | None = None,
        scale: float | None = None,
    ) -> AttentionOutputResult:
        """Compute standard dense attention.

        Parameters
        ----------
        queries, keys, values
            Standard 4D arrays ``(B, H, T, D)``.
        mask
            Optional attention mask.
        scale
            Optional scale factor.

        Returns
        -------
        AttentionOutputResult
        """
        if mx is None:
            raise RuntimeError("MLX is not installed")

        # scaled_dot_product_attention is in mlx.nn
        import mlx.nn as nn

        head_dim = queries.shape[-1]
        s = scale if scale is not None else (head_dim ** -0.5)
        output = nn.scaled_dot_product_attention(queries, keys, values, scale=s, mask=mask)

        return AttentionOutputResult(
            output=output,
            backend="mlx_standard",
            metrics={},
        )
