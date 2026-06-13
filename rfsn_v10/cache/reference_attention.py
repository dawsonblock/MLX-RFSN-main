"""Bounded-memory blockwise reference attention with online softmax.

This module now delegates to :func:`rfsn_v10.cache.mlx_packed_attention_reference.attend`
to avoid maintaining duplicate online-softmax logic.
"""
from __future__ import annotations

from typing import Any

from .cartesian_codec import CartesianCodec
from .contracts import AttentionScratch
from .incremental_layer_cache import QuantizedLayerCache
from .mlx_packed_attention_reference import attend


class BlockwiseReferenceAttention:
    """Reference attention that processes cache block-by-block.

    Thin wrapper around :func:`attend` so that callers using this class
    automatically receive the canonical numerically-safe implementation.
    """

    def __init__(
        self,
        key_codec: CartesianCodec,
        value_codec: CartesianCodec,
        scale: float | None = None,
    ) -> None:
        self.key_codec = key_codec
        self.value_codec = value_codec
        self.scale = scale

    def attend(
        self,
        queries: Any,  # (B, Hq, Lq, D)
        layer_cache: QuantizedLayerCache,
        mask: Any | None = None,
    ) -> tuple[Any, AttentionScratch]:
        """Compute attention from the layer cache blockwise.

        Delegates to the canonical :func:`attend` implementation.
        """
        return attend(
            queries,
            layer_cache,
            scale=self.scale,
            mask=mask,
            query_start_pos=layer_cache.total_token_count(),
            causal=True,
        )
