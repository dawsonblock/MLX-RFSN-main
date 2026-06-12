"""Bounded-memory blockwise reference attention with online softmax.

Before Metal kernels, implement correct attention that:
  1. Decodes one packed K block.
  2. Computes scores for that block.
  3. Releases reconstructed K.
  4. Decodes one V block.
  5. Accumulates its weighted contribution.
  6. Releases reconstructed V.

The runtime never reconstructs full-context K or V simultaneously.
Online softmax eliminates the full score vector too.
"""
from __future__ import annotations

from typing import Any

from rfsn_v10.compat import mx

from .cartesian_codec import CartesianCodec
from .contracts import AttentionScratch
from .incremental_layer_cache import QuantizedLayerCache


class BlockwiseReferenceAttention:
    """Reference attention that processes cache block-by-block.

    Uses online softmax so the full score vector (B, Hq, Lq, T) is never
    materialised.  Only one block of K and one block of V exist in
    dense form at any moment.

    Parameters
    ----------
    key_codec
        Codec for decoding key blocks.
    value_codec
        Codec for decoding value blocks.
    scale
        Attention scale (typically head_dim ** -0.5).
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

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def attend(
        self,
        queries: Any,  # (B, Hq, Lq, D)
        layer_cache: QuantizedLayerCache,
        mask: Any | None = None,
    ) -> tuple[Any, AttentionScratch]:
        """Compute attention from the layer cache blockwise.

        Returns
        -------
        output
            Attention output tensor (B, Hq, Lq, D).
        scratch
            Scratch-memory accounting.
        """
        B, Hq, Lq, D = queries.shape
        s = self.scale if self.scale is not None else (D ** -0.5)

        # GQA repeats
        n_kv_heads = self._infer_kv_heads(layer_cache)
        repeats = Hq // n_kv_heads

        max_block_tokens = 0
        position_offset = 0

        # ------------------------------------------------------------------
        # Online softmax state
        # ------------------------------------------------------------------
        # m: running maximum of scores, shape (B, Hq, Lq, 1)
        # sum_exp: running sum of exp(scores - m), shape (B, Hq, Lq, 1)
        # out: running weighted sum of values, shape (B, Hq, Lq, D)
        m = mx.full((B, Hq, Lq, 1), -1e9, dtype=mx.float32)
        sum_exp = mx.zeros((B, Hq, Lq, 1), dtype=mx.float32)
        out = mx.zeros((B, Hq, Lq, D), dtype=mx.float32)

        # ------------------------------------------------------------------
        # Process sealed blocks (K and V interleaved)
        # ------------------------------------------------------------------
        key_blocks = list(layer_cache.iter_key_blocks())
        value_blocks = list(layer_cache.iter_value_blocks())

        for key_block, value_block in zip(key_blocks, value_blocks):
            block_tokens = key_block.token_count
            max_block_tokens = max(max_block_tokens, block_tokens)

            # ---- Decode K block, compute scores ----
            k_flat = self.key_codec.decode(key_block)
            k_reshaped = k_flat.reshape(B, n_kv_heads, block_tokens, D)
            k_expanded = mx.repeat(k_reshaped, repeats, axis=1)

            scores = mx.matmul(
                queries, k_expanded.transpose(0, 1, 3, 2)
            ) * s  # (B, Hq, Lq, block_tokens)

            if mask is not None:
                scores = (
                    scores
                    + mask[..., position_offset:position_offset + block_tokens]
                )

            # ---- Online softmax update ----
            block_max = mx.max(scores, axis=-1, keepdims=True)
            m_new = mx.maximum(m, block_max)

            exp_diff_m = mx.exp(m - m_new)
            sum_exp = sum_exp * exp_diff_m
            out = out * exp_diff_m

            exp_scores = mx.exp(scores.astype(mx.float32) - m_new)
            sum_exp = sum_exp + mx.sum(exp_scores, axis=-1, keepdims=True)

            # ---- Decode V block, accumulate weighted contribution ----
            v_flat = self.value_codec.decode(value_block)
            v_reshaped = v_flat.reshape(B, n_kv_heads, block_tokens, D)
            v_expanded = mx.repeat(v_reshaped, repeats, axis=1)

            block_contrib = mx.matmul(exp_scores, v_expanded)
            out = out + block_contrib

            m = m_new
            position_offset += block_tokens

        # ------------------------------------------------------------------
        # Process staging (if any)
        # ------------------------------------------------------------------
        stage_k, stage_v, stage_n = layer_cache.get_staging()
        if stage_k is not None:
            max_block_tokens = max(max_block_tokens, stage_n)
            k_reshaped = stage_k.reshape(B, n_kv_heads, stage_n, D)
            k_expanded = mx.repeat(k_reshaped, repeats, axis=1)
            scores = mx.matmul(queries, k_expanded.transpose(0, 1, 3, 2)) * s

            if mask is not None:
                scores = (
                    scores
                    + mask[..., position_offset:position_offset + stage_n]
                )

            block_max = mx.max(scores, axis=-1, keepdims=True)
            m_new = mx.maximum(m, block_max)

            exp_diff_m = mx.exp(m - m_new)
            sum_exp = sum_exp * exp_diff_m
            out = out * exp_diff_m

            exp_scores = mx.exp(scores.astype(mx.float32) - m_new)
            sum_exp = sum_exp + mx.sum(exp_scores, axis=-1, keepdims=True)

            v_reshaped = stage_v.reshape(B, n_kv_heads, stage_n, D)
            v_expanded = mx.repeat(v_reshaped, repeats, axis=1)
            block_contrib = mx.matmul(exp_scores, v_expanded)
            out = out + block_contrib

            m = m_new
            position_offset += stage_n

        # ------------------------------------------------------------------
        # Process dense residual (if any)
        # ------------------------------------------------------------------
        dense_k, dense_v = layer_cache.get_dense_residual()
        if dense_k is not None:
            dense_tokens = dense_k.shape[2]
            max_block_tokens = max(max_block_tokens, dense_tokens)
            k_expanded = mx.repeat(dense_k, repeats, axis=1)
            scores = mx.matmul(queries, k_expanded.transpose(0, 1, 3, 2)) * s

            if mask is not None:
                scores = (
                    scores
                    + mask[..., position_offset:position_offset + dense_tokens]
                )

            block_max = mx.max(scores, axis=-1, keepdims=True)
            m_new = mx.maximum(m, block_max)

            exp_diff_m = mx.exp(m - m_new)
            sum_exp = sum_exp * exp_diff_m
            out = out * exp_diff_m

            exp_scores = mx.exp(scores.astype(mx.float32) - m_new)
            sum_exp = sum_exp + mx.sum(exp_scores, axis=-1, keepdims=True)

            v_expanded = mx.repeat(dense_v, repeats, axis=1)
            block_contrib = mx.matmul(exp_scores, v_expanded)
            out = out + block_contrib

            m = m_new
            position_offset += dense_tokens

        # ------------------------------------------------------------------
        # Final normalisation
        # ------------------------------------------------------------------
        output = out / sum_exp  # sum_exp has shape (B, Hq, Lq, 1)
        output = output.astype(queries.dtype)

        scratch = AttentionScratch(
            max_reconstructed_block_tokens=max_block_tokens,
            score_vector_bytes=0,  # online softmax: no full score vector!
            output_accumulator_bytes=int(out.size) * 4,
        )
        return output, scratch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_kv_heads(self, layer_cache: QuantizedLayerCache) -> int:
        """Infer n_kv_heads from the first available block."""
        dense_k, _ = layer_cache.get_dense_residual()
        if dense_k is not None:
            return dense_k.shape[1]
        return 2
