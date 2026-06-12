"""Bounded-memory blockwise reference attention.

Before Metal kernels, implement correct attention that:
  1. Decodes one packed K block.
  2. Computes scores for that block.
  3. Releases reconstructed K.
  4. Decodes one V block.
  5. Accumulates its weighted contribution.
  6. Releases reconstructed V.

The runtime never reconstructs full-context K or V simultaneously.
After this works, stable online softmax eliminates the full score vector too.
"""
from __future__ import annotations

from typing import Any

from rfsn_v10.compat import mx

from .cartesian_codec import CartesianCodec
from .contracts import AttentionScratch
from .incremental_layer_cache import QuantizedLayerCache


class BlockwiseReferenceAttention:
    """Reference attention that processes cache block-by-block.

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

        # ---- Phase A: compute softmax-normalised attention weights ----
        # We iterate over key blocks, accumulating scores in chunks.
        # For the reference path, we still materialise the full score
        # vector for simplicity.  The memory win is in K/V reconstruction.
        all_scores: list[Any] = []
        all_positions: list[int] = []

        # Sealed blocks
        for key_block in layer_cache.iter_key_blocks():
            # Decode one block
            k_flat = self.key_codec.decode(key_block)  # (block_tokens * Hkv * D,)
            block_tokens = key_block.token_count
            max_block_tokens = max(max_block_tokens, block_tokens)

            # Reshape to (B, Hkv, block_tokens, D)
            # The block stores flattened tokens; we need to know Hkv and D
            k_reshaped = k_flat.reshape(B, n_kv_heads, block_tokens, D)
            k_expanded = mx.repeat(k_reshaped, repeats, axis=1)  # (B, Hq, block_tokens, D)

            # QK dot product
            scores = mx.matmul(queries, k_expanded.transpose(0, 1, 3, 2)) * s
            all_scores.append(scores)
            all_positions.append(block_tokens)

        # Staging
        stage_k, stage_v, stage_n = layer_cache.get_staging()
        if stage_k is not None:
            stage_tokens = stage_n
            max_block_tokens = max(max_block_tokens, stage_tokens)
            # stage_k is (stage_tokens * Hkv, D); reshape
            # Need to handle the fact that stage_k shape may vary
            k_reshaped = stage_k.reshape(B, n_kv_heads, stage_tokens, D)
            k_expanded = mx.repeat(k_reshaped, repeats, axis=1)
            scores = mx.matmul(queries, k_expanded.transpose(0, 1, 3, 2)) * s
            all_scores.append(scores)
            all_positions.append(stage_tokens)

        # Dense residual
        dense_k, dense_v = layer_cache.get_dense_residual()
        if dense_k is not None:
            dense_tokens = dense_k.shape[2]
            max_block_tokens = max(max_block_tokens, dense_tokens)
            k_expanded = mx.repeat(dense_k, repeats, axis=1)
            scores = mx.matmul(queries, k_expanded.transpose(0, 1, 3, 2)) * s
            all_scores.append(scores)
            all_positions.append(dense_tokens)

        # Concatenate scores and apply softmax
        if len(all_scores) == 1:
            full_scores = all_scores[0]
        else:
            full_scores = mx.concatenate(all_scores, axis=-1)

        if mask is not None:
            full_scores = full_scores + mask

        weights = mx.softmax(full_scores.astype(mx.float32), axis=-1).astype(queries.dtype)

        # ---- Phase B: accumulate weighted values blockwise ----
        output = mx.zeros((B, Hq, Lq, D), dtype=queries.dtype)
        offset = 0

        # Sealed value blocks
        for value_block in layer_cache.iter_value_blocks():
            block_tokens = value_block.token_count
            v_flat = self.value_codec.decode(value_block)
            v_reshaped = v_flat.reshape(B, n_kv_heads, block_tokens, D)
            v_expanded = mx.repeat(v_reshaped, repeats, axis=1)

            w = weights[..., offset:offset + block_tokens]
            # (B, Hq, Lq, T) @ (B, Hq, T, D) → (B, Hq, Lq, D)
            contrib = mx.matmul(w, v_expanded)
            output = output + contrib
            offset += block_tokens

        # Staging values
        if stage_v is not None:
            stage_tokens = stage_n
            v_reshaped = stage_v.reshape(B, n_kv_heads, stage_tokens, D)
            v_expanded = mx.repeat(v_reshaped, repeats, axis=1)
            w = weights[..., offset:offset + stage_tokens]
            contrib = mx.matmul(w, v_expanded)
            output = output + contrib
            offset += stage_tokens

        # Dense residual values
        if dense_v is not None:
            dense_tokens = dense_v.shape[2]
            v_expanded = mx.repeat(dense_v, repeats, axis=1)
            w = weights[..., offset:offset + dense_tokens]
            contrib = mx.matmul(w, v_expanded)
            output = output + contrib
            offset += dense_tokens

        scratch = AttentionScratch(
            max_reconstructed_block_tokens=max_block_tokens,
            score_vector_bytes=int(weights.size) * 4,
            output_accumulator_bytes=int(output.size) * 4,
        )
        return output, scratch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_kv_heads(self, layer_cache: QuantizedLayerCache) -> int:
        """Infer n_kv_heads from the first available block."""
        # Try dense residual first (has shape info)
        dense_k, _ = layer_cache.get_dense_residual()
        if dense_k is not None:
            return dense_k.shape[1]
        # Fallback: we need the caller to provide this
        # For now, assume GQA with 2 kv heads (common for small models)
        # In production, the adapter passes this explicitly.
        return 2
