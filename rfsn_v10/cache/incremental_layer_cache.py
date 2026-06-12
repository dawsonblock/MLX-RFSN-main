"""Append-only per-layer quantized KV cache.

Architecture:
  * Immutable sealed packed blocks (never touched after creation)
  * One small staging block (mutable, accumulates new tokens)
  * Optional bounded dense residual window (for recent-token quality)
  * Token and encoding counters for proof

Append behaviour:
  1. Receive only new K/V tokens.
  2. Add them to staging.
  3. Encode a staging block once when full.
  4. Append the immutable block.
  5. Never recompress sealed history.
  6. Never concatenate the entire history.

Exit condition for Phase 3:
  encoded_token_count == 1024
  requantized_token_count == 0
  bytes_written grow linearly
  dense_storage stays bounded
"""
from __future__ import annotations

from typing import Any

from rfsn_v10.compat import mx

from .cartesian_codec import CartesianCodec
from .contracts import CacheStats, PackedBlock


class QuantizedLayerCache:
    """Per-layer cache that only appends, never recompresses.

    Parameters
    ----------
    key_codec
        CartesianCodec for keys (K8, group_size=64).
    value_codec
        CartesianCodec for values (V5, group_size=64).
    staging_capacity
        Number of tokens to accumulate before encoding a block.
    dense_residual_window
        Keep the last N tokens in dense FP16 (0 to disable).
    """

    def __init__(
        self,
        key_codec: CartesianCodec,
        value_codec: CartesianCodec,
        staging_capacity: int = 64,
        dense_residual_window: int = 0,
    ) -> None:
        self.key_codec = key_codec
        self.value_codec = value_codec
        self.staging_capacity = staging_capacity
        self.dense_residual_window = dense_residual_window

        # Immutable sealed blocks
        self._key_blocks: list[PackedBlock] = []
        self._value_blocks: list[PackedBlock] = []

        # Staging buffers (mutable)
        self._stage_keys: list[Any] = []
        self._stage_values: list[Any] = []
        self._stage_token_count: int = 0

        # Dense residual (optional, bounded)
        self._dense_keys: Any | None = None
        self._dense_values: Any | None = None
        self._dense_token_count: int = 0

        # Counters for proof
        self._encoded_tokens: int = 0
        self._requantized_tokens: int = 0

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, keys: Any, values: Any) -> None:
        """Append new K/V tokens.

        Parameters
        ----------
        keys, values
            Shape ``(batch, n_kv_heads, new_tokens, head_dim)``.
        """
        B, Hkv, new_T, D = keys.shape
        assert B == 1, "Batch size must be 1"

        # Flatten to (new_tokens * Hkv, D) for per-head-vector quantization
        # Each token produces one key vector and one value vector per head.
        # The codec flattens internally, so we just pass the raw tensor.
        new_keys = keys.reshape(-1, D)
        new_values = values.reshape(-1, D)

        # Add to staging
        self._stage_keys.append(new_keys)
        self._stage_values.append(new_values)
        self._stage_token_count += new_T

        # Dense residual: keep last N tokens in FP16
        if self.dense_residual_window > 0:
            self._update_dense_residual(keys, values)

        # Flush staging when capacity reached
        if self._stage_token_count >= self.staging_capacity:
            self._flush_staging()

    def _flush_staging(self) -> None:
        """Encode staged tokens into immutable blocks and clear staging."""
        if self._stage_token_count == 0:
            return

        # Concatenate staged pieces
        keys_flat = mx.concatenate(self._stage_keys, axis=0)
        values_flat = mx.concatenate(self._stage_values, axis=0)

        # Encode
        key_block_raw = self.key_codec.encode(keys_flat)
        value_block_raw = self.value_codec.encode(values_flat)

        # Override token_count to the actual number of tokens (not flattened elements)
        # keys_flat shape = (stage_tokens * Hkv, D), original_size = stage_tokens * Hkv * D
        # But token_count should be stage_tokens
        import dataclasses
        key_block = dataclasses.replace(key_block_raw, token_count=self._stage_token_count)
        value_block = dataclasses.replace(value_block_raw, token_count=self._stage_token_count)

        # Append immutable blocks
        self._key_blocks.append(key_block)
        self._value_blocks.append(value_block)

        # Update counters
        self._encoded_tokens += self._stage_token_count
        # Requantize count stays 0 — we never recompress sealed history

        # Clear staging
        self._stage_keys.clear()
        self._stage_values.clear()
        self._stage_token_count = 0

    def _update_dense_residual(self, keys: Any, values: Any) -> None:
        """Maintain a bounded dense FP16 window of the most recent tokens."""
        if self._dense_keys is None:
            self._dense_keys = keys
            self._dense_values = values
        else:
            self._dense_keys = mx.concatenate([self._dense_keys, keys], axis=2)
            self._dense_values = mx.concatenate([self._dense_values, values], axis=2)

        # Trim to window
        total_dense = self._dense_keys.shape[2]
        if total_dense > self.dense_residual_window:
            self._dense_keys = self._dense_keys[:, :, -self.dense_residual_window:, :]
            self._dense_values = self._dense_values[:, :, -self.dense_residual_window:, :]
            self._dense_token_count = self.dense_residual_window
        else:
            self._dense_token_count = total_dense

    # ------------------------------------------------------------------
    # Retrieval (for attention)
    # ------------------------------------------------------------------

    def iter_key_blocks(self):
        """Yield each sealed key block for blockwise attention."""
        for block in self._key_blocks:
            yield block

    def iter_value_blocks(self):
        """Yield each sealed value block for blockwise attention."""
        for block in self._value_blocks:
            yield block

    def get_dense_residual(self) -> tuple[Any | None, Any | None]:
        """Return the dense FP16 residual window, or (None, None)."""
        return self._dense_keys, self._dense_values

    def get_staging(self) -> tuple[Any | None, Any | None, int]:
        """Return staging keys, values, and token count."""
        if self._stage_token_count == 0:
            return None, None, 0
        keys = mx.concatenate(self._stage_keys, axis=0) if len(self._stage_keys) > 1 else self._stage_keys[0]
        values = mx.concatenate(self._stage_values, axis=0) if len(self._stage_values) > 1 else self._stage_values[0]
        return keys, values, self._stage_token_count

    # ------------------------------------------------------------------
    # Proof counters
    # ------------------------------------------------------------------

    @property
    def encoded_token_count(self) -> int:
        return self._encoded_tokens

    @property
    def requantized_token_count(self) -> int:
        return self._requantized_tokens

    def total_token_count(self) -> int:
        """Total tokens = encoded + staged + dense residual."""
        total = self._encoded_tokens + self._stage_token_count
        if self.dense_residual_window > 0:
            total += self._dense_token_count
        return total

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def payload_bytes(self) -> int:
        """Exact bytes from all sealed blocks (valid payload only)."""
        total = 0
        for kb, vb in zip(self._key_blocks, self._value_blocks):
            total += kb.payload_bytes()
            total += vb.payload_bytes()
        return total

    def dense_residual_bytes(self) -> int:
        """Bytes in the dense FP16 residual window."""
        if self._dense_keys is None:
            return 0
        return int(self._dense_keys.size) * 2 + int(self._dense_values.size) * 2

    def staging_bytes(self) -> int:
        """Bytes in staging buffers."""
        total = 0
        for k in self._stage_keys:
            total += int(k.size) * 4  # float32
        for v in self._stage_values:
            total += int(v.size) * 4
        return total

    def total_memory_bytes(self) -> int:
        """All accounted bytes: payload + dense + staging."""
        return self.payload_bytes() + self.dense_residual_bytes() + self.staging_bytes()

    def stats(self) -> CacheStats:
        return CacheStats(
            tokens_encoded=self._encoded_tokens,
            tokens_requantized=self._requantized_tokens,
            sealed_blocks=len(self._key_blocks),
            staged_tokens=self._stage_token_count,
            dense_residual_tokens=self._dense_token_count,
            payload_bytes=self.payload_bytes(),
        )

    def trim(self, new_token_count: int) -> None:
        """Trim cache to retain only first N tokens."""
        if new_token_count >= self.total_token_count():
            return
        if new_token_count <= 0:
            self.reset()
            return

        # Trim is coarse: if new count falls within staged/dense, reset
        # and re-append.  For sealed blocks, we keep full blocks until
        # the trim point and drop the rest.
        # Simplified: just reset if the trim is significant.
        # A proper implementation would slice into sealed blocks.
        if new_token_count < self._encoded_tokens:
            # Trim into sealed blocks — drop all sealed blocks after trim
            keep_blocks = 0
            cumulative = 0
            for kb in self._key_blocks:
                if cumulative + kb.token_count > new_token_count:
                    break
                cumulative += kb.token_count
                keep_blocks += 1

            self._key_blocks = self._key_blocks[:keep_blocks]
            self._value_blocks = self._value_blocks[:keep_blocks]
            self._encoded_tokens = cumulative

        # Drop staging and dense residual if they exceed the new count
        remaining = new_token_count - self._encoded_tokens
        if remaining <= 0:
            self._stage_keys.clear()
            self._stage_values.clear()
            self._stage_token_count = 0
            self._dense_keys = None
            self._dense_values = None
            self._dense_token_count = 0
        else:
            # Keep only remaining tokens in staging
            if self._stage_token_count > remaining:
                # This is approximate — proper trim would slice arrays
                self._flush_staging()
                # After flush, re-trim
                self.trim(new_token_count)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Destroy all state.  Called on session teardown."""
        self._key_blocks.clear()
        self._value_blocks.clear()
        self._stage_keys.clear()
        self._stage_values.clear()
        self._stage_token_count = 0
        self._dense_keys = None
        self._dense_values = None
        self._dense_token_count = 0
        self._encoded_tokens = 0
        self._requantized_tokens = 0
