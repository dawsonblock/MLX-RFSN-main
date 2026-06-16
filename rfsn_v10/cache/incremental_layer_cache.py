"""Append-only per-layer quantized KV cache.

Architecture:
  * Immutable sealed packed blocks (never touched after creation)
  * One small staging block (mutable, accumulates new tokens)
  * Optional bounded dense residual window (for recent-token quality)
  * Token and encoding counters for proof

Append behaviour:
  1. Receive only new K/V tokens.
  2. Add them to staging (or dense residual, never both).
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
from .contracts import CacheStats, PackedBlock, validate_block_positions

# Phase 5: GPU-resident paged arena
from .paged_arena import (
    PagedKVArena,
    PagedKVView,
    validate_direct_packed_format,
)


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
        layer_id: int = 0,
        session: Any = None,
        use_paged_arena: bool = False,
        max_pages: int = 256,
    ) -> None:
        self.key_codec = key_codec
        self.value_codec = value_codec
        self.staging_capacity = staging_capacity
        self.dense_residual_window = dense_residual_window
        self.layer_id = layer_id
        self.session = session
        self._use_paged_arena = use_paged_arena
        self._max_pages = max_pages

        # Direct packed paging is only valid for K8/V8 GS64.
        if use_paged_arena:
            validate_direct_packed_format(
                key_codec, value_codec, label="QuantizedLayerCache"
            )

        # Immutable sealed blocks (fallback when paged arena is inactive)
        self._key_blocks: list[PackedBlock] = []
        self._value_blocks: list[PackedBlock] = []

        # Phase 5: combined GPU-resident KV arena (lazy-initialized on first
        # flush when packed geometry is known exactly)
        self._kv_arena: PagedKVArena | None = None

        # Staging buffers (mutable) — stored as full-shaped (B, Hkv, T, D) tensors
        self._stage_keys: list[Any] = []
        self._stage_values: list[Any] = []
        self._stage_token_count: int = 0

        # Dense residual (optional, bounded) — full-shaped (B, Hkv, N, D)
        self._dense_keys: Any | None = None
        self._dense_values: Any | None = None
        self._dense_token_count: int = 0

        # Geometry freeze (validated on first append)
        self._geometry: tuple[int, int, int] | None = None  # (B, Hkv, D)

        # Counters for proof
        self._encoded_tokens: int = 0
        self._requantized_tokens: int = 0

        # Lifecycle
        self._destroyed: bool = False

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def _check_destroyed(self) -> None:
        if getattr(self, "_destroyed", False):
            raise RuntimeError("cache has been destroyed")

    def append(self, keys: Any, values: Any) -> None:
        """Append new K/V tokens.

        Parameters
        ----------
        keys, values
            Shape ``(batch, n_kv_heads, new_tokens, head_dim)``.
        """
        self._check_destroyed()
        if getattr(keys, "ndim", None) != 4:
            raise ValueError("keys must have shape (B,H,T,D)")
        if getattr(values, "ndim", None) != 4:
            raise ValueError("values must have shape (B,H,T,D)")
        if tuple(keys.shape) != tuple(values.shape):
            raise ValueError("keys and values must have identical shapes")

        B, Hkv, new_T, D = map(int, keys.shape)
        if new_T <= 0:
            raise ValueError("new token count must be positive")
        if B != 1:
            raise ValueError("batch size 1 is required")
        if Hkv <= 0:
            raise ValueError("n_kv_heads must be positive")
        if D <= 0:
            raise ValueError("head_dim must be positive")

        # dtype validation
        key_dtype = str(keys.dtype).split(".")[-1]
        value_dtype = str(values.dtype).split(".")[-1]
        _SUPPORTED_DTYPES = {"float16", "bfloat16", "float32"}
        if key_dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"unsupported key dtype: {key_dtype}")
        if value_dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"unsupported value dtype: {value_dtype}")

        # finite-value validation
        if not bool(mx.all(mx.isfinite(keys)).item()):
            raise ValueError("keys contain NaN or Inf")
        if not bool(mx.all(mx.isfinite(values)).item()):
            raise ValueError("values contain NaN or Inf")

        if self._geometry is None:
            # Codec geometry compatibility
            if D % self.key_codec.group_size != 0:
                raise ValueError(
                    f"head_dim {D} incompatible with key group_size {self.key_codec.group_size}"
                )
            if D % self.value_codec.group_size != 0:
                raise ValueError(
                    f"head_dim {D} incompatible with value group_size {self.value_codec.group_size}"
                )
            self._geometry = (B, Hkv, D)
        else:
            expected_B, expected_Hkv, expected_D = self._geometry
            if (B, Hkv, D) != (expected_B, expected_Hkv, expected_D):
                raise ValueError(
                    f"Geometry mismatch: expected {(expected_B, expected_Hkv, expected_D)}, "
                    f"got {(B, Hkv, D)}"
                )

        if self.dense_residual_window > 0:
            # Recent tokens go to dense residual; evicted tokens are staged.
            evicted_k, evicted_v = self._update_dense_residual(keys, values)
            if evicted_k is not None:
                self._add_to_staging(evicted_k, evicted_v)
        else:
            self._add_to_staging(keys, values)

        # Fix #2: Use typed method instead of string-based increment
        if self.session:
            self.session.runtime_counters.record_token_appended(new_T)

    def _add_to_staging(self, keys: Any, values: Any) -> None:
        """Add full-shaped tensors to staging."""
        self._stage_keys.append(keys)
        self._stage_values.append(values)
        self._stage_token_count += keys.shape[2]

        # Flush staging when capacity reached
        if self._stage_token_count >= self.staging_capacity:
            self._flush_staging()

    def _flush_staging(self) -> None:
        """Encode staged tokens into fixed-size immutable blocks.

        Staging is chunked into blocks of at most ``staging_capacity`` tokens.
        Any remainder stays in staging for the next append.
        """
        if self._stage_token_count == 0:
            return

        # Concatenate full-shaped staging tensors along token axis (2).
        keys_full = mx.concatenate(self._stage_keys, axis=2)
        values_full = mx.concatenate(self._stage_values, axis=2)

        B, Hkv, stage_T, D = keys_full.shape
        assert B == 1

        block_size = self.staging_capacity
        n_full_blocks = stage_T // block_size
        remainder = stage_T % block_size


        base_offset = self._encoded_tokens
        for i in range(n_full_blocks):
            start = i * block_size
            end = start + block_size
            keys_slice = keys_full[:, :, start:end, :]
            values_slice = values_full[:, :, start:end, :]

            logical_start = base_offset + start
            key_block = self.key_codec.encode_bhtd(
                keys_slice,
                logical_start=logical_start,
                layer_id=self.layer_id,
                stream_id="K",
            )
            value_block = self.value_codec.encode_bhtd(
                values_slice,
                logical_start=logical_start,
                layer_id=self.layer_id,
                stream_id="V",
            )

            # Phase 5: append to combined paged arena (primary storage)
            if self._use_paged_arena:
                arena = self._ensure_kv_arena(key_block, value_block)
                arena.append(key_block, value_block)
            else:
                self._key_blocks.append(key_block)
                self._value_blocks.append(value_block)

            # Fix #2: Use typed methods instead of string-based increment
            # Fix #4: Record actual block creation and bytes written including scales
            if self.session:
                from rfsn_v10.cache.contracts import _array_itemsize
                self.session.runtime_counters.record_block_created()
                # Track bytes written for keys and values including scales
                if key_block.packed_codes is not None:
                    self.session.runtime_counters.record_packed_write(int(key_block.packed_codes.size) * _array_itemsize(key_block.packed_codes))
                if value_block.packed_codes is not None:
                    self.session.runtime_counters.record_packed_write(int(value_block.packed_codes.size) * _array_itemsize(value_block.packed_codes))
                if key_block.scales is not None:
                    self.session.runtime_counters.record_packed_write(int(key_block.scales.size) * _array_itemsize(key_block.scales))
                if value_block.scales is not None:
                    self.session.runtime_counters.record_packed_write(int(value_block.scales.size) * _array_itemsize(value_block.scales))

        self._encoded_tokens += n_full_blocks * block_size

        # Validate sealed block positions after flush
        key_blocks_list = list(self.iter_key_blocks())
        if key_blocks_list:
            validate_block_positions(key_blocks_list)
        value_blocks_list = list(self.iter_value_blocks())
        if value_blocks_list:
            validate_block_positions(value_blocks_list)

        # Keep remainder in staging
        if remainder > 0:
            start = n_full_blocks * block_size
            self._stage_keys = [keys_full[:, :, start:, :]]
            self._stage_values = [values_full[:, :, start:, :]]
            self._stage_token_count = remainder
        else:
            self._stage_keys.clear()
            self._stage_values.clear()
            self._stage_token_count = 0

    def _ensure_kv_arena(
        self,
        key_block: PackedBlock,
        value_block: PackedBlock,
    ) -> PagedKVArena:
        """Lazy-init the combined GPU arena once packed geometry is known."""
        if self._kv_arena is None:
            self._kv_arena = PagedKVArena(
                max_pages=self._max_pages,
                page_tokens=self.staging_capacity,
                n_kv_heads=key_block.n_kv_heads,
                k_words_per_vector=key_block.words_per_vector,
                v_words_per_vector=value_block.words_per_vector,
                k_groups_per_vector=key_block.groups_per_vector,
                v_groups_per_vector=value_block.groups_per_vector,
            )
        return self._kv_arena

    def _update_dense_residual(
        self, keys: Any, values: Any
    ) -> tuple[Any | None, Any | None]:
        """Maintain a bounded dense FP16 window of the most recent tokens.

        Returns
        -------
        evicted_keys, evicted_values
            Full-shaped tensors of tokens that fell out of the window,
            or ``(None, None)`` if no tokens were evicted.
        """
        if self._dense_keys is None:
            self._dense_keys = keys
            self._dense_values = values
        else:
            self._dense_keys = mx.concatenate([self._dense_keys, keys], axis=2)
            self._dense_values = mx.concatenate([self._dense_values, values], axis=2)

        total_dense = self._dense_keys.shape[2]
        evicted_k: Any | None = None
        evicted_v: Any | None = None

        if total_dense > self.dense_residual_window:
            # Tokens that fall outside the window are evicted to staging
            n_evict = total_dense - self.dense_residual_window
            evicted_k = self._dense_keys[:, :, :n_evict, :]
            evicted_v = self._dense_values[:, :, :n_evict, :]
            self._dense_keys = self._dense_keys[:, :, -self.dense_residual_window:, :]
            self._dense_values = self._dense_values[:, :, -self.dense_residual_window:, :]
            self._dense_token_count = self.dense_residual_window
        else:
            self._dense_token_count = total_dense

        return evicted_k, evicted_v

    # ------------------------------------------------------------------
    # Retrieval (for attention)
    # ------------------------------------------------------------------

    def iter_key_blocks(self):
        """Yield each sealed key block for blockwise attention.

        Phase 5: When the paged arena is active, delegates to the arena's
        backward-compatible iterator.  Production code should prefer
        ``get_paged_kv_view()``.
        """
        self._check_destroyed()
        if self._kv_arena is not None:
            yield from self._kv_arena.iter_key_blocks()
        else:
            yield from self._key_blocks

    def iter_value_blocks(self):
        """Yield each sealed value block for blockwise attention.

        Phase 5: Delegates to the arena when active.
        """
        self._check_destroyed()
        if self._kv_arena is not None:
            yield from self._kv_arena.iter_value_blocks()
        else:
            yield from self._value_blocks

    def get_dense_residual(self) -> tuple[Any | None, Any | None]:
        """Return the dense FP16 residual window, or (None, None)."""
        self._check_destroyed()
        return self._dense_keys, self._dense_values

    def get_paged_kv_view(self) -> PagedKVView | None:
        """Return a kernel view into the GPU paged arena, or ``None``.

        Production attention should use this instead of
        ``iter_key_blocks()`` / ``iter_value_blocks()``.
        """
        self._check_destroyed()
        if self._kv_arena is None or self._kv_arena.num_pages == 0:
            return None
        return self._kv_arena.view()

    def get_paged_arena_stats(self) -> dict[str, Any] | None:
        """Return arena instrumentation statistics, or ``None`` if not paged."""
        self._check_destroyed()
        if self._kv_arena is None:
            return None
        return self._kv_arena.to_instrumentation()

    def get_staging(self) -> tuple[Any | None, Any | None, int]:
        """Return staging keys, values, and token count.

        Returns full-shaped tensors ``(B, Hkv, staged_T, D)`` or ``(None, None, 0)``.
        """
        self._check_destroyed()
        if self._stage_token_count == 0:
            return None, None, 0
        keys = mx.concatenate(self._stage_keys, axis=2) if len(self._stage_keys) > 1 else self._stage_keys[0]
        values = mx.concatenate(self._stage_values, axis=2) if len(self._stage_values) > 1 else self._stage_values[0]
        return keys, values, self._stage_token_count

    # ------------------------------------------------------------------
    # Proof counters
    # ------------------------------------------------------------------

    @property
    def encoded_token_count(self) -> int:
        self._check_destroyed()
        return self._encoded_tokens

    @property
    def requantized_token_count(self) -> int:
        self._check_destroyed()
        return self._requantized_tokens

    def total_token_count(self) -> int:
        """Total tokens = encoded + staged + dense residual.

        These three regions are mutually exclusive.
        """
        self._check_destroyed()
        total = self._encoded_tokens + self._stage_token_count
        if self.dense_residual_window > 0 and self._dense_keys is not None:
            total += self._dense_token_count
        return total

    # ------------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------------

    def payload_bytes(self) -> int:
        """Exact bytes from all sealed blocks (valid payload only)."""
        self._check_destroyed()
        if self._kv_arena is not None:
            return self._kv_arena.active_payload_bytes
        total = 0
        for kb, vb in zip(self.iter_key_blocks(), self.iter_value_blocks()):
            total += kb.payload_bytes()
            total += vb.payload_bytes()
        return total

    def dense_residual_bytes(self) -> int:
        """Bytes in the dense FP16 residual window."""
        self._check_destroyed()
        if self._dense_keys is None:
            return 0
        return int(self._dense_keys.size) * 2 + int(self._dense_values.size) * 2

    def staging_bytes(self) -> int:
        """Bytes in staging buffers."""
        self._check_destroyed()
        total = 0
        from rfsn_v10.cache.contracts import _array_itemsize
        for k in self._stage_keys:
            total += int(k.size) * _array_itemsize(k)
        for v in self._stage_values:
            total += int(v.size) * _array_itemsize(v)
        return total

    def total_memory_bytes(self) -> int:
        """All accounted bytes: payload + dense + staging."""
        self._check_destroyed()
        return self.payload_bytes() + self.dense_residual_bytes() + self.staging_bytes()

    def stats(self) -> CacheStats:
        self._check_destroyed()
        # Count sealed pages from arena or legacy list
        if self._kv_arena is not None:
            sealed_count = self._kv_arena.num_pages
        else:
            sealed_count = sum(1 for _ in self.iter_key_blocks())
        return CacheStats(
            tokens_encoded=self._encoded_tokens,
            tokens_requantized=self._requantized_tokens,
            sealed_blocks=sealed_count,
            staged_tokens=self._stage_token_count,
            dense_residual_tokens=self._dense_token_count,
            payload_bytes=self.payload_bytes(),
        )

    def trim(self, new_token_count: int) -> None:
        """Trim is disabled until position-partition validation is complete.

        Raises:
            NotImplementedError: Always, to prevent data loss from the
                known-buggy trim implementation.
        """
        self._check_destroyed()
        raise NotImplementedError(
            "trim() is disabled in this release. Use reset() and re-prefill."
        )

    # ------------------------------------------------------------------
    # Blockwise attention (direct packed path — no full dense reconstruction)
    # ------------------------------------------------------------------

    def blockwise_attention(
        self,
        queries: Any,  # (B, Hq, Lq, D)
        scale: float,
        mask: Any | None = None,
        query_start_pos: int | None = None,
    ) -> Any:
        """Compute attention output directly from quantized blocks.

        Dequantizes one block at a time, accumulates online softmax,
        and never materialises the full dense KV history.

        Parameters
        ----------
        query_start_pos
            Global sequence position of the first query token.
            For decode (one new token) this is ``total_token_count``.
            If ``None``, inferred from ``total_token_count()``.

        Returns
        -------
        output
            Shape ``(B, Hq, Lq, D)``.
        """
        self._check_destroyed()
        B, Hq, Lq, D = queries.shape
        assert B == 1, "Batch size must be 1"

        if query_start_pos is None:
            query_start_pos = self.total_token_count()

        # Online softmax attention over blocks.
        # For each block we maintain:
        #   m = running max score
        #   l = running sum of exp(scores - m)
        #   o = running weighted value sum
        #
        # When a new block arrives with max m_j:
        #   m_new = max(m, m_j)
        #   l_new = l * exp(m - m_new) + sum(exp(scores_j - m_new))
        #   o_new = o * exp(m - m_new) + matmul(exp(scores_j - m_new), V_j)
        #
        # This is a Python reference; a production Metal kernel would
        # fuse dequant + matmul inside the shader.

        output = mx.zeros((B, Hq, Lq, D), dtype=mx.float32)
        running_max = mx.full((B, Hq, Lq, 1), -mx.inf, dtype=mx.float32)
        running_sum = mx.zeros((B, Hq, Lq, 1), dtype=mx.float32)
        has_mass = mx.zeros((B, Hq, Lq, 1), dtype=mx.bool_)

        def _process_block(k_block: Any, v_block: Any, block_t: int, token_offset: int) -> None:
            nonlocal output, running_max, running_sum, has_mass
            if k_block.shape[1] != Hq:
                repeats = Hq // k_block.shape[1]
                k_block = mx.repeat(k_block, repeats, axis=1)
                v_block = mx.repeat(v_block, repeats, axis=1)

            if mask is not None:
                block_mask = mask[..., token_offset:token_offset + block_t]
            else:
                # Causal mask: query at global position q can attend to kv at position kv if q >= kv
                # Query positions: query_start_pos .. query_start_pos + Lq - 1
                q_positions = mx.arange(query_start_pos, query_start_pos + Lq)[:, None]
                kv_positions = mx.arange(token_offset, token_offset + block_t)[None, :]
                block_mask = (q_positions >= kv_positions).astype(queries.dtype)
                block_mask = mx.broadcast_to(block_mask[None, None, :, :], (B, Hq, Lq, block_t))

            scores = mx.matmul(queries.astype(mx.float32), k_block.swapaxes(2, 3)) * scale
            # Use -inf for masked positions so they contribute 0 to softmax
            scores = mx.where(block_mask, scores, mx.array(-mx.inf, dtype=scores.dtype))

            block_max = mx.max(scores, axis=-1, keepdims=True)
            new_max = mx.maximum(running_max, block_max)
            old_scale = mx.exp(running_max - new_max)
            # Guard exp(-inf - (-inf)) => NaN on rows with no valid mass yet.
            old_scale = mx.where(mx.isfinite(old_scale), old_scale, 1.0)

            running_sum = running_sum * old_scale
            output = output * old_scale

            # Compute block contributions relative to the new global max.
            block_exp = mx.exp(scores.astype(mx.float32) - new_max)
            # Zero out exp(NaN) from masked positions when new_max is also -inf.
            block_exp = mx.where(mx.isfinite(block_exp), block_exp, 0.0)
            running_sum = running_sum + mx.sum(block_exp, axis=-1, keepdims=True)
            output = output + mx.matmul(block_exp, v_block.astype(mx.float32))
            running_max = new_max
            has_mass = mx.logical_or(has_mass, mx.any(mx.isfinite(scores), axis=-1, keepdims=True))

        token_offset = 0
        for kb, vb in zip(self.iter_key_blocks(), self.iter_value_blocks()):
            k_block = self.key_codec.decode_bhtd(kb)
            v_block = self.value_codec.decode_bhtd(vb)
            block_T = kb.token_count
            _process_block(k_block, v_block, block_T, token_offset)
            token_offset += block_T

        if self._stage_token_count > 0:
            stage_k = mx.concatenate(self._stage_keys, axis=2)
            stage_v = mx.concatenate(self._stage_values, axis=2)
            stage_T = self._stage_token_count
            _process_block(stage_k, stage_v, stage_T, token_offset)
            token_offset += stage_T

        if self._dense_keys is not None:
            dense_k = self._dense_keys
            dense_v = self._dense_values
            dense_T = self._dense_token_count
            _process_block(dense_k, dense_v, dense_T, token_offset)

        # Fully-masked rows return defined zero output.
        # Guard against running_sum == 0 due to numerical underflow.
        output = mx.where(
            has_mass & (running_sum > 0),
            output / running_sum,
            mx.zeros_like(output)
        )
        return output.astype(queries.dtype)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all cache state but preserve codec references.

        The cache can be reused after reset with the same codecs.
        """
        self._key_blocks.clear()
        self._value_blocks.clear()
        if self._kv_arena is not None:
            self._kv_arena.reset()
        self._stage_keys.clear()
        self._stage_values.clear()
        self._stage_token_count = 0
        self._dense_keys = None
        self._dense_values = None
        self._dense_token_count = 0
        self._encoded_tokens = 0
        self._requantized_tokens = 0
        self._geometry = None

    def destroy(self) -> None:
        """Permanently destroy the cache and prevent reuse.

        After destroy(), every public method raises RuntimeError.
        """
        self.reset()
        self._destroyed = True
