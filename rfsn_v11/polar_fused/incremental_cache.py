"""Incremental quantized KV cache.

Avoids re-quantizing the full cache on every decode step by:
1. Maintaining persistent packed key/value indices and norms
2. Quantizing only newly appended tokens
3. Attending directly from packed data via reference kernels

Uses chunked accumulation to minimize mx.concatenate calls: new tokens are
staged in a small buffer and flushed to the main cache in batches.
"""
from __future__ import annotations

from typing import Any

from .contracts import QuantizedVectors
from .packing import pack_indices, unpack_indices
from .quantize import PolarQuantizer

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    HAS_MLX = False


class IncrementalPolarCache:
    """Persistent quantized cache that only quantizes new tokens.

    Uses chunked accumulation: new tokens are staged and flushed in batches
    to minimize ``mx.concatenate`` calls (from O(T) to O(T / chunk_size)).
    """

    def __init__(
        self,
        key_quantizer: PolarQuantizer,
        value_quantizer: PolarQuantizer,
        batch_size: int = 1,
        num_kv_heads: int = 1,
        head_dim: int = 64,
        chunk_size: int = 64,
    ) -> None:
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        self.key_q = key_quantizer
        self.value_q = value_quantizer
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        self._key_bits = key_quantizer.bits
        self._value_bits = value_quantizer.bits
        self._key_values_per_word = {2: 16, 3: 10, 4: 8}[self._key_bits]
        self._value_values_per_word = {2: 16, 3: 10, 4: 8}[self._value_bits]
        self._key_words_per_vec = (head_dim + self._key_values_per_word - 1) // self._key_values_per_word
        self._value_words_per_vec = (head_dim + self._value_values_per_word - 1) // self._value_values_per_word

        # Cache rotation matrices and centroids (avoid registry lookups per attend)
        self._Rk_T = self.key_q._rot_registry.get_transpose(head_dim, self.key_q.rotation_seed)
        self._Rv = self.value_q._rot_registry.get(head_dim, self.value_q.rotation_seed)
        self._key_centroids = self.key_q._cb_registry.centroids(self._key_bits)
        self._value_centroids = self.value_q._cb_registry.centroids(self._value_bits)

        # Main persistent state
        self._packed_key_indices: Any | None = None   # (B, Hkv, T, words)
        self._key_norms: Any | None = None            # (B, Hkv, T)
        self._packed_value_indices: Any | None = None # (B, Hkv, T, words)
        self._value_norms: Any | None = None          # (B, Hkv, T)
        self._token_count = 0

        # Staging buffer (pending flush)
        self._stage_keys: list[Any] = []
        self._stage_values: list[Any] = []
        self._stage_count = 0

    # ------------------------------------------------------------------
    # Append (chunked staging)
    # ------------------------------------------------------------------

    def append(self, keys: Any, values: Any) -> None:
        """Append new K/V tokens to the quantized cache.

        Parameters
        ----------
        keys, values
            Shape ``(batch, n_kv_heads, new_tokens, head_dim)``.
        """
        if not HAS_MLX:
            return

        B, Hkv, new_T, D = keys.shape
        assert B == self.batch_size
        assert Hkv == self.num_kv_heads
        assert D == self.head_dim

        # Quantize only new tokens
        key_qv = self.key_q.quantize(keys.reshape(-1, D))
        value_qv = self.value_q.quantize(values.reshape(-1, D))

        # Pack indices
        packed_keys = pack_indices(key_qv.indices.reshape(B, Hkv, new_T, D), self._key_bits)
        packed_values = pack_indices(value_qv.indices.reshape(B, Hkv, new_T, D), self._value_bits)
        key_norms_new = key_qv.norms.reshape(B, Hkv, new_T)
        value_norms_new = value_qv.norms.reshape(B, Hkv, new_T)

        if self._token_count == 0 and self._stage_count == 0:
            # First allocation — direct store, no staging needed
            self._packed_key_indices = packed_keys
            self._key_norms = key_norms_new
            self._packed_value_indices = packed_values
            self._value_norms = value_norms_new
            self._token_count = new_T
            return

        # Stage the new tokens
        self._stage_keys.append((packed_keys, key_norms_new))
        self._stage_values.append((packed_values, value_norms_new))
        self._stage_count += new_T

        # Flush if chunk_size reached
        if self._stage_count >= self.chunk_size:
            self._flush()

    def _flush(self) -> None:
        """Flush staged tokens into main cache via a single concatenate."""
        if self._stage_count == 0:
            return

        # Concatenate all staged key pieces
        key_packed_parts = [p for p, _ in self._stage_keys]
        key_norm_parts = [n for _, n in self._stage_keys]
        value_packed_parts = [p for p, _ in self._stage_values]
        value_norm_parts = [n for _, n in self._stage_values]

        if self._token_count > 0:
            self._packed_key_indices = mx.concatenate(
                [self._packed_key_indices, *key_packed_parts], axis=2
            )
            self._key_norms = mx.concatenate(
                [self._key_norms, *key_norm_parts], axis=2
            )
            self._packed_value_indices = mx.concatenate(
                [self._packed_value_indices, *value_packed_parts], axis=2
            )
            self._value_norms = mx.concatenate(
                [self._value_norms, *value_norm_parts], axis=2
            )
        else:
            self._packed_key_indices = mx.concatenate(key_packed_parts, axis=2)
            self._key_norms = mx.concatenate(key_norm_parts, axis=2)
            self._packed_value_indices = mx.concatenate(value_packed_parts, axis=2)
            self._value_norms = mx.concatenate(value_norm_parts, axis=2)

        self._token_count += self._stage_count
        self._stage_keys.clear()
        self._stage_values.clear()
        self._stage_count = 0

    # ------------------------------------------------------------------
    # Attend helpers
    # ------------------------------------------------------------------

    def _get_main_cache(self) -> tuple[Any, Any, Any, Any]:
        """Return main cache arrays, flushing staging buffer first."""
        self._flush()
        return (
            self._packed_key_indices,
            self._key_norms,
            self._packed_value_indices,
            self._value_norms,
        )

    # ------------------------------------------------------------------
    # Attend (dequantized oracle path)
    # ------------------------------------------------------------------

    def attend_naive(
        self,
        queries: Any,  # (B, Hq, Lq, D)
        mask: Any | None = None,
        scale: float | None = None,
    ) -> Any:
        """Attention via NaivePolarAttention (correctness oracle)."""
        from .attention import NaivePolarAttention

        if self._token_count == 0 and self._stage_count == 0:
            raise RuntimeError("Cache is empty; call append() first")

        pk, kn, pv, vn = self._get_main_cache()

        key_indices = unpack_indices(pk, self._key_bits, self.head_dim)
        value_indices = unpack_indices(pv, self._value_bits, self.head_dim)

        key_qv = QuantizedVectors(
            indices=key_indices, norms=kn,
            original_dim=self.head_dim, bits=self._key_bits,
            rotation_id=self.key_q._rotation_id, codebook_id=self.key_q._codebook_id,
        )
        value_qv = QuantizedVectors(
            indices=value_indices, norms=vn,
            original_dim=self.head_dim, bits=self._value_bits,
            rotation_id=self.value_q._rotation_id, codebook_id=self.value_q._codebook_id,
        )

        attn = NaivePolarAttention(self.key_q, self.value_q, scale=scale)
        result = attn.attend(queries, key_qv, value_qv, mask)
        return result.output

    # ------------------------------------------------------------------
    # Attend (kernel path — no dequantize)
    # ------------------------------------------------------------------

    def attend_kernel(
        self,
        queries: Any,  # (B, Hq, Lq, D)
        mask: Any | None = None,
        scale: float | None = None,
    ) -> Any:
        """Attention via memory-efficient reference kernels on packed data."""
        if self._token_count == 0 and self._stage_count == 0:
            raise RuntimeError("Cache is empty; call append() first")

        pk, kn, pv, vn = self._get_main_cache()
        head_dim = self.head_dim
        s = scale if scale is not None else (head_dim ** -0.5)

        # Rotate queries into key basis (R.T basis)
        q_rot = queries @ self._Rk_T  # (B, Hq, Lq, D)

        # QK kernel — memory-efficient: no full unpack materialization
        scores = self._qk_kernel_memeff(
            q_rot, pk, kn, self._key_centroids, s, self._key_bits, self._key_values_per_word
        )

        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(queries.dtype)

        # SV kernel — memory-efficient
        output_rot = self._sv_kernel_memeff(
            weights, pv, vn, self._value_centroids, self._value_bits, self._value_values_per_word
        )

        # Rotate back to original basis
        output = output_rot @ self._Rv
        return output

    # ------------------------------------------------------------------
    # Internal: memory-efficient kernels (no full materialization)
    # ------------------------------------------------------------------

    def _qk_kernel_memeff(
        self,
        rotated_queries: Any,      # (B, Hq, Lq, D)
        packed_key_indices: Any,   # (B, Hkv, Lkv, n_words)
        key_norms: Any,            # (B, Hkv, Lkv)
        key_centroids: Any,        # (n_centroids,)
        scale: float,
        bits: int,
        values_per_word: int,
    ) -> Any:
        """Memory-efficient QK: accumulate dot product word-by-word.

        Avoids materializing a full (B, Hkv, Lkv, D) float32 array by
        extracting each coordinate from packed words and accumulating
        the partial dot product directly.
        """
        B, Hq, Lq, D = rotated_queries.shape
        _, Hkv, Lkv, n_words = packed_key_indices.shape
        mask = (1 << bits) - 1
        repeats = Hq // Hkv

        scores = mx.zeros((B, Hq, Lq, Lkv), dtype=rotated_queries.dtype)

        # Expand norms for GQA
        key_norms_rep = mx.repeat(key_norms, repeats, axis=1)  # (B, Hq, Lkv)

        for w in range(n_words):
            word = packed_key_indices[..., w]  # (B, Hkv, Lkv)
            for slot in range(values_per_word):
                coord_idx = w * values_per_word + slot
                if coord_idx >= D:
                    break
                # Extract index for this slot
                idx = mx.bitwise_and(mx.right_shift(word, slot * bits), mask).astype(mx.uint8)
                # Lookup centroid: (B, Hkv, Lkv)
                cvals = key_centroids[idx]
                # GQA expand
                cvals_rep = mx.repeat(cvals, repeats, axis=1)  # (B, Hq, Lkv)
                # Query coordinate: (B, Hq, Lq, 1)
                q_coord = rotated_queries[..., coord_idx:coord_idx + 1]
                # Accumulate: (B, Hq, Lq, 1) * (B, Hq, 1, Lkv) → (B, Hq, Lq, Lkv)
                scores = scores + q_coord * cvals_rep[..., None, :]

        # Apply norms and scale
        scores = scores * key_norms_rep[..., None, :] * scale
        return scores

    def _sv_kernel_memeff(
        self,
        attention_weights: Any,    # (B, Hq, Lq, Lkv)
        packed_value_indices: Any, # (B, Hkv, Lkv, n_words)
        value_norms: Any,          # (B, Hkv, Lkv)
        value_centroids: Any,      # (n_centroids,)
        bits: int,
        values_per_word: int,
    ) -> Any:
        """Memory-efficient SV: accumulate weighted sum word-by-word.

        Avoids materializing a full (B, Hkv, Lkv, D) float32 array by
        extracting each coordinate from packed words and accumulating
        the weighted sum directly into the output buffer.
        """
        B, Hq, Lq, Lkv = attention_weights.shape
        _, Hkv, _, n_words = packed_value_indices.shape
        mask = (1 << bits) - 1
        repeats = Hq // Hkv
        D = self.head_dim

        # Accumulate output as a list of per-coordinate values, then stack
        output_parts: list[Any] = []

        for w in range(n_words):
            word = packed_value_indices[..., w]  # (B, Hkv, Lkv)
            for slot in range(values_per_word):
                coord_idx = w * values_per_word + slot
                if coord_idx >= D:
                    break
                # Extract index for this slot
                idx = mx.bitwise_and(mx.right_shift(word, slot * bits), mask).astype(mx.uint8)
                # Lookup centroid and apply norm: (B, Hkv, Lkv)
                cvals = value_centroids[idx] * value_norms
                # GQA expand: (B, Hq, Lkv)
                cvals_rep = mx.repeat(cvals, repeats, axis=1)
                # Weighted sum over Lkv: (B, Hq, Lq, Lkv) * (B, Hq, 1, Lkv)
                # sum(axis=-1) → (B, Hq, Lq)
                weighted = mx.sum(
                    attention_weights * cvals_rep[..., None, :],
                    axis=-1
                )
                output_parts.append(weighted)

        # Stack all coordinates: list of (B, Hq, Lq) → (B, Hq, Lq, D)
        return mx.stack(output_parts, axis=-1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def token_count(self) -> int:
        """Total tokens in cache (including staged but not yet flushed)."""
        return self._token_count + self._stage_count

    def memory_bytes(self) -> int:
        """Return total bytes used by packed cache (valid + staged)."""
        total = 0
        if self._packed_key_indices is not None:
            total += int(self._packed_key_indices.size) * 4
        if self._packed_value_indices is not None:
            total += int(self._packed_value_indices.size) * 4
        if self._key_norms is not None:
            total += int(self._key_norms.size) * 4
        if self._value_norms is not None:
            total += int(self._value_norms.size) * 4
        # Staged tokens
        for pk, _ in self._stage_keys:
            total += int(pk.size) * 4
        for _, kn in self._stage_keys:
            total += int(kn.size) * 4
        for pv, _ in self._stage_values:
            total += int(pv.size) * 4
        for _, vn in self._stage_values:
            total += int(vn.size) * 4
        return total

    def metadata(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "flushed_tokens": self._token_count,
            "staged_tokens": self._stage_count,
            "memory_bytes": self.memory_bytes(),
            "key_bits": self._key_bits,
            "value_bits": self._value_bits,
            "head_dim": self.head_dim,
            "num_kv_heads": self.num_kv_heads,
            "chunk_size": self.chunk_size,
        }
