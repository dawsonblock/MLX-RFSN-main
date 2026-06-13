"""NumPy reference oracle for the incremental KV cache.

Provides :class:`NumpyLayerCache` — a CPU-only reference that mirrors
:class:`QuantizedLayerCache` semantics without MLX or quantization.

Use this to:
  * Validate cache invariants on CI runners without Apple Silicon.
  * Cross-check the MLX cache against a simple, obviously-correct oracle.
  * Catch token-count and ordering mismatches before they reach hardware.
"""
from __future__ import annotations

import numpy as np


class NumpyLayerCache:
    """Reference layer cache using plain NumPy.

    Stores K/V exactly (no quantization) so that any structural mismatch
    with the real MLX cache is immediately visible.
    """

    def __init__(self, staging_capacity: int = 64, dense_residual_window: int = 0) -> None:
        self.staging_capacity = staging_capacity
        self.dense_residual_window = dense_residual_window

        # Immutable sealed blocks — list of (keys, values) tuples
        self._key_blocks: list[np.ndarray] = []
        self._value_blocks: list[np.ndarray] = []

        # Staging buffers — list of full-shaped (B, Hkv, T, D) arrays
        self._stage_keys: list[np.ndarray] = []
        self._stage_values: list[np.ndarray] = []
        self._stage_token_count: int = 0

        # Dense residual
        self._dense_keys: np.ndarray | None = None
        self._dense_values: np.ndarray | None = None
        self._dense_token_count: int = 0

        self._encoded_tokens: int = 0

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(self, keys: np.ndarray, values: np.ndarray) -> None:
        B, Hkv, new_T, D = keys.shape
        assert B == 1, "Batch size must be 1"

        if self.dense_residual_window > 0:
            evicted_k, evicted_v = self._update_dense_residual(keys, values)
            if evicted_k is not None:
                self._add_to_staging(evicted_k, evicted_v)
        else:
            self._add_to_staging(keys, values)

    def _add_to_staging(self, keys: np.ndarray, values: np.ndarray) -> None:
        self._stage_keys.append(keys)
        self._stage_values.append(values)
        self._stage_token_count += keys.shape[2]

        if self._stage_token_count >= self.staging_capacity:
            self._flush_staging()

    def _flush_staging(self) -> None:
        if self._stage_token_count == 0:
            return

        keys_full = np.concatenate(self._stage_keys, axis=2)
        values_full = np.concatenate(self._stage_values, axis=2)

        B, Hkv, stage_T, D = keys_full.shape
        assert B == 1

        block_size = self.staging_capacity
        n_full_blocks = stage_T // block_size
        remainder = stage_T % block_size

        for i in range(n_full_blocks):
            start = i * block_size
            end = start + block_size
            self._key_blocks.append(keys_full[:, :, start:end, :])
            self._value_blocks.append(values_full[:, :, start:end, :])
            self._encoded_tokens += block_size

        if remainder > 0:
            start = n_full_blocks * block_size
            self._stage_keys = [keys_full[:, :, start:, :]]
            self._stage_values = [values_full[:, :, start:, :]]
            self._stage_token_count = remainder
        else:
            self._stage_keys.clear()
            self._stage_values.clear()
            self._stage_token_count = 0

    def _update_dense_residual(
        self, keys: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self._dense_keys is None:
            self._dense_keys = keys
            self._dense_values = values
        else:
            self._dense_keys = np.concatenate([self._dense_keys, keys], axis=2)
            self._dense_values = np.concatenate([self._dense_values, values], axis=2)

        total_dense = self._dense_keys.shape[2]
        evicted_k: np.ndarray | None = None
        evicted_v: np.ndarray | None = None

        if total_dense > self.dense_residual_window:
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
    # Retrieval
    # ------------------------------------------------------------------

    def iter_key_blocks(self):
        yield from self._key_blocks

    def iter_value_blocks(self):
        yield from self._value_blocks

    def get_dense_residual(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self._dense_keys, self._dense_values

    def get_staging(self) -> tuple[np.ndarray | None, np.ndarray | None, int]:
        if self._stage_token_count == 0:
            return None, None, 0
        keys = np.concatenate(self._stage_keys, axis=2) if len(self._stage_keys) > 1 else self._stage_keys[0]
        values = np.concatenate(self._stage_values, axis=2) if len(self._stage_values) > 1 else self._stage_values[0]
        return keys, values, self._stage_token_count

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------

    @property
    def encoded_token_count(self) -> int:
        return self._encoded_tokens

    @property
    def requantized_token_count(self) -> int:
        return 0

    def total_token_count(self) -> int:
        total = self._encoded_tokens + self._stage_token_count
        if self.dense_residual_window > 0 and self._dense_keys is not None:
            total += self._dense_token_count
        return total

    # ------------------------------------------------------------------
    # Trim
    # ------------------------------------------------------------------

    def trim(self, new_token_count: int) -> None:
        if new_token_count >= self.total_token_count():
            return
        if new_token_count <= 0:
            self.reset()
            return

        if new_token_count < self._encoded_tokens:
            keep_blocks = 0
            cumulative = 0
            for kb in self._key_blocks:
                if cumulative + kb.shape[2] > new_token_count:
                    break
                cumulative += kb.shape[2]
                keep_blocks += 1

            self._key_blocks = self._key_blocks[:keep_blocks]
            self._value_blocks = self._value_blocks[:keep_blocks]
            self._encoded_tokens = cumulative
            self._stage_keys.clear()
            self._stage_values.clear()
            self._stage_token_count = 0
            self._dense_keys = None
            self._dense_values = None
            self._dense_token_count = 0
            return

        remaining = new_token_count - self._encoded_tokens
        if remaining <= 0:
            self._stage_keys.clear()
            self._stage_values.clear()
            self._stage_token_count = 0
            self._dense_keys = None
            self._dense_values = None
            self._dense_token_count = 0
            return

        if self._stage_token_count > 0:
            if remaining < self._stage_token_count:
                keys_full = np.concatenate(self._stage_keys, axis=2)
                values_full = np.concatenate(self._stage_values, axis=2)
                self._stage_keys = [keys_full[:, :, :remaining, :]]
                self._stage_values = [values_full[:, :, :remaining, :]]
                self._stage_token_count = remaining
                self._dense_keys = None
                self._dense_values = None
                self._dense_token_count = 0
                return
            remaining -= self._stage_token_count

        if remaining <= 0:
            self._dense_keys = None
            self._dense_values = None
            self._dense_token_count = 0
        elif remaining < self._dense_token_count:
            self._dense_keys = self._dense_keys[:, :, :remaining, :]
            self._dense_values = self._dense_values[:, :, :remaining, :]
            self._dense_token_count = remaining

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._key_blocks.clear()
        self._value_blocks.clear()
        self._stage_keys.clear()
        self._stage_values.clear()
        self._stage_token_count = 0
        self._dense_keys = None
        self._dense_values = None
        self._dense_token_count = 0
        self._encoded_tokens = 0

    # ------------------------------------------------------------------
    # Dense reconstruction (reference path)
    # ------------------------------------------------------------------

    def reconstruct_dense(self) -> tuple[np.ndarray, np.ndarray]:
        key_parts = list(self._key_blocks)
        value_parts = list(self._value_blocks)

        stage_k, stage_v, _ = self.get_staging()
        if stage_k is not None:
            key_parts.append(stage_k)
            value_parts.append(stage_v)

        dense_k, dense_v = self.get_dense_residual()
        if dense_k is not None:
            key_parts.append(dense_k)
            value_parts.append(dense_v)

        if not key_parts:
            # Need shape metadata to return empty arrays.  Infer from
            # whatever staging last saw, or use a minimal default.
            empty = np.zeros((1, 1, 0, 1), dtype=np.float32)
            return empty, empty

        full_k = np.concatenate(key_parts, axis=2)
        full_v = np.concatenate(value_parts, axis=2)
        return full_k, full_v

    def numpy_attention(
        self,
        queries: np.ndarray,
        scale: float | None = None,
    ) -> np.ndarray:
        """Pure-NumPy blockwise attention (reference for MLX path).

        Returns shape (B, Hq, Lq, D).
        """
        B, Hq, Lq, D = queries.shape
        s = scale if scale is not None else (D ** -0.5)

        # GQA: infer repeats from first block
        n_kv_heads = None
        for kb in self.iter_key_blocks():
            n_kv_heads = kb.shape[1]
            break
        if n_kv_heads is None:
            sk, _, _ = self.get_staging()
            if sk is not None:
                n_kv_heads = sk.shape[1]
        if n_kv_heads is None:
            dk, _ = self.get_dense_residual()
            if dk is not None:
                n_kv_heads = dk.shape[1]
        if n_kv_heads is None:
            return np.zeros((B, Hq, Lq, D), dtype=queries.dtype)

        repeats = Hq // n_kv_heads

        m = np.full((B, Hq, Lq, 1), -1e9, dtype=np.float32)
        sum_exp = np.zeros((B, Hq, Lq, 1), dtype=np.float32)
        out = np.zeros((B, Hq, Lq, D), dtype=np.float32)

        def _process(k_block: np.ndarray, v_block: np.ndarray, block_t: int) -> None:
            nonlocal m, sum_exp, out
            k_expanded = np.repeat(k_block, repeats, axis=1)
            v_expanded = np.repeat(v_block, repeats, axis=1)

            scores = np.matmul(queries, k_expanded.transpose(0, 1, 3, 2)) * s

            block_max = np.max(scores, axis=-1, keepdims=True)
            m_new = np.maximum(m, block_max)

            exp_diff_m = np.exp(m - m_new)
            sum_exp = sum_exp * exp_diff_m
            out = out * exp_diff_m

            exp_scores = np.exp(scores.astype(np.float32) - m_new)
            sum_exp = sum_exp + np.sum(exp_scores, axis=-1, keepdims=True)

            block_contrib = np.matmul(exp_scores, v_expanded)
            out = out + block_contrib
            m = m_new

        # Sealed blocks
        for kb, vb in zip(self.iter_key_blocks(), self.iter_value_blocks()):
            _process(kb, vb, kb.shape[2])

        # Staging
        sk, sv, sn = self.get_staging()
        if sk is not None:
            _process(sk, sv, sn)

        # Dense residual
        dk, dv = self.get_dense_residual()
        if dk is not None:
            _process(dk, dv, dk.shape[2])

        # Final normalisation
        output = out / sum_exp
        return output.astype(queries.dtype)


# ------------------------------------------------------------------
# Reference attention (NumPy)
# ------------------------------------------------------------------

def numpy_attention(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
    scale: float | None = None,
) -> np.ndarray:
    """Compute dense attention using NumPy (reference oracle).

    Parameters
    ----------
    queries
        Shape ``(B, Hq, Lq, D)``.
    keys, values
        Shape ``(B, Hkv, T, D)``.
    scale
        Attention scale; defaults to ``D ** -0.5``.

    Returns
    -------
    output
        Shape ``(B, Hq, Lq, D)``.
    """
    if scale is None:
        scale = queries.shape[-1] ** -0.5

    # GQA: repeat K/V heads to match query heads
    n_kv_heads = keys.shape[1]
    repeats = queries.shape[1] // n_kv_heads
    if repeats > 1:
        keys = np.repeat(keys, repeats, axis=1)
        values = np.repeat(values, repeats, axis=1)

    # Q @ K.T  → (B, Hq, Lq, T)
    scores = np.einsum("bhqd,bhtd->bhqt", queries, keys) * scale

    # Softmax
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
    weights = exp_scores / sum_exp

    # weights @ V  → (B, Hq, Lq, D)
    output = np.einsum("bhqt,bhtd->bhqd", weights, values)
    return output.astype(queries.dtype)
