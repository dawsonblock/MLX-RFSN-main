"""FP16 residual window cache for RFSN v11.

Recent tokens stay in full FP16 precision (the "residual window").
When the window exceeds R tokens, the oldest batch is compressed into the main
compressed cache using WHT + grouped symmetric quantization.

Design based on KIVI's residual cache idea: keep recent tokens in full
precision to protect local context while reducing memory for older history.

Parameters
----------
head_dim : int
    Attention head dimension.
residual_length : int
    Number of recent tokens to keep in FP16 (default 128).
key_bits : int
    Bits for compressed keys (default 8).
value_bits : int
    Bits for compressed values (default 4).
group_size : int
    Quantization group size (default 64).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).parent.parent.parent)
)

_MLX_AVAILABLE = False
try:
    import mlx.core as mx
    _MLX_AVAILABLE = True
except ImportError:
    pass


class ResidualKVCache:
    """FP16 residual window + compressed history KV cache."""

    def __init__(
        self,
        head_dim: int,
        residual_length: int = 128,
        key_bits: int = 8,
        value_bits: int = 5,
        group_size: int = 64,
    ) -> None:
        if not _MLX_AVAILABLE:
            raise ImportError("mlx is required for ResidualKVCache")

        self.head_dim = head_dim
        self.residual_length = residual_length
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.group_size = group_size
        self.offset = 0

        # Compressed history (A1-style cache)
        from benchmarks.candidates.a1_wht_grouped_k8v4_gs64 import (
            A1_WHT_GroupedKVCache,
        )
        self.compressed_cache = A1_WHT_GroupedKVCache(
            head_dim=head_dim,
            key_bits=key_bits,
            value_bits=value_bits,
            group_size=group_size,
        )

        # FP16 residual window (raw K, V tensors)
        self.residual_keys: "mx.array" | None = None
        self.residual_values: "mx.array" | None = None

        # Timing
        self.compression_time_ms: float = 0.0
        self.decompression_time_ms: float = 0.0

    def update_and_fetch(
        self,
        keys: "mx.array",
        values: "mx.array",
    ) -> tuple["mx.array", "mx.array"]:
        """Add new tokens and return full history (compressed + residual).

        Parameters
        ----------
        keys, values : (B, H, T, D)

        Returns
        -------
        (full_keys, full_values) : (B, H, offset, D)
        """
        B, H, T, D = keys.shape
        assert D == self.head_dim

        # Append new tokens to residual window
        if self.residual_keys is None:
            self.residual_keys = keys
            self.residual_values = values
        else:
            self.residual_keys = mx.concatenate(
                [self.residual_keys, keys], axis=-2
            )
            self.residual_values = mx.concatenate(
                [self.residual_values, values], axis=-2
            )

        self.offset += T

        # If residual window exceeds R, compress oldest tokens
        if (
            self.residual_keys is not None
            and self.residual_keys.shape[-2] > self.residual_length
        ):
            excess = self.residual_keys.shape[-2] - self.residual_length
            # Oldest excess tokens go to compressed cache
            oldest_keys = self.residual_keys[..., :excess, :]
            oldest_values = self.residual_values[..., :excess, :]
            self.compressed_cache.update_and_fetch(oldest_keys, oldest_values)
            # Trim residual window
            self.residual_keys = self.residual_keys[..., excess:, :]
            self.residual_values = self.residual_values[..., excess:, :]

        mx.eval(self.residual_keys, self.residual_values)

        # Build full history: compressed (decompressed) + residual
        if self.compressed_cache.offset > 0:
            comp_keys, comp_values = self._get_compressed_state(B, H)
            full_keys = mx.concatenate(
                [comp_keys, self.residual_keys], axis=-2
            )
            full_values = mx.concatenate(
                [comp_values, self.residual_values], axis=-2
            )
        else:
            full_keys = self.residual_keys
            full_values = self.residual_values

        mx.eval(full_keys, full_values)
        return full_keys, full_values

    def _get_compressed_state(
        self, B: int, H: int
    ) -> tuple["mx.array", "mx.array"]:
        """Return decompressed history from compressed cache."""
        # We need to call update_and_fetch with zero new tokens to get the
        # current decompressed state.  Since the compressed cache has no new
        # tokens to add, we use a dummy zero-token tensor.
        dummy_k = mx.zeros((B, H, 0, self.head_dim), dtype=mx.float16)
        dummy_v = mx.zeros((B, H, 0, self.head_dim), dtype=mx.float16)
        return self.compressed_cache.update_and_fetch(dummy_k, dummy_v)

    @property
    def residual_memory_mb(self) -> float:
        """Size of FP16 residual window in MB."""
        if self.residual_keys is None:
            return 0.0
        tokens = self.residual_keys.shape[-2]
        B_H = self.residual_keys.shape[0] * self.residual_keys.shape[1]
        bytes_per_token = B_H * self.head_dim * 2  # FP16 = 2 bytes
        return tokens * bytes_per_token / (1024 ** 2)

    @property
    def compressed_history_memory_mb(self) -> float:
        """Size of compressed history in MB."""
        return self.compressed_cache.compressed_bytes() / (1024 ** 2)

    def compressed_bytes(self) -> int:
        """Total compressed bytes (compressed history only;
        residual is FP16).
        """
        return self.compressed_cache.compressed_bytes()

    # ------------------------------------------------------------------
    # mlx_lm KV cache compat interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> tuple:
        # For compatibility, return raw state as (keys_tuple, values_tuple)
        if self.residual_keys is not None:
            return (
                (self.residual_keys, self.residual_values),
                self.compressed_cache.state,
            )
        return None, self.compressed_cache.state

    @state.setter
    def state(self, v: tuple) -> None:
        (self.residual_keys, self.residual_values), comp_state = v
        self.compressed_cache.state = comp_state

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        if self.residual_keys is not None and n > 0:
            residual_tokens = self.residual_keys.shape[-2]
            if n <= residual_tokens:
                self.residual_keys = self.residual_keys[..., :-n, :]
                self.residual_values = self.residual_values[..., :-n, :]
            else:
                # Need to trim compressed cache too
                excess = n - residual_tokens
                self.residual_keys = None
                self.residual_values = None
                self.compressed_cache.trim(excess)
        return n
