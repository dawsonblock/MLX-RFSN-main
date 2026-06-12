"""Lazy FP16-to-Polar packed conversion.

Short contexts are usually better left in FP16 because compressed attention has
fixed overhead.  This module implements the lazy conversion policy.

Cache states:
  EMPTY         → No data yet
  FP16_WARMUP   → Keeping full FP16 cache
  CONVERTING    → Bulk conversion in progress
  POLAR_PACKED  → Compressed cache active
  FALLBACK      → Conversion failed, retain FP16
"""
from __future__ import annotations

from enum import Enum, auto
from typing import Any

from .cache import PolarCache
from .config import PolarFusedConfig
from .packing import pack_indices
from .quantize import PolarQuantizer

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    HAS_MLX = False


class CacheState(Enum):
    EMPTY = auto()
    FP16_WARMUP = auto()
    CONVERTING = auto()
    POLAR_PACKED = auto()
    FALLBACK = auto()


class LazyPolarCache:
    """Cache that transitions from FP16 to Polar packed at a threshold.

    Behavior:
    1. Keep initial tokens in FP16.
    2. At the configured threshold, bulk-quantize existing cache once.
    3. Release FP16 storage after successful conversion.
    4. Quantize future decode tokens incrementally.
    5. If conversion fails, retain FP16 and mark as fallback.
    """

    def __init__(
        self,
        config: PolarFusedConfig,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        key_quantizer: PolarQuantizer,
        value_quantizer: PolarQuantizer,
    ) -> None:
        self.cfg = config
        self.batch_size = batch_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.key_q = key_quantizer
        self.value_q = value_quantizer

        self.state = CacheState.EMPTY
        self.token_count = 0

        # FP16 cache (only valid in FP16_WARMUP)
        self._fp16_keys: Any | None = None
        self._fp16_values: Any | None = None

        # Polar packed cache (only valid in POLAR_PACKED)
        self._polar_cache: PolarCache | None = None

        # Threshold
        self._threshold = config.lazy_quantization_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, key: Any, value: Any) -> None:
        """Append a single token.  Handles state transitions automatically."""
        if not HAS_MLX:
            raise RuntimeError("MLX is not installed")

        if self.state == CacheState.EMPTY:
            self._fp16_keys = key
            self._fp16_values = value
            self.state = CacheState.FP16_WARMUP
            self.token_count = key.shape[2]
            return

        if self.state == CacheState.FP16_WARMUP:
            # Concatenate with existing FP16 cache
            self._fp16_keys = mx.concatenate([self._fp16_keys, key], axis=2)
            self._fp16_values = mx.concatenate([self._fp16_values, value], axis=2)
            self.token_count += key.shape[2]

            # Check threshold
            if self.token_count >= self._threshold:
                self._convert()
            return

        if self.state == CacheState.POLAR_PACKED:
            # Quantize new token and append to Polar cache
            self._append_packed(key, value)
            self.token_count += key.shape[2]
            return

        if self.state == CacheState.FALLBACK:
            # Keep appending to FP16
            self._fp16_keys = mx.concatenate([self._fp16_keys, key], axis=2)
            self._fp16_values = mx.concatenate([self._fp16_values, value], axis=2)
            self.token_count += key.shape[2]
            return

        # CONVERTING should not receive appends
        raise RuntimeError(f"Cannot append in state {self.state}")

    def get_cache_for_attention(self) -> dict[str, Any]:
        """Return cache data in a format suitable for attention.

        Returns dict with:
          - "mode": "fp16" or "polar"
          - "keys", "values": FP16 tensors (if mode == fp16)
          - "polar_cache": PolarCache (if mode == polar)
        """
        if self.state in (CacheState.FP16_WARMUP, CacheState.FALLBACK):
            return {
                "mode": "fp16",
                "keys": self._fp16_keys,
                "values": self._fp16_values,
            }
        elif self.state == CacheState.POLAR_PACKED:
            return {
                "mode": "polar",
                "polar_cache": self._polar_cache,
            }
        else:
            raise RuntimeError(f"Cannot attend in state {self.state}")

    def trim(self, new_token_count: int) -> None:
        """Trim cache to retain only first N tokens."""
        if new_token_count >= self.token_count:
            return
        if new_token_count <= 0:
            self._reset()
            return

        if self.state in (CacheState.FP16_WARMUP, CacheState.FALLBACK):
            self._fp16_keys = self._fp16_keys[..., :new_token_count, :]
            self._fp16_values = self._fp16_values[..., :new_token_count, :]
            self.token_count = new_token_count
        elif self.state == CacheState.POLAR_PACKED and self._polar_cache:
            self._polar_cache.trim(new_token_count)
            self.token_count = new_token_count

    def memory_bytes(self) -> int:
        """Current memory usage."""
        if self.state in (CacheState.FP16_WARMUP, CacheState.FALLBACK):
            if self._fp16_keys is not None:
                return (
                    int(self._fp16_keys.size) * 2 +
                    int(self._fp16_values.size) * 2
                )
            return 0
        elif self.state == CacheState.POLAR_PACKED and self._polar_cache:
            return self._polar_cache.memory_bytes()
        return 0

    def metadata(self) -> dict[str, Any]:
        return {
            "state": self.state.name,
            "token_count": self.token_count,
            "threshold": self._threshold,
            "memory_bytes": self.memory_bytes(),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _convert(self) -> None:
        """Bulk convert FP16 cache to Polar packed."""
        if self._fp16_keys is None or self._fp16_values is None:
            self.state = CacheState.FALLBACK
            return

        try:
            self.state = CacheState.CONVERTING

            # Quantize full cache
            B, Hkv, T, D = self._fp16_keys.shape

            # Flatten and quantize
            keys_flat = self._fp16_keys.reshape(-1, D)
            values_flat = self._fp16_values.reshape(-1, D)

            key_qv = self.key_q.quantize(keys_flat)
            value_qv = self.value_q.quantize(values_flat)

            # Pack indices
            key_packed = pack_indices(key_qv.indices.reshape(B, Hkv, T, D), self.cfg.key_bits)
            value_packed = pack_indices(value_qv.indices.reshape(B, Hkv, T, D), self.cfg.value_bits)

            # Create PolarCache
            self._polar_cache = PolarCache(
                config=self.cfg,
                batch_size=B,
                num_kv_heads=Hkv,
                head_dim=D,
                block_size=self.cfg.allocation_block_tokens,
            )

            # Write converted data
            self._polar_cache.state = type("State", (), {
                "key_indices": key_packed,
                "key_norms": key_qv.norms.reshape(B, Hkv, T),
                "value_indices": value_packed,
                "value_norms": value_qv.norms.reshape(B, Hkv, T),
                "offset": T,
                "capacity": T,
            })()

            # Release FP16
            self._fp16_keys = None
            self._fp16_values = None

            self.state = CacheState.POLAR_PACKED

        except Exception:
            self.state = CacheState.FALLBACK

    def _append_packed(self, key: Any, value: Any) -> None:
        """Append a single token to the Polar packed cache."""
        B, Hkv, _, D = key.shape

        key_qv = self.key_q.quantize(key.reshape(-1, D))
        value_qv = self.value_q.quantize(value.reshape(-1, D))

        key_packed = pack_indices(key_qv.indices.reshape(B, Hkv, 1, D), self.cfg.key_bits)
        value_packed = pack_indices(value_qv.indices.reshape(B, Hkv, 1, D), self.cfg.value_bits)

        if self._polar_cache is None:
            self._polar_cache = PolarCache(
                config=self.cfg,
                batch_size=B,
                num_kv_heads=Hkv,
                head_dim=D,
                block_size=self.cfg.allocation_block_tokens,
            )

        self._polar_cache.append(
            key_packed,
            key_qv.norms.reshape(B, Hkv, 1),
            value_packed,
            value_qv.norms.reshape(B, Hkv, 1),
        )

    def _reset(self) -> None:
        self.state = CacheState.EMPTY
        self.token_count = 0
        self._fp16_keys = None
        self._fp16_values = None
        self._polar_cache = None
