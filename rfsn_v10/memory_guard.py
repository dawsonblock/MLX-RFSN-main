"""RFSN v10 Memory Pressure Safety module.

Monitors and enforces memory safety for MLX/Metal workloads by tracking
active GPU memory, estimated cache usage, and enforcing configurable
soft/hard limits with proactive eviction callbacks.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

from .compat import mx


class MemoryGuard:
    """
    Monitors and enforces memory safety for MLX/Metal workloads.

    Responsibilities:
    - Track active MLX/Metal memory (defensive wrapping for API availability)
    - Track estimated cache memory
    - Enforce safety margin
    - Trigger proactive eviction callbacks
    - Disable sparse/quantized mode under pressure
    - Protect against kernel panic risk from memory exhaustion
    """

    def __init__(
        self,
        safety_margin_gb: float = 0.5,
        soft_limit_gb: float | None = None,
        hard_limit_gb: float | None = None,
        eviction_callback: Callable[[int], int] | None = None,
    ):
        """
        Args:
            safety_margin_gb: Minimum free memory to maintain (GB).
            soft_limit_gb: Total memory soft limit (GB). Triggers eviction.
            hard_limit_gb: Total memory hard limit (GB). Triggers emergency mode.
            eviction_callback: Callable that evicts caches and returns bytes freed.
                It receives the target bytes to free.
        """
        self.safety_margin_gb = safety_margin_gb
        self.soft_limit_gb = soft_limit_gb
        self.hard_limit_gb = hard_limit_gb
        self.eviction_callback = eviction_callback
        self._pressure_active = False
        self._sparse_disabled = False
        self._quantized_disabled = False
        self._has_mlx_memory_api = self._check_mlx_memory_api()

    def _soft_limit_bytes(self) -> int | None:
        if self.soft_limit_gb is None:
            return None
        return int(self.soft_limit_gb * (1024 ** 3))

    def _hard_limit_bytes(self) -> int | None:
        if self.hard_limit_gb is None:
            return None
        return int(self.hard_limit_gb * (1024 ** 3))

    @staticmethod
    def _check_mlx_memory_api() -> bool:
        """Check if MLX memory introspection APIs are available."""
        try:
            if hasattr(mx, 'get_active_memory'):
                return True
            if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_active_memory'):
                return True
            return False
        except Exception:
            return False

    def get_active_memory_bytes(self) -> int:
        """Get current active GPU memory usage in bytes, or 0 if unavailable."""
        if not self._has_mlx_memory_api:
            return 0
        try:
            if hasattr(mx, 'get_active_memory'):
                return int(mx.get_active_memory())
            if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_active_memory'):
                return int(mx.metal.get_active_memory())
        except Exception:
            pass
        return 0

    def get_peak_memory_bytes(self) -> int:
        """Get peak GPU memory usage in bytes, or 0 if unavailable."""
        if not self._has_mlx_memory_api:
            return 0
        try:
            if hasattr(mx, 'get_peak_memory'):
                return int(mx.get_peak_memory())
            if hasattr(mx, 'metal') and hasattr(mx.metal, 'get_peak_memory'):
                return int(mx.metal.get_peak_memory())
        except Exception:
            pass
        return 0

    def check_pressure(self, estimated_cache_bytes: int = 0) -> bool:
        """
        Check if memory pressure is exceeded.

        Args:
            estimated_cache_bytes: Current estimated cache memory in bytes.

        Returns:
            True if memory pressure is detected.
        """
        active_bytes = self.get_active_memory_bytes()
        total_estimated = active_bytes + estimated_cache_bytes

        # Check hard limit
        if self.hard_limit_gb is not None:
            if total_estimated > self.hard_limit_gb * (1024 ** 3):
                self._pressure_active = True
                return True

        # Check soft limit
        if self.soft_limit_gb is not None:
            if total_estimated > self.soft_limit_gb * (1024 ** 3):
                self._pressure_active = True
                return True

        self._pressure_active = False
        return False

    def enforce_safety(self, estimated_cache_bytes: int = 0) -> int:
        """
        Enforce memory safety: trigger eviction if under pressure.

        Returns:
            Bytes freed by eviction (0 if no action taken).
        """
        if estimated_cache_bytes < 0:
            estimated_cache_bytes = 0

        active_bytes = self.get_active_memory_bytes()
        projected_active = active_bytes + int(estimated_cache_bytes)

        soft_limit_bytes = self._soft_limit_bytes()
        hard_limit_bytes = self._hard_limit_bytes()

        if hard_limit_bytes is not None and projected_active > hard_limit_bytes:
            self.enter_emergency_mode()
            return 0

        if soft_limit_bytes is None or projected_active <= soft_limit_bytes:
            self._pressure_active = False
            return 0

        self._pressure_active = True
        target_free_bytes = max(0, projected_active - soft_limit_bytes)
        # Under pressure, request eviction for at least the estimated cache footprint
        # to avoid under-evicting from byte-rounding at tiny configured limits.
        target_free_bytes = max(target_free_bytes, int(estimated_cache_bytes))
        bytes_freed = 0
        if self.eviction_callback is not None:
            try:
                bytes_freed = int(self.eviction_callback(target_free_bytes))
            except Exception as e:
                warnings.warn(f"Eviction callback failed: {e}")
                bytes_freed = 0

        if hard_limit_bytes is not None and projected_active - bytes_freed > hard_limit_bytes:
            self.enter_emergency_mode()

        return bytes_freed

    def should_disable_sparse(self) -> bool:
        """Return True if sparse attention should be disabled due to pressure."""
        return self._sparse_disabled or self._pressure_active

    def should_disable_quantized(self) -> bool:
        """Return True if quantized KV cache should be disabled due to pressure."""
        return self._quantized_disabled or self._pressure_active

    def enter_emergency_mode(self) -> None:
        """Disable sparse and quantized modes as a protective measure."""
        self._sparse_disabled = True
        self._quantized_disabled = True
        self._pressure_active = True
        warnings.warn("MemoryGuard: entered emergency mode — sparse and quantized modes disabled")

    def exit_emergency_mode(self) -> None:
        """Re-enable sparse and quantized modes."""
        self._sparse_disabled = False
        self._quantized_disabled = False
        self._pressure_active = False

    def get_status(self) -> dict:
        """Return current memory guard status."""
        return {
            "has_mlx_memory_api": self._has_mlx_memory_api,
            "active_memory_bytes": self.get_active_memory_bytes(),
            "peak_memory_bytes": self.get_peak_memory_bytes(),
            "pressure_active": self._pressure_active,
            "sparse_disabled": self._sparse_disabled,
            "quantized_disabled": self._quantized_disabled,
            "safety_margin_gb": self.safety_margin_gb,
            "soft_limit_gb": self.soft_limit_gb,
            "hard_limit_gb": self.hard_limit_gb,
        }
