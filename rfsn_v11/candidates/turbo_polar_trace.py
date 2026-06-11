"""TurboPolar trace / instrumentation helpers.

Every TurboPolar run must emit an honest trace. If any counter is estimated,
promotion_eligible is forced to False.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurboPolarTrace:
    """Runtime trace for a single TurboPolar decode step or full generation."""

    cache_backend_used: str = ""
    real_cache_used: bool = False
    prefill_polar_encode_events: int = 0
    decode_polar_fetch_events: int = 0
    cache_bytes_written_actual: int = 0
    cache_bytes_read_actual: int = 0
    fallback_used: bool = False

    # Metal kernel trace
    metal_kernel_launched: bool = False
    kernel_name: str = ""
    kernel_fallback_reason: str = ""

    # QJL trace
    qjl_enabled: bool = False
    qjl_score_correction_applied: bool = False

    # Events log (append-only)
    events: list[str] = field(default_factory=list)

    # Metadata
    token_sequence_hash: str = ""
    methodology_status: str = "PENDING"
    promotion_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_backend_used": self.cache_backend_used,
            "real_cache_used": self.real_cache_used,
            "prefill_polar_encode_events": self.prefill_polar_encode_events,
            "decode_polar_fetch_events": self.decode_polar_fetch_events,
            "cache_bytes_written_actual": self.cache_bytes_written_actual,
            "cache_bytes_read_actual": self.cache_bytes_read_actual,
            "fallback_used": self.fallback_used,
            "metal_kernel_launched": self.metal_kernel_launched,
            "kernel_name": self.kernel_name,
            "kernel_fallback_reason": self.kernel_fallback_reason,
            "qjl_enabled": self.qjl_enabled,
            "qjl_score_correction_applied": self.qjl_score_correction_applied,
            "events": self.events,
            "token_sequence_hash": self.token_sequence_hash,
            "methodology_status": self.methodology_status,
            "promotion_allowed": self.promotion_allowed,
        }

    def mark_event(self, name: str) -> None:
        self.events.append(name)

    def validate_for_promotion(self) -> tuple[bool, str]:
        """Return (ok, reason) for promotion eligibility.

        If any required field is missing or estimated, returns False.
        """
        if not self.real_cache_used:
            return False, "real_cache_used is false"
        if self.cache_bytes_written_actual <= 0:
            return False, "cache_bytes_written_actual is zero or missing"
        if self.cache_bytes_read_actual <= 0:
            return False, "cache_bytes_read_actual is zero or missing"
        if self.fallback_used:
            return False, "fallback_used is true"
        if not self.token_sequence_hash:
            return False, "token_sequence_hash is empty"
        if self.methodology_status != "TEACHER_FORCED_RERUN_COMPLETE_NO_PROMOTION":
            return False, f"methodology_status is {self.methodology_status}"
        return True, ""
