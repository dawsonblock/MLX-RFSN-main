"""Candidate: TurboPolar — PolarQuant + optional QJL + optional fused Metal.

Build order enforced by the candidate itself:
  1. Offline PolarQuant encoder/decoder must pass attention-score gate.
  2. Offline QJL must improve score error (otherwise disabled).
  3. Teacher-forced logit comparison against baseline must pass.
  4. Runtime cache with real counters must be proven.
  5. Fused Metal kernel must match Python reference.
  6. Online softmax with dense V must match dense attention output.
  7. Only then is promotion even considered.

Default baseline remains rfsn_v10_k8_v5_gs64.
TurboPolar is EXPERIMENTAL and never the default.
"""
from __future__ import annotations

import time
from typing import Any

from .base import CandidateResult, KVCompressionCandidate
from .candidate_status import CandidateStatus, get_status_for_name
from .quality_gates import (
    GATE_STATUS_FAIL,
    GATE_STATUS_PASS,
    GATE_STATUS_PENDING_LOGIT_GATE,
    evaluate_quality_gate,
    logit_quality_metrics,
)
from .turbo_polar_config import TurboPolarConfig
from .turbo_polar_trace import TurboPolarTrace


class TurboPolarAdapter(KVCompressionCandidate):
    """TurboPolar candidate adapter.

    Parameters
    ----------
    config
        TurboPolarConfig instance. Defaults to the first experimental preset.
    """

    candidate_status = CandidateStatus.EXPERIMENTAL

    def __init__(self, config: TurboPolarConfig | None = None) -> None:
        self.cfg = config or TurboPolarConfig()
        self.name = self.cfg.candidate_name
        self._trace = TurboPolarTrace()

    def is_available(self) -> bool:
        try:
            import mlx.core as mx  # noqa: F401
            import mlx_lm  # noqa: F401
            return True
        except ImportError:
            return False

    def _build_trace(self) -> TurboPolarTrace:
        """Return the current trace, resetting internal state."""
        return self._trace

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> CandidateResult:
        """Run generation with TurboPolar.

        At this stage (Alpha 9) the runtime path is NOT yet wired to real
        model generation.  We return an honest OFFLINE_ONLY / EXPERIMENTAL
        result with the config embedded in notes.
        """
        if not self.is_available():
            return CandidateResult(
                name=self.name,
                model_id=getattr(model, "name_or_path", "unknown"),
                prompt=prompt,
                gate_status="ERROR",
                error="mlx or mlx_lm not importable",
            )

        t0 = time.perf_counter()

        # Phase 0/1/2: offline-only until real cache injection exists
        status = get_status_for_name(self.name)
        trace = self._build_trace()
        trace.cache_backend_used = "turbo_polar_k_only"
        trace.real_cache_used = False
        trace.fallback_used = True
        trace.methodology_status = "PENDING"
        trace.promotion_allowed = False
        trace.mark_event("run_called")
        trace.mark_event("offline_only_fallback")

        total_ms = (time.perf_counter() - t0) * 1000

        return CandidateResult(
            name=self.name,
            model_id=getattr(model, "name_or_path", "unknown"),
            prompt=prompt,
            total_ms=total_ms,
            tokens_per_sec=0.0,
            generated_tokens=0,
            generated_text="",
            gate_status=GATE_STATUS_PENDING_LOGIT_GATE
            if status != CandidateStatus.OFFLINE_ONLY
            else "OFFLINE_ONLY",
            promotion_eligible=False,
            candidate_status=status,
            cache_backend_used=trace.cache_backend_used,
            cache_events=trace.events,
            cache_bytes_written=trace.cache_bytes_written_actual,
            cache_bytes_read=trace.cache_bytes_read_actual,
            notes=f"TurboPolar config: {self.cfg}",
        )

    def capture_logprobs(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        target_text: str | None = None,
        max_tokens: int = 200,
        temp: float = 0.0,
    ) -> Any:
        """Teacher-forced logit capture — not yet implemented.

        Returns None so the shootout knows this candidate cannot yet
        participate in the full-logit gate.
        """
        return None
