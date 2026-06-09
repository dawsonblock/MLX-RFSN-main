"""RFSN v10 — Live auto-adjusting quantisation controller.

Observes per-step audit metrics and dynamically tunes ``k_bits``, ``v_bits``,
and ``group_size`` to stay within quality thresholds while maximising
compression (minimising bits).  The controller adds hysteresis and a
quality-history window to avoid oscillation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import AuditMetrics


@dataclass
class AdaptiveQuantController:
    """Controller that auto-adjusts quantisation parameters based on live
    quality feedback.

    The controller maintains a sliding window of recent audit metrics.
    From each metric it derives a *composite safety margin*:

    * ``cosine_margin = logit_cosine - min_logit_cosine``
    * ``kl_margin    = max_kl - kl_divergence``
    * ``top5_margin  = top5_overlap - min_top5_overlap``

    The *minimum* of the three margins is the step's safety margin.

    **Adjustment rules**

    1. If the window average margin ≥ ``improvement_margin`` for
       ``quality_window_size`` consecutive audited steps → decrease
       ``k_bits`` or ``v_bits`` by 1 (alternating).
    2. If any step's margin < ``recovery_margin`` → immediately increase
       the offending bit width by 1.
    3. Only adjust once every ``adjustment_interval`` decode steps.
    4. Never drop below ``min_k_bits`` / ``min_v_bits`` or exceed
       ``max_k_bits`` / ``max_v_bits``.
    """

    # --- quality thresholds (same semantics as audit) ---
    min_logit_cosine: float = 0.999
    max_kl: float = 0.001
    min_top5_overlap: float = 0.95

    # --- tuning knobs ---
    adjustment_interval: int = 32
    quality_window_size: int = 4
    improvement_margin: float = 0.002
    recovery_margin: float = 0.001

    # --- bit search space ---
    min_k_bits: int = 2
    min_v_bits: int = 2
    max_k_bits: int = 8
    max_v_bits: int = 8
    group_size: int = 64

    # --- current state ---
    current_k_bits: int = 8
    current_v_bits: int = 5
    _last_adjustment_step: int = 0
    _consecutive_good: int = 0
    _consecutive_bad: int = 0
    _history: list[tuple[int, float]] = field(default_factory=list)

    # --- telemetry ---
    adjustment_events: list[dict[str, Any]] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_effective_bits(self) -> tuple[int, int, int]:
        """Return the current *(k_bits, v_bits, group_size)*."""
        return self.current_k_bits, self.current_v_bits, self.group_size

    def update(self, metrics: AuditMetrics, step_num: int) -> bool:
        """Ingest a fresh :class:`AuditMetrics` and possibly adjust bits.

        Returns ``True`` if an adjustment was made this step.
        """
        margin = self._compute_margin(metrics)
        self._history.append((step_num, margin))

        # Prune history to the last ``quality_window_size`` audited steps.
        if len(self._history) > self.quality_window_size:
            self._history = self._history[-self.quality_window_size:]

        # Guard: only adjust once per ``adjustment_interval``.
        if step_num - self._last_adjustment_step < self.adjustment_interval:
            return False

        # Rule 2: immediate recovery if margin is negative.
        if margin < -self.recovery_margin:
            self._consecutive_bad += 1
            self._consecutive_good = 0
            if self._consecutive_bad >= 2:
                made = self._raise_bits()
                if made:
                    self._last_adjustment_step = step_num
                return made
            return False

        self._consecutive_bad = 0

        # Rule 1: try lower bits if window average is comfortably good.
        if len(self._history) < self.quality_window_size:
            return False

        avg_margin = sum(m for _, m in self._history) / len(self._history)
        if avg_margin >= self.improvement_margin:
            self._consecutive_good += 1
            if self._consecutive_good >= self.quality_window_size:
                made = self._lower_bits()
                if made:
                    self._last_adjustment_step = step_num
                return made
        else:
            self._consecutive_good = 0

        return False

    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of controller state."""
        return {
            "current_k_bits": self.current_k_bits,
            "current_v_bits": self.current_v_bits,
            "group_size": self.group_size,
            "min_logit_cosine": self.min_logit_cosine,
            "max_kl": self.max_kl,
            "min_top5_overlap": self.min_top5_overlap,
            "improvement_margin": self.improvement_margin,
            "recovery_margin": self.recovery_margin,
            "adjustment_interval": self.adjustment_interval,
            "quality_window_size": self.quality_window_size,
            "adjustment_event_count": len(self.adjustment_events),
        }

    def export_adjustments(
        self, path: Path | str, limit: int = 1000
    ) -> None:
        """Write the most recent *limit* adjustment events to JSONL."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            for evt in self.adjustment_events[-limit:]:
                fh.write(json.dumps(evt, default=str) + "\n")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_margin(self, metrics: AuditMetrics) -> float:
        """Composite safety margin (positive = safe, negative = drift)."""
        if metrics.has_nan_inf:
            return -1.0  # Force immediate recovery
        cosine_margin = metrics.logit_cosine - self.min_logit_cosine
        kl_margin = self.max_kl - metrics.kl_divergence
        top5_margin = metrics.top5_overlap - self.min_top5_overlap
        # The weakest link determines safety.
        return min(cosine_margin, kl_margin, top5_margin)

    def _lower_bits(self) -> bool:
        """Attempt to decrease one bit width.  Returns ``True`` on change."""
        # Alternate between lowering k and v to keep balance.
        if (
            self.current_k_bits > self.min_k_bits
            and self.current_k_bits >= self.current_v_bits
        ):
            old = self.current_k_bits
            self.current_k_bits = max(
                self.min_k_bits, self.current_k_bits - 1
            )
            if self.current_k_bits != old:
                self._log_event("lower_k", old, self.current_k_bits)
                return True

        if self.current_v_bits > self.min_v_bits:
            old = self.current_v_bits
            self.current_v_bits = max(
                self.min_v_bits, self.current_v_bits - 1
            )
            if self.current_v_bits != old:
                self._log_event("lower_v", old, self.current_v_bits)
                return True

        return False

    def _raise_bits(self) -> bool:
        """Attempt to increase one bit width.  Returns ``True`` on change."""
        # Raise the bit width that is currently lower first (it is the
        # more likely culprit of drift).
        if self.current_k_bits < self.current_v_bits:
            first, second = "k", "v"
        else:
            first, second = "v", "k"

        if first == "k" and self.current_k_bits < self.max_k_bits:
            old = self.current_k_bits
            self.current_k_bits = min(
                self.max_k_bits, self.current_k_bits + 1
            )
            if self.current_k_bits != old:
                self._log_event("raise_k", old, self.current_k_bits)
                return True

        if first == "v" and self.current_v_bits < self.max_v_bits:
            old = self.current_v_bits
            self.current_v_bits = min(
                self.max_v_bits, self.current_v_bits + 1
            )
            if self.current_v_bits != old:
                self._log_event("raise_v", old, self.current_v_bits)
                return True

        # Try the other one if the first did not change.
        if second == "k" and self.current_k_bits < self.max_k_bits:
            old = self.current_k_bits
            self.current_k_bits = min(
                self.max_k_bits, self.current_k_bits + 1
            )
            if self.current_k_bits != old:
                self._log_event("raise_k", old, self.current_k_bits)
                return True

        if second == "v" and self.current_v_bits < self.max_v_bits:
            old = self.current_v_bits
            self.current_v_bits = min(
                self.max_v_bits, self.current_v_bits + 1
            )
            if self.current_v_bits != old:
                self._log_event("raise_v", old, self.current_v_bits)
                return True

        return False

    def _log_event(self, action: str, old: int, new: int) -> None:
        self.adjustment_events.append(
            {
                "timestamp": time.time(),
                "action": action,
                "old_bits": old,
                "new_bits": new,
                "current_k_bits": self.current_k_bits,
                "current_v_bits": self.current_v_bits,
            }
        )
