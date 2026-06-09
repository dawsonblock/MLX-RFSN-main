"""RFSN v10 — Live auto-adjusting quantisation controller tests."""

from __future__ import annotations

import pytest

from rfsn_v10.runtime.adaptive_controller import AdaptiveQuantController
from rfsn_v10.runtime.audit import AuditMetrics


class TestAdaptiveQuantController:
    """Unit tests for the live auto-adjust controller."""

    def test_init_defaults(self):
        c = AdaptiveQuantController()
        assert c.current_k_bits == 8
        assert c.current_v_bits == 5
        assert c.group_size == 64
        assert c._history == []

    def test_get_effective_bits(self):
        c = AdaptiveQuantController()
        assert c.get_effective_bits() == (8, 5, 64)

    def test_compute_margin_all_safe(self):
        c = AdaptiveQuantController()
        m = AuditMetrics(
            logit_cosine=0.9999,
            top5_overlap=0.98,
            kl_divergence=0.0001,
        )
        margin = c._compute_margin(m)
        # cosine_margin = 0.0009, kl_margin = 0.0009, top5_margin = 0.03
        assert margin == pytest.approx(0.0009, abs=1e-6)

    def test_compute_margin_drift(self):
        c = AdaptiveQuantController()
        m = AuditMetrics(
            logit_cosine=0.998,
            top5_overlap=0.98,
            kl_divergence=0.0001,
        )
        margin = c._compute_margin(m)
        # cosine_margin = -0.001 (drift)
        assert margin == pytest.approx(-0.001, abs=1e-6)

    def test_lower_bits_decreases_k_first(self):
        c = AdaptiveQuantController(current_k_bits=8, current_v_bits=5)
        changed = c._lower_bits()
        assert changed is True
        assert c.current_k_bits == 7
        assert c.current_v_bits == 5
        assert len(c.adjustment_events) == 1
        assert c.adjustment_events[0]["action"] == "lower_k"

    def test_lower_bits_alternates_to_v(self):
        c = AdaptiveQuantController(current_k_bits=6, current_v_bits=5)
        # k >= v, so lower k
        changed = c._lower_bits()
        assert changed is True
        assert c.current_k_bits == 5
        # Now k == v, lower k again (because k >= v)
        changed = c._lower_bits()
        assert changed is True
        assert c.current_k_bits == 4
        # Now k < v, so lower v
        changed = c._lower_bits()
        assert changed is True
        assert c.current_v_bits == 4

    def test_lower_bits_honours_floor(self):
        c = AdaptiveQuantController(current_k_bits=2, current_v_bits=2)
        changed = c._lower_bits()
        assert changed is False
        assert c.current_k_bits == 2
        assert c.current_v_bits == 2

    def test_raise_bits_raises_lower_first(self):
        c = AdaptiveQuantController(current_k_bits=4, current_v_bits=5)
        # k < v, so raise k first
        changed = c._raise_bits()
        assert changed is True
        assert c.current_k_bits == 5
        assert c.adjustment_events[-1]["action"] == "raise_k"

    def test_raise_bits_honours_ceiling(self):
        c = AdaptiveQuantController(current_k_bits=8, current_v_bits=8)
        changed = c._raise_bits()
        assert changed is False
        assert c.current_k_bits == 8
        assert c.current_v_bits == 8

    def test_update_no_adjustment_before_interval(self):
        c = AdaptiveQuantController(adjustment_interval=32)
        m = AuditMetrics(
            logit_cosine=0.9999,
            top5_overlap=0.98,
            kl_divergence=0.0001,
        )
        changed = c.update(m, step_num=1)
        assert changed is False
        assert c.current_k_bits == 8

    def test_update_lowers_after_sustained_good(self):
        c = AdaptiveQuantController(
            adjustment_interval=1,
            quality_window_size=2,
            improvement_margin=0.0005,
            current_k_bits=8,
            current_v_bits=5,
        )
        m = AuditMetrics(
            logit_cosine=0.99995,
            top5_overlap=0.98,
            kl_divergence=0.00005,
        )
        # First two audited steps fill the window.
        changed = c.update(m, step_num=1)
        assert changed is False
        changed = c.update(m, step_num=2)
        assert changed is False
        # Third step: consecutive_good now reaches window size.
        changed = c.update(m, step_num=3)
        assert changed is True
        assert c.current_k_bits == 7

    def test_update_raises_on_drift(self):
        c = AdaptiveQuantController(
            adjustment_interval=1,
            recovery_margin=0.0005,
            current_k_bits=4,
            current_v_bits=3,
        )
        # Slightly good step
        good = AuditMetrics(
            logit_cosine=0.999,
            top5_overlap=0.95,
            kl_divergence=0.001,
        )
        c.update(good, step_num=1)
        # Bad step: cosine drops below threshold
        bad = AuditMetrics(
            logit_cosine=0.998,
            top5_overlap=0.95,
            kl_divergence=0.001,
        )
        # First bad step triggers consecutive_bad=1, no raise yet
        changed = c.update(bad, step_num=2)
        assert changed is False
        # Second bad step triggers raise
        changed = c.update(bad, step_num=3)
        assert changed is True
        assert c.current_v_bits == 4  # v is lower, raised first

    def test_state_dict(self):
        c = AdaptiveQuantController(current_k_bits=6, current_v_bits=4)
        d = c.state_dict()
        assert d["current_k_bits"] == 6
        assert d["current_v_bits"] == 4
        assert d["group_size"] == 64
        assert "adjustment_event_count" in d

    def test_history_pruning(self):
        c = AdaptiveQuantController(quality_window_size=2)
        m = AuditMetrics(
            logit_cosine=0.9999,
            top5_overlap=0.98,
            kl_divergence=0.0001,
        )
        c.update(m, step_num=1)
        c.update(m, step_num=2)
        c.update(m, step_num=3)
        assert len(c._history) == 2
        assert c._history[0][0] == 2
        assert c._history[1][0] == 3

    def test_nan_inf_immediate_margin_negative(self):
        c = AdaptiveQuantController()
        m = AuditMetrics(
            logit_cosine=1.0,
            top5_overlap=1.0,
            kl_divergence=0.0,
            has_nan_inf=True,
        )
        margin = c._compute_margin(m)
        assert margin == -1.0  # Force immediate recovery
