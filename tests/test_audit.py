#!/usr/bin/env python3
"""
RFSN v10 — Audit mode acceptance tests.

Verifies drift detection thresholds, fallback recommendations,
and audit event logging without requiring a full model run.
"""
from __future__ import annotations

import json

import pytest

mx = pytest.importorskip("mlx.core")
from rfsn_v10.runtime.audit import (  # noqa: E402, I001
    AuditMetrics,
    AuditEvent,
    check_drift,
    log_audit_event,
    audit_decode_step,
)


# ------------------------------------------------------------------
# check_drift threshold tests
# ------------------------------------------------------------------

class TestCheckDrift:
    def test_no_drift_returns_none(self):
        m = AuditMetrics(
            logit_cosine=0.9995,
            top5_overlap=0.96,
            kl_divergence=0.0005,
            has_nan_inf=False,
        )
        assert check_drift(m) is None

    def test_nan_inf_triggers_fp16(self):
        m = AuditMetrics(
            logit_cosine=0.9995,
            top5_overlap=0.96,
            kl_divergence=0.0005,
            has_nan_inf=True,
        )
        assert check_drift(m) == "FP16"

    def test_low_cosine_triggers_fallback(self):
        m = AuditMetrics(
            logit_cosine=0.970,
            top5_overlap=0.80,
            kl_divergence=0.0005,
            has_nan_inf=False,
        )
        assert check_drift(m) == "k8_v5_gs64"

    def test_low_top5_triggers_fallback(self):
        m = AuditMetrics(
            logit_cosine=0.9995,
            top5_overlap=0.90,
            kl_divergence=0.0005,
            has_nan_inf=False,
        )
        assert check_drift(m) == "k8_v5_gs64"

    def test_high_kl_triggers_fallback(self):
        m = AuditMetrics(
            logit_cosine=0.9995,
            top5_overlap=0.96,
            kl_divergence=0.10,
            has_nan_inf=False,
        )
        assert check_drift(m) == "k8_v5_gs64"

    def test_priority_nan_over_cosine(self):
        """NaN/Inf has highest priority."""
        m = AuditMetrics(
            logit_cosine=0.500,
            top5_overlap=0.20,
            kl_divergence=1.0,
            has_nan_inf=True,
        )
        assert check_drift(m) == "FP16"

    def test_priority_cosine_over_kl(self):
        """Low cosine has higher priority than high KL."""
        m = AuditMetrics(
            logit_cosine=0.500,
            top5_overlap=0.20,
            kl_divergence=1.0,
            has_nan_inf=False,
        )
        assert check_drift(m) == "k8_v5_gs64"


# ------------------------------------------------------------------
# audit_decode_step interval tests
# ------------------------------------------------------------------

class TestAuditDecodeStep:
    def test_returns_none_when_not_due(self):
        comp = mx.random.normal((1, 4, 1, 64))
        ref = mx.random.normal((1, 4, 1, 64))
        result = audit_decode_step(comp, ref, step_num=3, audit_interval=10)
        assert result is None

    def test_returns_metrics_when_due(self, tmp_path):
        comp = mx.random.normal((1, 4, 1, 64))
        ref = mx.random.normal((1, 4, 1, 64))
        result = audit_decode_step(
            comp, ref, step_num=10, audit_interval=10,
            log_path=tmp_path / "audit.jsonl",
        )
        assert result is not None
        assert isinstance(result, AuditMetrics)
        assert -1.0 <= result.logit_cosine <= 1.0
        assert 0.0 <= result.top5_overlap <= 1.0
        assert result.step_num == 10

    def test_nan_input_triggers_fallback(self, tmp_path):
        comp = mx.array([[[[float("nan")]]]])
        ref = mx.zeros((1, 1, 1, 1))
        result = audit_decode_step(
            comp, ref, step_num=20, audit_interval=10,
            log_path=tmp_path / "audit.jsonl",
        )
        assert result is not None
        assert result.has_nan_inf is True


# ------------------------------------------------------------------
# Audit event logging tests
# ------------------------------------------------------------------

class TestLogAuditEvent:
    def test_log_appends_jsonl(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        event = AuditEvent(
            event_type="audit_decode_step",
            step_num=5,
            metrics={"cosine": 0.99},
            fallback_recommendation=None,
        )
        log_audit_event(event, log_path=path)
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.strip() != ""
        lines = text.strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["event_type"] == "audit_decode_step"
        assert record["step_num"] == 5
        assert record["metrics"]["cosine"] == 0.99

    def test_multiple_logs_append(self, tmp_path):
        path = tmp_path / "audit.jsonl"
        for i in range(3):
            event = AuditEvent(
                event_type="audit_decode_step",
                step_num=i,
                metrics={},
                fallback_recommendation=None,
            )
            log_audit_event(event, log_path=path)
        text = path.read_text(encoding="utf-8")
        assert text.strip() != ""
        lines = text.strip().split("\n")
        assert len(lines) == 3


class TestAuditDecodeStepExtras:
    def test_step_num_included_in_metrics(self, tmp_path):
        comp = mx.random.normal((1, 4, 1, 64))
        ref = mx.random.normal((1, 4, 1, 64))
        result = audit_decode_step(
            comp, ref, step_num=20, audit_interval=10,
            log_path=tmp_path / "audit.jsonl",
        )
        assert result is not None
        assert result.step_num == 20

    def test_labels_path_computes_nll_delta(self, tmp_path):
        comp = mx.random.normal((1, 4, 1, 64))
        ref = mx.random.normal((1, 4, 1, 64))
        labels = mx.array([0])
        result = audit_decode_step(
            comp, ref, step_num=10, audit_interval=10,
            labels=labels,
            log_path=tmp_path / "audit.jsonl",
        )
        assert result is not None
        assert result.nll_delta is not None
