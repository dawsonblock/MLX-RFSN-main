#!/usr/bin/env python3
"""Split sparse/quant adaptive sparsity controller tests."""

from __future__ import annotations

from rfsn_v10.adaptive_sparsity import AdaptiveSparsityController


def test_sparse_degradation_increases_top_k() -> None:
    ctl = AdaptiveSparsityController(initial_top_k_ratio=0.5, increase_step_size=0.2)

    before = ctl.get_decision(model_id="m", layer_id="l", skill_pattern="s", seq_len=1024)
    after = ctl.update(
        sparse_success=True,
        fallback_used=False,
        sparse_audit_cosine=0.85,
        sparse_audit_rel_mae=0.08,
        quant_audit_cosine=0.99,
        quant_audit_rel_mae=0.01,
        model_id="m",
        layer_id="l",
        skill_pattern="s",
        seq_len=1024,
    )

    assert after.top_k_ratio > before.top_k_ratio
    assert after.reason == "sparse_quality_degraded"


def test_quant_degradation_does_not_falsely_blame_sparse() -> None:
    ctl = AdaptiveSparsityController(initial_top_k_ratio=0.5)

    decision = ctl.update(
        sparse_success=True,
        fallback_used=False,
        sparse_audit_cosine=0.99,
        sparse_audit_rel_mae=0.01,
        quant_audit_cosine=0.90,
        quant_audit_rel_mae=0.09,
        model_id="m",
        layer_id="l",
        skill_pattern="s",
        seq_len=512,
    )

    assert decision.reason == "quant_quality_degraded"
    assert decision.disable_sparse is False


def test_repeated_clean_sparse_audits_reduce_top_k() -> None:
    ctl = AdaptiveSparsityController(
        initial_top_k_ratio=0.8,
        decrease_step_size=0.1,
        stabilization_steps=2,
    )

    first = ctl.update(
        sparse_success=True,
        fallback_used=False,
        sparse_audit_cosine=0.995,
        sparse_audit_rel_mae=0.01,
        quant_audit_cosine=0.995,
        quant_audit_rel_mae=0.01,
        model_id="m",
        layer_id="l",
        skill_pattern="s",
        seq_len=2048,
    )
    second = ctl.update(
        sparse_success=True,
        fallback_used=False,
        sparse_audit_cosine=0.996,
        sparse_audit_rel_mae=0.01,
        quant_audit_cosine=0.996,
        quant_audit_rel_mae=0.01,
        model_id="m",
        layer_id="l",
        skill_pattern="s",
        seq_len=2048,
    )

    assert second.top_k_ratio < first.top_k_ratio
    assert second.reason == "stable_clean_reduce_topk"


def test_fallback_disables_sparse_for_profile() -> None:
    ctl = AdaptiveSparsityController(initial_top_k_ratio=0.5)

    decision = ctl.update(
        sparse_success=False,
        fallback_used=True,
        sparse_audit_cosine=0.70,
        sparse_audit_rel_mae=0.20,
        quant_audit_cosine=0.99,
        quant_audit_rel_mae=0.01,
        model_id="m",
        layer_id="l",
        skill_pattern="s",
        seq_len=4096,
    )

    assert decision.disable_sparse is True
    snapshot = ctl.get_decision(model_id="m", layer_id="l", skill_pattern="s", seq_len=4096)
    assert snapshot.disable_sparse is True
