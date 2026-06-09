#!/usr/bin/env python3
"""Main12 sparse safety gate tests."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.adaptive_sparsity import AdaptiveSparsityController
from rfsn_v10.kv_manager import RFSNTurboQuantKVManager
from rfsn_v10 import RFSNRuntime


def test_sparse_disabled_by_default(tmp_path) -> None:
    manager = RFSNTurboQuantKVManager(cache_dir=str(tmp_path))
    runtime = RFSNRuntime(kv_manager=manager, model_id="m", top_k_ratio=0.5)

    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    _, info = runtime.execute_decode_step(
        skill_pattern="safety_default",
        layer_id="l0",
        batch_id="b0",
        queries=q,
        keys=k,
        values=v,
    )

    assert info["sparse_enabled"] is False
    assert info["sparse_gate_reason"] == "disabled_by_default"


def test_bad_sparse_audit_disables_sparse() -> None:
    ctl = AdaptiveSparsityController(initial_top_k_ratio=0.5)

    decision = ctl.update(
        sparse_success=True,
        fallback_used=False,
        sparse_audit_cosine=0.80,
        sparse_audit_rel_mae=0.10,
        quant_audit_cosine=0.99,
        quant_audit_rel_mae=0.01,
        model_id="m",
        layer_id="l",
        skill_pattern="s",
        seq_len=1024,
    )

    assert decision.disable_sparse is True


def test_clean_audit_streak_can_lower_topk() -> None:
    ctl = AdaptiveSparsityController(
        initial_top_k_ratio=0.8,
        decrease_step_size=0.1,
        stabilization_steps=2,
    )

    first = ctl.update(
        sparse_success=True,
        fallback_used=False,
        sparse_audit_cosine=0.99,
        sparse_audit_rel_mae=0.01,
        quant_audit_cosine=0.99,
        quant_audit_rel_mae=0.01,
        model_id="m",
        layer_id="l",
        skill_pattern="s",
        seq_len=1024,
    )
    second = ctl.update(
        sparse_success=True,
        fallback_used=False,
        sparse_audit_cosine=0.99,
        sparse_audit_rel_mae=0.01,
        quant_audit_cosine=0.99,
        quant_audit_rel_mae=0.01,
        model_id="m",
        layer_id="l",
        skill_pattern="s",
        seq_len=1024,
    )

    assert second.top_k_ratio < first.top_k_ratio


def test_unsafe_profile_forces_dense(tmp_path) -> None:
    manager = RFSNTurboQuantKVManager(cache_dir=str(tmp_path))
    ctl = AdaptiveSparsityController(initial_top_k_ratio=0.5)

    ctl.update(
        sparse_success=False,
        fallback_used=True,
        sparse_audit_cosine=0.80,
        sparse_audit_rel_mae=0.10,
        quant_audit_cosine=0.99,
        quant_audit_rel_mae=0.01,
        model_id="m",
        layer_id="l0",
        skill_pattern="unsafe",
        seq_len=128,
    )

    runtime = RFSNRuntime(
        kv_manager=manager,
        model_id="m",
        top_k_ratio=0.5,
        enable_sparse_decode=True,
        adaptive_sparsity_controller=ctl,
    )

    q = mx.random.normal((1, 4, 1, 64))
    k = mx.random.normal((1, 4, 128, 64))
    v = mx.random.normal((1, 4, 128, 64))

    _, info = runtime.execute_decode_step(
        skill_pattern="unsafe",
        layer_id="l0",
        batch_id="b0",
        queries=q,
        keys=k,
        values=v,
    )

    assert info["sparse_enabled"] is False
    assert info["execution_mode"].startswith("dense")
