#!/usr/bin/env python3
"""
RFSN v10 — Regression tests for teacher_forced_step_trace.json.

Validates that:
1. The artifact is populated (not a placeholder).
2. Every non-error row has the required schema fields.
3. baseline_fp16 rows are exact identity (cosine=1.0, kl=0.0).
4. Stable configs (k8_v5_gs64, k8_v5_gs32) pass all teacher-forced
   steps with cosine >= 0.99.
5. The prefill_decode_split.json reconciliation fields are present.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_EXP_DIR = Path("artifacts/proof/experimental")

_REQUIRED_FIELDS = {
    "config",
    "prompt_tokens",
    "step",
    "forced_token_id",
    "continuation_mode",
    "kv_len_before",
    "kv_len_after",
    "position_id",
    "cache_position",
    "logit_cosine_vs_fp16",
    "top5_overlap_vs_fp16",
    "kl_vs_fp16",
    "max_abs_logit_delta",
    "mean_abs_logit_delta",
    "argmax_fp16_token_id",
    "argmax_quant_token_id",
    "rank_of_fp16_argmax_in_quant",
    "logprob_forced_token_fp16",
    "logprob_forced_token_quant",
    "logprob_forced_token_delta",
    "entropy_fp16",
    "entropy_quant",
    "entropy_delta",
    "status",
}


def test_teacher_forced_step_trace_not_placeholder():
    """teacher_forced_step_trace.json must be executed and non-empty."""
    path = _EXP_DIR / "teacher_forced_step_trace.json"
    assert path.exists(), f"{path} does not exist"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("status") not in {"awaiting_execution", "placeholder"}, (
        f"teacher_forced_step_trace.json is a placeholder: "
        f"status={data.get('status')}"
    )
    assert data.get("traces"), (
        "teacher_forced_step_trace.json has empty traces list"
    )


def test_teacher_forced_step_trace_row_schema():
    """Every non-error row must have required fields."""
    path = _EXP_DIR / "teacher_forced_step_trace.json"
    if not path.exists():
        pytest.skip("teacher_forced_step_trace.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    if not traces:
        pytest.skip("no traces")
    for i, row in enumerate(traces):
        if "error" in row:
            continue
        missing = _REQUIRED_FIELDS - set(row)
        assert not missing, (
            f"teacher_forced_step_trace row {i} missing: {sorted(missing)}"
        )


def test_teacher_forced_step_trace_baseline_is_identity():
    """baseline_fp16 rows must have cosine=1.0, kl=0.0."""
    path = _EXP_DIR / "teacher_forced_step_trace.json"
    if not path.exists():
        pytest.skip("teacher_forced_step_trace.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    baseline_rows = [
        r for r in traces
        if r.get("config") == "baseline_fp16" and "error" not in r
    ]
    if not baseline_rows:
        pytest.skip("no baseline_fp16 rows")
    for r in baseline_rows:
        cosine = r.get("logit_cosine_vs_fp16", 0.0)
        kl = r.get("kl_vs_fp16", 999.0)
        assert abs(cosine - 1.0) < 1e-4, (
            f"baseline_fp16 step {r['step']} cosine {cosine:.6f} != 1.0"
        )
        assert abs(kl) < 1e-4, (
            f"baseline_fp16 step {r['step']} kl {kl:.6f} != 0.0"
        )


def test_teacher_forced_step_trace_stable_configs_pass_all_steps():
    """Stable configs must pass all teacher-forced steps (cosine >= 0.99)."""
    path = _EXP_DIR / "teacher_forced_step_trace.json"
    if not path.exists():
        pytest.skip("teacher_forced_step_trace.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    if not traces:
        pytest.skip("no traces")
    stable_configs = {"k8_v5_gs64", "k8_v5_gs32"}
    failures = []
    for r in traces:
        if "error" in r:
            continue
        if r.get("config") not in stable_configs:
            continue
        cosine = r.get("logit_cosine_vs_fp16", 0.0)
        if cosine < 0.99:
            failures.append(
                f"{r['config']} @ {r['prompt_tokens']}t "
                f"step={r['step']} cosine={cosine:.6f}"
            )
    assert not failures, (
        "Stable config teacher-forced steps failed (cosine < 0.99):\n"
        + "\n".join(failures[:10])
    )


def test_teacher_forced_step_trace_continuation_mode_is_teacher_forced():
    """All rows must have continuation_mode=teacher_forced."""
    path = _EXP_DIR / "teacher_forced_step_trace.json"
    if not path.exists():
        pytest.skip("teacher_forced_step_trace.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    for i, r in enumerate(traces):
        if "error" in r:
            continue
        mode = r.get("continuation_mode")
        assert mode == "teacher_forced", (
            f"Row {i} continuation_mode={mode!r}, expected 'teacher_forced'"
        )


def test_prefill_decode_split_has_reconciliation_fields():
    """prefill_decode_split.json rows must have continuation_mode and token_sequence_source."""
    path = _EXP_DIR / "prefill_decode_split.json"
    if not path.exists():
        pytest.skip("prefill_decode_split.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        pytest.skip("no results in prefill_decode_split.json")
    for i, r in enumerate(results):
        if r.get("status") == "error":
            continue
        assert "continuation_mode" in r, (
            f"prefill_decode_split result {i} missing continuation_mode"
        )
        assert "token_sequence_source" in r, (
            f"prefill_decode_split result {i} missing token_sequence_source"
        )
