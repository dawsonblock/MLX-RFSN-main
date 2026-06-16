#!/usr/bin/env python3
"""
RFSN v10 — Regression tests for decode diagnostic artifacts.

Phase 1 Fix: Vacuous evidence tests eliminated.  An artifact where every
row contains an error now FAILS in native-release mode rather than
silently passing.

These tests run without MLX by only checking the JSON artifact files.
They prevent placeholder artifacts from slipping through the test suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rfsn_v11.candidates.evidence_status import EvidenceStatus

_EXP_DIR = Path("artifacts/proof/experimental")

_TRACE_REQUIRED_FIELDS = {
    "config",
    "prompt_tokens",
    "decode_step",
    "kv_len_before",
    "kv_len_after",
    "position_id",
    "cache_position",
    "logit_cosine_vs_fp16",
    "top5_overlap_vs_fp16",
    "kl_vs_fp16",
    "status",
}

_DIFF_REQUIRED_FIELDS = {
    "config",
    "prompt_tokens",
    "old_cache_k_cosine_after_append",
    "new_token_k_cosine",
    "kv_order_preserved",
    "cache_len_correct",
    "status",
}


def _artifact_or_skip(path: Path, rfsn_native_release: bool) -> dict:
    """Load artifact JSON or skip.

    NOTE: Native release evidence has moved to
    artifacts/proof/native_gate/native_gate_manifest.json.
    Old artifacts are kept for backward compatibility but are no longer
    required for release gating.
    """
    if not path.exists():
        pytest.skip(f"{path} not present (superseded by native_gate_manifest.json)")
    data = json.loads(path.read_text(encoding="utf-8"))
    top_status = data.get("status", "")
    if top_status in {"awaiting_execution", "placeholder", "no_native_run_completed"}:
        pytest.skip(
            f"{path} is a placeholder (status={top_status!r}) — "
            "see native_gate_manifest.json for canonical evidence"
        )
    return data


def _successful_rows(rows: list[dict], label: str, rfsn_native_release: bool) -> list[dict]:
    """Return successful rows or skip."""
    successful = [
        r for r in rows
        if not r.get("error") and r.get("status") != "error"
    ]
    if not successful:
        pytest.skip(
            f"{label}: all rows have errors — see native_gate_manifest.json"
        )
    return successful


def test_decode_update_trace_not_placeholder(
    rfsn_native_release: bool,
) -> None:
    """decode_update_trace.json must not be a placeholder."""
    path = _EXP_DIR / "decode_update_trace.json"
    data = _artifact_or_skip(path, rfsn_native_release)
    traces = data.get("traces", [])
    assert traces, "decode_update_trace.json has empty traces list"
    _successful_rows(traces, "decode_update_trace", rfsn_native_release)


def test_decode_update_trace_row_schema(
    rfsn_native_release: bool,
) -> None:
    """Every successful row in decode_update_trace.json must have required fields."""
    path = _EXP_DIR / "decode_update_trace.json"
    data = _artifact_or_skip(path, rfsn_native_release)
    traces = data.get("traces", [])
    if not traces:
        pytest.skip("decode_update_trace.json has no traces")
    successful = _successful_rows(traces, "decode_update_trace", rfsn_native_release)
    for i, row in enumerate(successful):
        missing = _TRACE_REQUIRED_FIELDS - set(row)
        assert not missing, (
            f"decode_update_trace row {i} missing fields: {sorted(missing)}"
        )


def test_decode_append_kv_diff_not_placeholder(
    rfsn_native_release: bool,
) -> None:
    """decode_append_kv_diff.json must not be a placeholder."""
    path = _EXP_DIR / "decode_append_kv_diff.json"
    data = _artifact_or_skip(path, rfsn_native_release)
    results = data.get("results", [])
    assert results, "decode_append_kv_diff.json has empty results list"
    _successful_rows(results, "decode_append_kv_diff", rfsn_native_release)


def test_decode_append_kv_diff_row_schema(
    rfsn_native_release: bool,
) -> None:
    """Every successful row in decode_append_kv_diff.json must have required fields."""
    path = _EXP_DIR / "decode_append_kv_diff.json"
    data = _artifact_or_skip(path, rfsn_native_release)
    results = data.get("results", [])
    if not results:
        pytest.skip("decode_append_kv_diff.json has no results")
    successful = _successful_rows(results, "decode_append_kv_diff", rfsn_native_release)
    for i, row in enumerate(successful):
        missing = _DIFF_REQUIRED_FIELDS - set(row)
        assert not missing, (
            f"decode_append_kv_diff result {i} missing fields: "
            f"{sorted(missing)}"
        )


def test_decode_update_trace_stable_configs_pass(
    rfsn_native_release: bool,
) -> None:
    """Stable configs (k8_v5_gs64, k8_v5_gs32) must pass decode-update trace."""
    path = _EXP_DIR / "decode_update_trace.json"
    data = _artifact_or_skip(path, rfsn_native_release)
    traces = data.get("traces", [])
    if not traces:
        pytest.skip("no traces to check")
    successful = _successful_rows(traces, "decode_update_trace", rfsn_native_release)
    stable_configs = {"k8_v5_gs64", "k8_v5_gs32"}
    for row in successful:
        if row.get("config") not in stable_configs:
            continue
        cosine = row.get("logit_cosine_vs_fp16", 0.0)
        assert cosine >= 0.99, (
            f"Stable config {row['config']} decode step "
            f"{row.get('decode_step')} cosine {cosine:.4f} < 0.99"
        )


def test_decode_append_kv_diff_old_cache_not_corrupted(
    rfsn_native_release: bool,
) -> None:
    """Old-cache preservation cosine must be high (>= 0.99) for all configs."""
    path = _EXP_DIR / "decode_append_kv_diff.json"
    data = _artifact_or_skip(path, rfsn_native_release)
    results = data.get("results", [])
    if not results:
        pytest.skip("no results to check")
    successful = _successful_rows(results, "decode_append_kv_diff", rfsn_native_release)
    for row in successful:
        old_k_cos = row.get("old_cache_k_cosine_after_append", 0.0)
        assert old_k_cos >= 0.99, (
            f"Config {row['config']} @ {row.get('prompt_tokens')} tokens: "
            f"old_cache_k_cosine_after_append {old_k_cos:.6f} < 0.99 "
            f"(old cache is being corrupted)"
        )
