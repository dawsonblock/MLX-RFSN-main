#!/usr/bin/env python3
"""
RFSN v10 — Regression tests for decode diagnostic artifacts.

These tests run without MLX by only checking the JSON artifact files.
They prevent placeholder artifacts from slipping through the test suite.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_decode_update_trace_not_placeholder():
    """decode_update_trace.json must not be a placeholder."""
    path = _EXP_DIR / "decode_update_trace.json"
    assert path.exists(), f"{path} does not exist"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("status") != "awaiting_execution", (
        "decode_update_trace.json is still a placeholder "
        "(status=awaiting_execution)"
    )
    assert data.get("status") not in {"placeholder", "awaiting_execution"}, (
        f"decode_update_trace.json has placeholder status: "
        f"{data.get('status')}"
    )
    assert data.get("traces"), (
        "decode_update_trace.json has empty traces list"
    )


def test_decode_update_trace_row_schema():
    """Every non-error row in decode_update_trace.json must have required fields."""
    path = _EXP_DIR / "decode_update_trace.json"
    if not path.exists():
        pytest.skip("decode_update_trace.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    if not traces:
        pytest.skip("decode_update_trace.json has no traces")
    for i, row in enumerate(traces):
        if "error" in row:
            continue
        missing = _TRACE_REQUIRED_FIELDS - set(row)
        assert not missing, (
            f"decode_update_trace row {i} missing fields: {sorted(missing)}"
        )


def test_decode_append_kv_diff_not_placeholder():
    """decode_append_kv_diff.json must not be a placeholder."""
    path = _EXP_DIR / "decode_append_kv_diff.json"
    assert path.exists(), f"{path} does not exist"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data.get("status") != "awaiting_execution", (
        "decode_append_kv_diff.json is still a placeholder "
        "(status=awaiting_execution)"
    )
    assert data.get("status") not in {"placeholder", "awaiting_execution"}, (
        f"decode_append_kv_diff.json has placeholder status: "
        f"{data.get('status')}"
    )
    assert data.get("results"), (
        "decode_append_kv_diff.json has empty results list"
    )


def test_decode_append_kv_diff_row_schema():
    """Every non-error row in decode_append_kv_diff.json must have required fields."""
    path = _EXP_DIR / "decode_append_kv_diff.json"
    if not path.exists():
        pytest.skip("decode_append_kv_diff.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        pytest.skip("decode_append_kv_diff.json has no results")
    for i, row in enumerate(results):
        if "error" in row:
            continue
        missing = _DIFF_REQUIRED_FIELDS - set(row)
        assert not missing, (
            f"decode_append_kv_diff result {i} missing fields: "
            f"{sorted(missing)}"
        )


def test_decode_update_trace_stable_configs_pass():
    """Stable configs (k8_v5_gs64, k8_v5_gs32) must pass decode-update trace."""
    path = _EXP_DIR / "decode_update_trace.json"
    if not path.exists():
        pytest.skip("decode_update_trace.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    traces = data.get("traces", [])
    if not traces:
        pytest.skip("no traces to check")
    stable_configs = {"k8_v5_gs64", "k8_v5_gs32"}
    for row in traces:
        if "error" in row:
            continue
        if row.get("config") not in stable_configs:
            continue
        cosine = row.get("logit_cosine_vs_fp16", 0.0)
        assert cosine >= 0.99, (
            f"Stable config {row['config']} decode step "
            f"{row.get('decode_step')} cosine {cosine:.4f} < 0.99"
        )


def test_decode_append_kv_diff_old_cache_not_corrupted():
    """Old-cache preservation cosine must be high (≥ 0.99) for all configs."""
    path = _EXP_DIR / "decode_append_kv_diff.json"
    if not path.exists():
        pytest.skip("decode_append_kv_diff.json not present")
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        pytest.skip("no results to check")
    for row in results:
        if "error" in row:
            continue
        old_k_cos = row.get("old_cache_k_cosine_after_append", 0.0)
        assert old_k_cos >= 0.99, (
            f"Config {row['config']} @ {row.get('prompt_tokens')} tokens: "
            f"old_cache_k_cosine_after_append {old_k_cos:.6f} < 0.99 "
            "(old cache is being corrupted)"
        )
