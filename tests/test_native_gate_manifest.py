#!/usr/bin/env python3
"""Canonical regression tests for the native gate manifest.

This is the single source of truth for native-release evidence.
All old artifact tests (teacher_forced_step_trace, decode_update_trace,
decode_append_kv_diff) are superseded by this manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_MANIFEST_PATH = Path("artifacts/proof/native_gate/native_gate_manifest.json")

_REQUIRED_MANIFEST_KEYS = {
    "candidate",
    "model_id",
    "config",
    "backend_report",
    "runs",
    "timestamp_utc",
    "status",
    "exit_code",
    "violations",
}

_REQUIRED_RUN_KEYS = {
    "context_length",
    "prompt_token_hash",
    "prompt_tokens_actual",
    "dense",
    "eight_bit",
    "packed",
    "token_match",
    "quality",
}

_REQUIRED_RESULT_KEYS = {
    "model_id",
    "prompt",
    "prompt_tokens",
    "generated_tokens",
    "generated_text",
    "elapsed_ms",
    "token_sequence_hash",
    "free_running_token_ids",
    "backend",
    "timestamp_utc",
    "memory",
}

_REQUIRED_PACKED_KEYS = _REQUIRED_RESULT_KEYS | {"counters"}

_REQUIRED_COUNTER_KEYS = {
    "requested_strict_mode",
    "effective_strict_mode",
    "packed_attention_calls",
    "dense_fallback_calls",
    "full_history_materialization_calls",
}

_REQUIRED_QUALITY_KEYS = {
    "kl_divergence",
    "max_logit_delta",
    "mean_logit_delta",
    "top1_match",
    "top5_overlap",
    "top10_overlap",
    "logit_cosine",
    "first_divergent_token",
    "steps_compared",
}


@pytest.fixture
def manifest() -> dict:
    assert _MANIFEST_PATH.exists(), f"Manifest missing: {_MANIFEST_PATH}"
    return json.loads(_MANIFEST_PATH.read_text())


def test_manifest_exists() -> None:
    assert _MANIFEST_PATH.exists()


def test_manifest_is_valid_json(manifest: dict) -> None:
    assert isinstance(manifest, dict)


def test_manifest_has_required_keys(manifest: dict) -> None:
    missing = _REQUIRED_MANIFEST_KEYS - manifest.keys()
    assert not missing, f"Missing manifest keys: {missing}"


def test_manifest_status_passed(manifest: dict) -> None:
    """The manifest must record a passed status."""
    assert manifest.get("status") == "passed", (
        f"Manifest status is {manifest.get('status')!r}, expected 'passed'. "
        f"Violations: {manifest.get('violations', [])}"
    )


def test_manifest_exit_code_zero(manifest: dict) -> None:
    assert manifest.get("exit_code") == 0, (
        f"Manifest exit_code is {manifest.get('exit_code')!r}"
    )


def test_manifest_no_violations(manifest: dict) -> None:
    violations = manifest.get("violations", [])
    assert violations == [], f"Manifest has violations: {violations}"


def test_manifest_has_runs(manifest: dict) -> None:
    runs = manifest.get("runs", [])
    assert len(runs) > 0, "Manifest has no runs"


def test_runs_have_context_lengths(manifest: dict) -> None:
    for i, run in enumerate(manifest["runs"]):
        assert "context_length" in run, f"Run {i} missing context_length"
        assert isinstance(run["context_length"], int)
        assert run["context_length"] > 0


def test_runs_prompt_length_exact(manifest: dict) -> None:
    """Every run must have prompt_tokens_actual == context_length."""
    for run in manifest["runs"]:
        ctx = run["context_length"]
        actual = run.get("prompt_tokens_actual")
        assert actual == ctx, (
            f"Run context={ctx} but prompt_tokens_actual={actual}"
        )
        # All candidates must agree
        for candidate in ("dense", "eight_bit", "packed"):
            pt = run.get(candidate, {}).get("prompt_tokens")
            assert pt == ctx, (
                f"Run context={ctx} but {candidate}.prompt_tokens={pt}"
            )


def test_runs_token_match_true(manifest: dict) -> None:
    """Token match must be True for every run."""
    for run in manifest["runs"]:
        match = run.get("token_match")
        assert match is True, (
            f"Run context={run['context_length']} token_match={match}"
        )


def test_dense_baseline_produces_free_running_tokens(manifest: dict) -> None:
    for run in manifest["runs"]:
        dense = run.get("dense", {})
        ids = dense.get("free_running_token_ids")
        assert ids is not None and len(ids) > 0, (
            f"Run context={run['context_length']}: dense has no free-running tokens"
        )


def test_packed_produces_free_running_tokens(manifest: dict) -> None:
    for run in manifest["runs"]:
        packed = run.get("packed", {})
        ids = packed.get("free_running_token_ids")
        assert ids is not None and len(ids) > 0, (
            f"Run context={run['context_length']}: packed has no free-running tokens"
        )


def test_quality_present_for_all_runs(manifest: dict) -> None:
    for run in manifest["runs"]:
        quality = run.get("quality")
        assert quality is not None, (
            f"Run context={run['context_length']}: missing quality"
        )
        missing = _REQUIRED_QUALITY_KEYS - quality.keys()
        assert not missing, f"Run context={run['context_length']}: missing quality keys: {missing}"


def test_quality_kl_acceptable(manifest: dict) -> None:
    for run in manifest["runs"]:
        kl = run.get("quality", {}).get("kl_divergence")
        assert kl is not None
        assert kl < 0.01, (
            f"Run context={run['context_length']}: KL={kl} >= 0.01"
        )


def test_quality_top1_perfect(manifest: dict) -> None:
    for run in manifest["runs"]:
        top1 = run.get("quality", {}).get("top1_match")
        assert top1 == 1.0, (
            f"Run context={run['context_length']}: top1_match={top1} != 1.0"
        )


def test_quality_cosine_high(manifest: dict) -> None:
    for run in manifest["runs"]:
        cos = run.get("quality", {}).get("logit_cosine")
        assert cos is not None
        assert cos >= 0.999, (
            f"Run context={run['context_length']}: cosine={cos} < 0.999"
        )


def test_packed_counters_strict(manifest: dict) -> None:
    for run in manifest["runs"]:
        counters = run.get("packed", {}).get("counters", {})
        missing = _REQUIRED_COUNTER_KEYS - counters.keys()
        assert not missing, f"Run context={run['context_length']}: missing counters: {missing}"
        assert counters["requested_strict_mode"] is True
        assert counters["effective_strict_mode"] is True
        assert counters["dense_fallback_calls"] == 0
        assert counters["full_history_materialization_calls"] == 0
        assert counters["packed_attention_calls"] > 0


def test_dense_memory_reported(manifest: dict) -> None:
    for run in manifest["runs"]:
        mem = run.get("dense", {}).get("memory", {})
        assert "total_accounted_mb" in mem
        assert mem["total_accounted_mb"] > 0


def test_packed_memory_reported(manifest: dict) -> None:
    for run in manifest["runs"]:
        mem = run.get("packed", {}).get("memory", {})
        assert "total_accounted_mb" in mem
        assert mem["total_accounted_mb"] > 0


def test_eight_bit_memory_reported(manifest: dict) -> None:
    for run in manifest["runs"]:
        mem = run.get("eight_bit", {}).get("memory", {})
        assert "total_accounted_mb" in mem
        assert mem["total_accounted_mb"] > 0


def test_packed_memory_less_than_dense(manifest: dict) -> None:
    """Packed memory should be less than dense memory."""
    for run in manifest["runs"]:
        dense_mb = run.get("dense", {}).get("memory", {}).get("total_accounted_mb", 0)
        packed_mb = run.get("packed", {}).get("memory", {}).get("total_accounted_mb", 0)
        assert packed_mb < dense_mb, (
            f"Run context={run['context_length']}: packed_mem={packed_mb}MB "
            f">= dense_mem={dense_mb}MB"
        )


def test_backend_report_has_provenance(manifest: dict) -> None:
    br = manifest.get("backend_report", {})
    assert br.get("kernel_source_hash"), "Missing kernel_source_hash"
    assert br.get("mlx_version"), "Missing mlx_version"
    assert br.get("git_commit"), "Missing git_commit"
    assert br.get("python_version"), "Missing python_version"
    assert br.get("platform_machine"), "Missing platform_machine"


def test_free_running_timing_present(manifest: dict) -> None:
    """Every candidate must report free-running generation time."""
    for run in manifest["runs"]:
        for candidate in ("dense", "eight_bit", "packed"):
            result = run.get(candidate, {})
            fr = result.get("free_running_elapsed_ms")
            assert fr is not None, (
                f"Run context={run['context_length']}: {candidate} missing "
                "free_running_elapsed_ms"
            )
            assert isinstance(fr, (int, float)), (
                f"Run context={run['context_length']}: {candidate} "
                f"free_running_elapsed_ms={fr!r} is not numeric"
            )
            assert fr >= 0, (
                f"Run context={run['context_length']}: {candidate} "
                f"free_running_elapsed_ms={fr} < 0"
            )


def test_teacher_forced_timing_present_for_dense_and_packed(manifest: dict) -> None:
    """Dense and packed must report teacher-forced re-run time."""
    for run in manifest["runs"]:
        for candidate in ("dense", "packed"):
            result = run.get(candidate, {})
            tf = result.get("teacher_forced_elapsed_ms")
            assert tf is not None, (
                f"Run context={run['context_length']}: {candidate} missing "
                "teacher_forced_elapsed_ms"
            )
            assert isinstance(tf, (int, float)), (
                f"Run context={run['context_length']}: {candidate} "
                f"teacher_forced_elapsed_ms={tf!r} is not numeric"
            )
            assert tf >= 0, (
                f"Run context={run['context_length']}: {candidate} "
                f"teacher_forced_elapsed_ms={tf} < 0"
            )


def test_decode_ms_per_token_present(manifest: dict) -> None:
    """Every candidate must report per-token decode time."""
    for run in manifest["runs"]:
        for candidate in ("dense", "eight_bit", "packed"):
            result = run.get(candidate, {})
            dpt = result.get("decode_ms_per_token")
            assert dpt is not None, (
                f"Run context={run['context_length']}: {candidate} missing "
                "decode_ms_per_token"
            )
            assert isinstance(dpt, (int, float)), (
                f"Run context={run['context_length']}: {candidate} "
                f"decode_ms_per_token={dpt!r} is not numeric"
            )
            assert dpt >= 0, (
                f"Run context={run['context_length']}: {candidate} "
                f"decode_ms_per_token={dpt} < 0"
            )


def test_multi_context_runs_present(manifest: dict) -> None:
    """At least 3 context lengths should be tested for coverage."""
    contexts = [r["context_length"] for r in manifest["runs"]]
    assert len(contexts) >= 3, f"Only {len(contexts)} context lengths tested"


def test_multi_block_contexts_have_multiple_blocks(manifest: dict) -> None:
    """Contexts >= 128 tokens should create multiple packed blocks."""
    for run in manifest["runs"]:
        ctx = run["context_length"]
        if ctx >= 128:
            counters = run.get("packed", {}).get("counters", {})
            blocks = counters.get("packed_blocks_created", 0)
            assert blocks > 1, (
                f"Run context={ctx}: expected multiple blocks, got {blocks}"
            )


def test_all_contexts_zero_fallback(manifest: dict) -> None:
    """Every run must have zero dense fallback."""
    for run in manifest["runs"]:
        counters = run.get("packed", {}).get("counters", {})
        fb = counters.get("dense_fallback_calls", 0)
        assert fb == 0, f"Run context={run['context_length']}: fallback={fb}"


def test_all_contexts_zero_materialization(manifest: dict) -> None:
    """Every run must have zero full-history materialization."""
    for run in manifest["runs"]:
        counters = run.get("packed", {}).get("counters", {})
        mat = counters.get("full_history_materialization_calls", 0)
        assert mat == 0, f"Run context={run['context_length']}: materialization={mat}"


def test_attention_calls_scale_with_context(manifest: dict) -> None:
    """Packed attention calls should increase with context length."""
    contexts = []
    calls = []
    for run in manifest["runs"]:
        ctx = run["context_length"]
        c = run.get("packed", {}).get("counters", {}).get("packed_attention_calls", 0)
        contexts.append(ctx)
        calls.append(c)
    # At minimum, longer contexts should not have fewer calls than shorter ones
    for i in range(1, len(contexts)):
        if contexts[i] > contexts[i - 1]:
            assert calls[i] >= calls[i - 1], (
                f"Context {contexts[i]} has {calls[i]} calls but "
                f"context {contexts[i-1]} has {calls[i-1]} calls"
            )


def test_long_context_token_match(manifest: dict) -> None:
    """Token match must hold even for the longest tested context."""
    longest = max(manifest["runs"], key=lambda r: r["context_length"])
    assert longest.get("token_match") is True, (
        f"Token match failed at longest context={longest['context_length']}"
    )
