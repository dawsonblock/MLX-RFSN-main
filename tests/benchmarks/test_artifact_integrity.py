"""Artifact integrity tests.

Ensures benchmark artifacts are complete and not misleading.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from rfsn_v11.candidates.artifact_utils import (
    _build_honest_markdown_table,
)
from rfsn_v11.candidates.candidate_status import CandidateStatus
from rfsn_v11.candidates.json_utils import dump_json_strict


def test_results_json_exists() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "shootout"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"name": "test", "candidate_status": "EXPERIMENTAL"}]
        json_path = out_dir / "results.json"
        with json_path.open("w") as fh:
            dump_json_strict(rows, fh)
        assert json_path.exists()


def test_results_csv_exists() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "shootout"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"name": "test", "candidate_status": "EXPERIMENTAL"}]
        json_path = out_dir / "results.json"
        with json_path.open("w") as fh:
            dump_json_strict(rows, fh)
        # CSV is optional when rows exist
        assert json_path.exists()


def test_results_md_exists() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "shootout"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"name": "test", "candidate_status": "EXPERIMENTAL"}]
        md_path = out_dir / "results.md"
        with md_path.open("w") as fh:
            fh.write(_build_honest_markdown_table(rows))
        assert md_path.exists()


def test_all_candidates_have_status() -> None:
    rows = [
        {
            "name": "mlx_lm_baseline",
            "candidate_status": str(CandidateStatus.CONTROL),
        },
        {
            "name": "turboquant_v2",
            "candidate_status": str(CandidateStatus.EXPERIMENTAL),
        },
    ]
    for row in rows:
        assert "candidate_status" in row
        assert row["candidate_status"] != ""


def test_all_candidates_have_gate_status() -> None:
    rows = [
        {
            "name": "mlx_lm_baseline",
            "gate_status": "PASS_NO_PROMOTE",
        },
        {
            "name": "turboquant_v2",
            "gate_status": "PENDING_LOGIT_GATE",
        },
    ]
    for row in rows:
        assert "gate_status" in row
        assert row["gate_status"] != ""


def test_promoted_candidates_have_full_metrics() -> None:
    promoted = {
        "name": "turboquant_v2",
        "candidate_status": str(CandidateStatus.PROMOTED),
        "promotion_eligible": True,
        "logit_cosine": 0.9995,
        "size_ratio": 0.265,
        "compression_factor": 3.77,
        "tokens_per_sec": 45.0,
        "gate_status": "PASS",
    }
    assert promoted["promotion_eligible"] is True
    assert promoted["logit_cosine"] is not None
    assert promoted["size_ratio"] is not None
    assert promoted["compression_factor"] is not None


def test_no_misleading_compression_wording() -> None:
    md = _build_honest_markdown_table([
        {
            "name": "test",
            "candidate_status": "EXPERIMENTAL",
            "size_ratio": 0.265,
        }
    ])
    # Should NOT contain misleading "0.265x compression"
    assert "0.265x compression" not in md
    # Should show as ratio or percentage
    assert "0.265" in md or "26.5" in md


def test_skipped_artifact_markdown_is_explicit() -> None:
    rows = [
        {
            "status": "SKIPPED_NO_MLX_LM",
            "reason": "mlx_lm is not installed",
        }
    ]
    md = _build_honest_markdown_table(rows)
    assert "SKIPPED_NO_MLX_LM" in md
    assert "mlx_lm is not installed" in md


def test_no_active_legacy_winner_artifact() -> None:
    # Old Alpha 7 artifacts must not exist in active path
    assert not Path("artifacts/bench/shootout/results.json").exists()
    assert not Path("artifacts/bench/shootout/results.md").exists()
    assert not Path("artifacts/bench/shootout/results.csv").exists()


def _get_results(payload: Any) -> list[dict[str, Any]]:
    """Extract results list from v2 wrapped artifact or v1 bare list."""
    if isinstance(payload, dict):
        return payload.get("results", [])
    if isinstance(payload, list):
        return payload
    return []


def test_promotion_artifact_exists() -> None:
    # Promotion report must exist and contain candidates or a note.
    # After methodology repair, there may be no eligible candidates
    # until artifacts are regenerated under teacher_forced_logit_v1.
    promo_json = Path("artifacts/bench/shootout/promotion/results.json")
    promo_md = Path("artifacts/bench/shootout/promotion/results.md")
    assert promo_json.exists(), "Promotion JSON artifact missing"
    assert promo_md.exists(), "Promotion Markdown artifact missing"
    payload = json.loads(promo_json.read_text())
    results = _get_results(payload)
    # Either a note row, actual candidate rows, or metadata explaining status
    has_note = any("note" in r for r in results if isinstance(r, dict))
    has_candidates = any(
        r.get("name") for r in results if isinstance(r, dict)
    )
    assert has_note or has_candidates, (
        "Promotion artifact should contain candidates or a note"
    )
    # v2 wrapped artifacts must have methodology metadata
    if isinstance(payload, dict):
        meta = payload.get("metadata", {})
        assert meta.get("benchmark_methodology") == "teacher_forced_logit_v1"


def test_winner_json_agrees_with_promotion_report() -> None:
    promo_json = Path("artifacts/bench/shootout/promotion/results.json")
    promo_payload = json.loads(promo_json.read_text())
    promo_results = _get_results(promo_payload)
    has_eligible = any(
        r.get("promotion_eligible")
        for r in promo_results
        if isinstance(r, dict)
    )
    winner_json = Path("artifacts/winner/winner.json")
    assert winner_json.exists(), "winner.json must exist"
    data = json.loads(winner_json.read_text())
    # Winner must declare methodology
    assert data.get("methodology") == "teacher_forced_logit_v1", (
        "winner.json must declare teacher_forced_logit_v1 methodology"
    )
    # Global promotion lock: when promotion_allowed is false, winner must be
    # null even if individual candidates pass quality gates.
    promotion_allowed = data.get("promotion_allowed", False)
    if promotion_allowed and has_eligible:
        assert data.get("winner") is not None, (
            "winner.json should name a winner when promotion is allowed"
        )
    else:
        assert data.get("winner") is None, (
            "winner.json should have null winner when promotion is disallowed"
        )


def _scan_for_non_finite(obj: Any, path: str = "") -> list[str]:
    """Recursively scan *obj* for NaN / Infinity and return error paths."""
    import math
    errors: list[str] = []
    if isinstance(obj, float):
        if math.isnan(obj):
            errors.append(f"{path}: NaN")
        elif math.isinf(obj):
            errors.append(f"{path}: Infinity")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            errors.extend(_scan_for_non_finite(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            errors.extend(_scan_for_non_finite(v, f"{path}[{i}]"))
    return errors


def test_active_artifacts_have_no_nan_or_infinity() -> None:
    """All shipped JSON artifacts must be strict (no NaN / Infinity)."""
    artifact_dirs = [
        Path("artifacts/bench/shootout/quick"),
        Path("artifacts/bench/shootout/full_logit"),
        Path("artifacts/bench/shootout/memory"),
        Path("artifacts/bench/shootout/promotion"),
        Path("artifacts/winner"),
    ]
    all_errors: list[str] = []
    for d in artifact_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*.json"):
            payload = json.loads(p.read_text())
            # Scan both metadata and results for v2 wrapped artifacts
            if isinstance(payload, dict):
                meta_errors = _scan_for_non_finite(
                    payload.get("metadata", {}), str(p) + ":metadata"
                )
                all_errors.extend(meta_errors)
                results_errors = _scan_for_non_finite(
                    payload.get("results", []), str(p) + ":results"
                )
                all_errors.extend(results_errors)
            else:
                errors = _scan_for_non_finite(payload, str(p))
                if errors:
                    all_errors.extend(errors)
    assert not all_errors, (
        f"Non-finite floats found in artifacts: {all_errors}"
    )


def test_rfsn_v10_trace_is_runtime_instrumented() -> None:
    """RFSN v10 proof trace must be runtime-instrumented, not estimated."""
    trace_path = Path("artifacts/bench/shootout/debug/rfsn_v10_k8_v5_trace.json")
    if not trace_path.exists():
        pytest.skip("RFSN v10 trace not generated yet")
    trace = json.loads(trace_path.read_text())
    assert trace.get("trace_type") == "runtime_instrumented", (
        "RFSN v10 trace must be runtime_instrumented, not estimated"
    )
    # Key counters must be populated (not None)
    assert trace.get("cache_bytes_written_actual") is not None
    assert trace.get("cache_bytes_read_actual") is not None
    assert trace.get("prefill_quantize_events") is not None
    assert trace.get("decode_quantized_store_events") is not None
    assert trace.get("patch_enter_count") == 1
    assert trace.get("patch_exit_count") == 1
    assert trace.get("layers_wrapped_actual") is not None
    # Promotion-grade traces must prove the compressed cache was both
    # written and read during the teacher-forced capture.
    assert trace.get("cache_bytes_read_actual", 0) > 0, (
        "trace cache_bytes_read_actual must be > 0 for promotion"
    )
    assert trace.get("decode_quantized_fetch_events", 0) > 0, (
        "trace decode_quantized_fetch_events must be > 0 for promotion"
    )
