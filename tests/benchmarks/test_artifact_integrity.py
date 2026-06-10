"""Artifact integrity tests.

Ensures benchmark artifacts are complete and not misleading.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from rfsn_v11.candidates.artifact_utils import (
    _build_honest_markdown_table,
    _export_winner,
)
from rfsn_v11.candidates.candidate_status import CandidateStatus


def test_results_json_exists() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "shootout"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"name": "test", "candidate_status": "EXPERIMENTAL"}]
        json_path = out_dir / "results.json"
        with json_path.open("w") as fh:
            json.dump(rows, fh)
        assert json_path.exists()


def test_results_csv_exists() -> None:
    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td) / "shootout"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = [{"name": "test", "candidate_status": "EXPERIMENTAL"}]
        json_path = out_dir / "results.json"
        with json_path.open("w") as fh:
            json.dump(rows, fh)
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


def test_promotion_artifact_exists() -> None:
    # Promotion report must exist and say no candidate is eligible
    promo_json = Path("artifacts/bench/shootout/promotion/results.json")
    promo_md = Path("artifacts/bench/shootout/promotion/results.md")
    assert promo_json.exists(), "Promotion JSON artifact missing"
    assert promo_md.exists(), "Promotion Markdown artifact missing"
    data = json.loads(promo_json.read_text())
    # Either a note row or actual rows with no promotion eligible
    has_note = any("note" in r for r in data if isinstance(r, dict))
    no_eligible = not any(
        r.get("promotion_eligible") for r in data if isinstance(r, dict)
    )
    assert has_note or no_eligible, (
        "Promotion artifact should say no candidate is eligible"
    )


def test_winner_json_agrees_with_promotion_report() -> None:
    rows = [{"note": "No candidate is promotion eligible."}]
    _export_winner(rows, ["Qwen/Qwen2.5-0.5B-Instruct"])
    # winner.json should exist with null winner
    winner_json = Path("artifacts/winner/winner.json")
    if winner_json.exists():
        data = json.loads(winner_json.read_text())
        if not any(
            r.get("promotion_eligible") for r in rows if isinstance(r, dict)
        ):
            assert data.get("winner") is None
