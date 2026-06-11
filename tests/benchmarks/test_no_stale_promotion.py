"""Prevent stale promotion artifacts from being trusted.

After the teacher-forced logit gate was introduced, all prior promotion
artifacts are considered stale unless they were regenerated under the
corrected methodology. These tests enforce that invariant.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rfsn_v11.integrations.cache_policy import PROMOTED_POLICIES


def _read_shootout_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return payload
    # Legacy v1 format: bare list
    return {"results": payload}


def test_winner_json_declares_teacher_forced_methodology() -> None:
    """winner.json must declare teacher_forced_logit_v1 methodology."""
    winner_json = Path("artifacts/winner/winner.json")
    assert winner_json.exists(), "winner.json missing"
    data = json.loads(winner_json.read_text())
    assert data.get("methodology") == "teacher_forced_logit_v1", (
        "winner.json must declare teacher_forced_logit_v1 methodology"
    )


def test_promotion_artifact_disallows_promotion_if_mismatch() -> None:
    """Promotion is only allowed when artifact metadata says so."""
    promo_json = Path("artifacts/bench/shootout/promotion/results.json")
    assert promo_json.exists(), "promotion artifact missing"
    payload = _read_shootout_payload(promo_json)
    meta = payload.get("metadata", {})
    if meta.get("promotion_allowed") is False:
        # Global promotion lock: winner.json must be null even if the
        # promotion report lists gate-passing candidates.
        winner_json = Path("artifacts/winner/winner.json")
        data = json.loads(winner_json.read_text())
        assert data.get("winner") is None, (
            "promotion_allowed=false but winner.json names a winner"
        )


def test_full_logit_artifact_has_teacher_forced_methodology() -> None:
    """full_logit artifact must declare teacher_forced_logit_v1."""
    full_logit_json = Path("artifacts/bench/shootout/full_logit/results.json")
    assert full_logit_json.exists(), "full_logit artifact missing"
    payload = _read_shootout_payload(full_logit_json)
    meta = payload.get("metadata", {})
    assert meta.get("benchmark_methodology") == "teacher_forced_logit_v1", (
        "full_logit artifact must declare teacher_forced_logit_v1 methodology"
    )


def test_memory_artifact_has_teacher_forced_methodology() -> None:
    """memory artifact must declare teacher_forced_logit_v1."""
    memory_json = Path("artifacts/bench/shootout/memory/results.json")
    assert memory_json.exists(), "memory artifact missing"
    payload = _read_shootout_payload(memory_json)
    meta = payload.get("metadata", {})
    assert meta.get("benchmark_methodology") == "teacher_forced_logit_v1", (
        "memory artifact must declare teacher_forced_logit_v1 methodology"
    )


def test_cache_policy_promoted_registry_empty_when_no_winner() -> None:
    """PROMOTED_POLICIES must be empty when winner.json has no winner."""
    winner_json = Path("artifacts/winner/winner.json")
    assert winner_json.exists(), "winner.json missing"
    data = json.loads(winner_json.read_text())
    if data.get("winner") is None:
        assert not PROMOTED_POLICIES, (
            "cache_policy.py PROMOTED_POLICIES must be empty when "
            "winner.json has no winner"
        )


def test_cache_policy_promoted_registry_nonempty_when_winner_exists() -> None:
    """If winner.json names a winner, cache_policy.py must list it."""
    winner_json = Path("artifacts/winner/winner.json")
    assert winner_json.exists(), "winner.json missing"
    data = json.loads(winner_json.read_text())
    winner = data.get("winner")
    if winner is not None:
        assert winner in PROMOTED_POLICIES, (
            f"cache_policy.py must list winner {winner!r} in PROMOTED_POLICIES"
        )


def test_winner_requires_full_logit_and_memory_artifacts() -> None:
    """If winner.json names a winner, both full_logit and memory must exist."""
    winner_json = Path("artifacts/winner/winner.json")
    assert winner_json.exists(), "winner.json missing"
    data = json.loads(winner_json.read_text())
    if data.get("winner") is not None:
        assert Path(
            "artifacts/bench/shootout/full_logit/results.json"
        ).exists()
        assert Path(
            "artifacts/bench/shootout/memory/results.json"
        ).exists()


def test_promotion_artifact_not_stale_legacy_path() -> None:
    """Old Alpha 7 promotion artifacts must not exist in active path."""
    assert not Path("artifacts/bench/shootout/results.json").exists()
    assert not Path("artifacts/bench/shootout/results.md").exists()
    assert not Path("artifacts/bench/shootout/results.csv").exists()


def test_no_promotion_when_roadmap_says_none() -> None:
    """If RELEASE_MANIFEST says no candidate promoted, winner must agree."""
    manifest = Path("RELEASE_MANIFEST.md")
    assert manifest.exists(), "RELEASE_MANIFEST.md missing"
    text = manifest.read_text()
    winner_json = Path("artifacts/winner/winner.json")
    data = json.loads(winner_json.read_text())
    # If manifest says "Promoted candidate: NONE", winner must be null
    if "Promoted candidate: NONE" in text:
        assert data.get("winner") is None, (
            "RELEASE_MANIFEST says NONE but winner.json names a winner"
        )


def test_global_promotion_lock_forces_all_rows_non_promotable() -> None:
    """If promotion_allowed=false, no row may claim promotion_eligible=true."""
    for art_dir in [
        Path("artifacts/bench/shootout/quick"),
        Path("artifacts/bench/shootout/full_logit"),
        Path("artifacts/bench/shootout/memory"),
        Path("artifacts/bench/shootout/promotion"),
    ]:
        json_path = art_dir / "results.json"
        if not json_path.exists():
            continue
        payload = _read_shootout_payload(json_path)
        meta = payload.get("metadata", {})
        if meta.get("promotion_allowed") is False:
            rows = payload.get("results", [])
            for row in rows:
                if not isinstance(row, dict) or "note" in row:
                    continue
                assert row.get("promotion_eligible") is not True, (
                    f"{art_dir.name} row {row.get('name')} has "
                    f"promotion_eligible=true but promotion_allowed=false"
                )
                assert row.get("gate_status") != "PASS", (
                    f"{art_dir.name} row {row.get('name')} has "
                    f"gate_status=PASS while promotion is globally locked"
                )
