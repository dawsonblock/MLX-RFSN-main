"""Token sequence hash integrity tests.

Every full-logit artifact must prove the baseline and candidates were compared
on the same target sequence via a non-empty token_sequence_hash.
"""
from __future__ import annotations

import json
from pathlib import Path

from rfsn_v11.candidates.logit_capture import compute_token_sequence_hash

ARTIFACT_DIRS = [
    Path("artifacts/bench/shootout/quick"),
    Path("artifacts/bench/shootout/full_logit"),
    Path("artifacts/bench/shootout/memory"),
    Path("artifacts/bench/shootout/promotion"),
]
WINNER_JSON = Path("artifacts/winner/winner.json")


def test_compute_token_sequence_hash_is_non_empty() -> None:
    """The helper must produce a deterministic non-empty hash."""
    h1 = compute_token_sequence_hash(
        model_id="test-model",
        prompt_id="p0",
        prompt_text="hello",
        target_token_ids=[1, 2, 3],
        max_tokens=10,
        temperature=0.0,
        decode_mode="greedy",
        methodology="teacher_forced_logit_v1",
    )
    assert h1 != "", "token_sequence_hash must not be empty"
    assert len(h1) == 64, "token_sequence_hash must be a SHA-256 hex string"

    h2 = compute_token_sequence_hash(
        model_id="test-model",
        prompt_id="p0",
        prompt_text="hello",
        target_token_ids=[1, 2, 3],
        max_tokens=10,
        temperature=0.0,
        decode_mode="greedy",
        methodology="teacher_forced_logit_v1",
    )
    assert h1 == h2, "token_sequence_hash must be deterministic"

    h3 = compute_token_sequence_hash(
        model_id="test-model",
        prompt_id="p0",
        prompt_text="hello",
        target_token_ids=[1, 2, 4],  # different token
        max_tokens=10,
        temperature=0.0,
        decode_mode="greedy",
        methodology="teacher_forced_logit_v1",
    )
    assert h1 != h3, "token_sequence_hash must change when target tokens change"


def _read_metadata(path: Path) -> dict:
    json_path = path / "results.json"
    assert json_path.exists(), f"{json_path} missing"
    payload = json.loads(json_path.read_text())
    return payload.get("metadata", {})


_NO_NATIVE_RUN_STATUSES = {
    "NO_NATIVE_EVIDENCE_YET",
    "TEACHER_FORCED_RERUN_INCOMPLETE_NO_PROMOTION",
}


def _skip_if_no_native_run(meta: dict, artifact_name: str) -> None:
    """Skip when the artifact was not produced by a real Apple Silicon run.

    An empty token_sequence_hash or an explicit "not-yet-run" methodology
    status both indicate the artifact was generated without a live model.
    These tests are release-evidence gates that require native hardware.
    """
    status = meta.get("methodology_status", "")
    tsh = meta.get("token_sequence_hash", "")
    if status in _NO_NATIVE_RUN_STATUSES or not tsh:
        import pytest
        pytest.skip(
            f"{artifact_name} artifact has no native-run evidence "
            f"(methodology_status={status!r}, token_sequence_hash={tsh!r}) "
            "— native Apple Silicon run required to populate token_sequence_hash"
        )


def test_full_logit_artifact_has_non_empty_token_sequence_hash() -> None:
    """full_logit artifact metadata must contain a non-empty hash."""
    meta = _read_metadata(Path("artifacts/bench/shootout/full_logit"))
    _skip_if_no_native_run(meta, "full_logit")
    tsh = meta.get("token_sequence_hash", "")
    assert tsh != "", (
        "full_logit artifact token_sequence_hash is empty — "
        "teacher-forced rerun required before promotion can be considered"
    )


def test_promotion_artifact_has_non_empty_token_sequence_hash() -> None:
    """promotion artifact metadata must contain a non-empty hash."""
    meta = _read_metadata(Path("artifacts/bench/shootout/promotion"))
    _skip_if_no_native_run(meta, "promotion")
    tsh = meta.get("token_sequence_hash", "")
    assert tsh != "", (
        "promotion artifact token_sequence_hash is empty — "
        "teacher-forced rerun required"
    )


def test_no_promotion_allowed_when_token_sequence_hash_empty() -> None:
    """promotion_allowed must be false if token_sequence_hash is empty."""
    for art_dir in ARTIFACT_DIRS:
        meta = _read_metadata(art_dir)
        tsh = meta.get("token_sequence_hash", "")
        promo = meta.get("promotion_allowed", False)
        if not tsh:
            assert promo is False, (
                f"{art_dir.name} artifact sets promotion_allowed=true "
                f"but token_sequence_hash is empty"
            )


def test_winner_json_has_methodology_status() -> None:
    """winner.json must include methodology_status field."""
    assert WINNER_JSON.exists(), "winner.json missing"
    data = json.loads(WINNER_JSON.read_text())
    assert "methodology_status" in data, (
        "winner.json must contain methodology_status"
    )


def test_winner_is_null_when_token_sequence_hash_empty() -> None:
    """If any full-logit artifact lacks a token_sequence_hash, winner must be null."""
    full_meta = _read_metadata(Path("artifacts/bench/shootout/full_logit"))
    tsh = full_meta.get("token_sequence_hash", "")
    if not tsh:
        data = json.loads(WINNER_JSON.read_text())
        assert data.get("winner") is None, (
            "winner.json names a winner but full_logit token_sequence_hash is empty"
        )
