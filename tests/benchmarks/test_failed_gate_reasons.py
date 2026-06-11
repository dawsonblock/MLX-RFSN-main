"""Every non-pass row must explain why."""
from __future__ import annotations

import json
from pathlib import Path

SHOOTOUT_DIRS = [
    Path("artifacts/bench/shootout/quick"),
    Path("artifacts/bench/shootout/full_logit"),
    Path("artifacts/bench/shootout/memory"),
    Path("artifacts/bench/shootout/promotion"),
]


def _read_results(path: Path) -> list[dict]:
    json_path = path / "results.json"
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text())
    return payload.get("results", [])


def test_no_fail_row_without_failed_gate_reasons() -> None:
    """Any row with gate_status=FAIL must have non-empty failed_gate_reasons."""
    for art_dir in SHOOTOUT_DIRS:
        for row in _read_results(art_dir):
            if row.get("gate_status") == "FAIL":
                reasons = row.get("failed_gate_reasons")
                assert reasons is not None, (
                    f"{art_dir.name} row {row.get('name')} is FAIL but "
                    f"failed_gate_reasons is missing"
                )
                assert len(reasons) > 0, (
                    f"{art_dir.name} row {row.get('name')} is FAIL but "
                    f"failed_gate_reasons is empty"
                )


def test_gate_thresholds_match_quality_gates() -> None:
    """Artifact gate_thresholds must match the canonical LogitGateThresholds."""
    from rfsn_v11.candidates.quality_gates import LogitGateThresholds

    canonical = LogitGateThresholds().to_dict()
    for art_dir in SHOOTOUT_DIRS:
        json_path = art_dir / "results.json"
        if not json_path.exists():
            continue
        payload = json.loads(json_path.read_text())
        meta = payload.get("metadata", {})
        artifact_thresholds = meta.get("gate_thresholds", {})
        if not artifact_thresholds:
            continue
        for key, expected in canonical.items():
            actual = artifact_thresholds.get(key)
            assert actual == expected, (
                f"{art_dir.name} artifact threshold {key}={actual} "
                f"does not match canonical {expected}"
            )
