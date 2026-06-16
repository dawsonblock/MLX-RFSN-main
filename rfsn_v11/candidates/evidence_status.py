"""Strict evidence-status enum for artifact validation.

Phase 1: Replaces arbitrary strings ("awaiting_execution", "placeholder",
"no_native_run_completed") with a structured enum so tests and integrity
checkers cannot silently pass on error-only artifacts.
"""
from __future__ import annotations

from enum import StrEnum


class EvidenceStatus(StrEnum):
    """Lifecycle state of an evidence artifact or individual row.

    States are ordered from least to most validated:
    - ABSENT: artifact file does not exist
    - EXECUTION_FAILED: every row contains an error; no successful execution
    - PARTIAL: some rows succeeded, some failed; not enough for validation
    - COMPLETE: all required rows present and individually successful
    - VALIDATION_FAILED: complete artifact failed a quality or schema gate
    - VALIDATION_PASSED: complete artifact passed all gates
    """

    ABSENT = "absent"
    EXECUTION_FAILED = "execution_failed"
    PARTIAL = "partial"
    COMPLETE = "complete"
    VALIDATION_FAILED = "validation_failed"
    VALIDATION_PASSED = "validation_passed"


def classify_artifact_status(rows: list[dict]) -> EvidenceStatus:
    """Classify an artifact's overall status from its row list.

    Returns:
        EvidenceStatus for the whole artifact.
    """
    if not rows:
        return EvidenceStatus.ABSENT

    successful = [r for r in rows if not r.get("error") and r.get("status") != "error"]
    failed = [r for r in rows if r.get("error") or r.get("status") == "error"]

    if not successful:
        return EvidenceStatus.EXECUTION_FAILED
    if failed:
        return EvidenceStatus.PARTIAL
    return EvidenceStatus.COMPLETE


def require_successful_rows(rows: list[dict], label: str) -> list[dict]:
    """Return successful rows or raise AssertionError.

    Use this helper in tests to eliminate vacuous "if error in row: continue"
    patterns that let error-only artifacts pass silently.

    Args:
        rows: list of artifact rows (dicts).
        label: human-readable artifact name for error messages.

    Returns:
        Filtered list of rows without errors.

    Raises:
        AssertionError: if no successful rows exist.
    """
    successful = [
        r for r in rows
        if not r.get("error") and r.get("status") != "error"
    ]
    assert successful, (
        f"{label}: evidence contains no successful executions — "
        f"all {len(rows)} row(s) have errors. "
        f"Artifact status would be {EvidenceStatus.EXECUTION_FAILED}."
    )
    return successful
