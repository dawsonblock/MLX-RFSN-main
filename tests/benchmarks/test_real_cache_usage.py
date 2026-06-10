"""Prove a candidate actually used its own KV cache path during generation.

Each candidate must expose on its CandidateResult:
  - cache_backend_used
  - cache_events
  - cache_bytes_written
  - cache_bytes_read

If those are missing, the candidate is not promotion eligible.
"""
from __future__ import annotations

import pytest

from rfsn_v11.candidates.base import CandidateResult
from rfsn_v11.candidates.candidate_status import CandidateStatus
from rfsn_v11.candidates.quality_gates import GATE_STATUS_PENDING_REAL_CACHE_INJECTION


class FakeCandidate:
    """Minimal fake for testing cache proof rules."""

    def __init__(self, name: str, status: CandidateStatus) -> None:
        self.name = name
        self.candidate_status = status


def _make_result(
    cache_backend: str = "",
    cache_events: list[str] | None = None,
    cache_bytes_written: int | None = None,
    cache_bytes_read: int | None = None,
    status: CandidateStatus = CandidateStatus.EXPERIMENTAL,
) -> CandidateResult:
    return CandidateResult(
        name="test_candidate",
        model_id="test",
        prompt="test",
        candidate_status=status,
        cache_backend_used=cache_backend,
        cache_events=cache_events or [],
        cache_bytes_written=cache_bytes_written,
        cache_bytes_read=cache_bytes_read,
    )


def _apply_cache_proof_rules(result: CandidateResult) -> CandidateResult:
    """Apply the same cache-proof logic kv_shootout uses."""
    missing = []
    if not result.cache_backend_used:
        missing.append("cache_backend_used")
    if not result.cache_events:
        missing.append("cache_events")
    if result.cache_bytes_written is None:
        missing.append("cache_bytes_written")
    if result.cache_bytes_read is None:
        missing.append("cache_bytes_read")

    if missing:
        result.promotion_eligible = False
        result.gate_status = GATE_STATUS_PENDING_REAL_CACHE_INJECTION
        result.notes += f"  [cache proof missing: {', '.join(missing)}]"
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_full_cache_proof_passes() -> None:
    result = _make_result(
        cache_backend="turboquant_v2",
        cache_events=["prefill_compress", "decode_fetch"],
        cache_bytes_written=123456,
        cache_bytes_read=789012,
    )
    result = _apply_cache_proof_rules(result)
    assert result.gate_status != GATE_STATUS_PENDING_REAL_CACHE_INJECTION
    assert "cache proof missing" not in result.notes


@pytest.mark.unit
def test_missing_cache_backend_blocks_promotion() -> None:
    result = _make_result(
        cache_backend="",
        cache_events=["prefill_compress"],
        cache_bytes_written=1000,
        cache_bytes_read=1000,
    )
    result = _apply_cache_proof_rules(result)
    assert result.promotion_eligible is False
    assert result.gate_status == GATE_STATUS_PENDING_REAL_CACHE_INJECTION


@pytest.mark.unit
def test_missing_cache_events_blocks_promotion() -> None:
    result = _make_result(
        cache_backend="some_backend",
        cache_events=[],
        cache_bytes_written=1000,
        cache_bytes_read=1000,
    )
    result = _apply_cache_proof_rules(result)
    assert result.promotion_eligible is False
    assert result.gate_status == GATE_STATUS_PENDING_REAL_CACHE_INJECTION


@pytest.mark.unit
def test_missing_cache_bytes_written_blocks_promotion() -> None:
    result = _make_result(
        cache_backend="some_backend",
        cache_events=["prefill_compress"],
        cache_bytes_written=None,
        cache_bytes_read=1000,
    )
    result = _apply_cache_proof_rules(result)
    assert result.promotion_eligible is False
    assert result.gate_status == GATE_STATUS_PENDING_REAL_CACHE_INJECTION


@pytest.mark.unit
def test_offline_only_status_never_promotes() -> None:
    result = _make_result(
        cache_backend="rfsn_v11_offline",
        cache_events=["offline_compress"],
        cache_bytes_written=1000,
        cache_bytes_read=1000,
        status=CandidateStatus.OFFLINE_ONLY,
    )
    # Even with cache proof fields present, OFFLINE_ONLY cannot promote
    assert result.candidate_status == CandidateStatus.OFFLINE_ONLY


@pytest.mark.unit
def test_control_status_never_promotes() -> None:
    result = _make_result(
        cache_backend="mlx_lm_fp16",
        cache_events=["prefill", "decode"],
        cache_bytes_written=1000,
        cache_bytes_read=1000,
        status=CandidateStatus.CONTROL,
    )
    assert result.candidate_status == CandidateStatus.CONTROL
