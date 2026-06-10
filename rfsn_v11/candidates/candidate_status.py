"""Candidate status enum for the KV-compression shootout.

Each candidate has a lifecycle status that determines whether it can
be promoted, referenced, or must remain experimental.
"""
from __future__ import annotations

from enum import StrEnum


class CandidateStatus(StrEnum):
    """Lifecycle status of a compression candidate."""

    CONTROL = "CONTROL"
    BASELINE = "BASELINE"
    EXPERIMENTAL = "EXPERIMENTAL"
    OFFLINE_ONLY = "OFFLINE_ONLY"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    PROMOTION_ELIGIBLE = "PROMOTION_ELIGIBLE"
    PROMOTED = "PROMOTED"
    FAILED = "FAILED"


# Canonical status assignments for known candidates.
# Adapters should set these on their CandidateResult instances.
CANDIDATE_STATUSES: dict[str, CandidateStatus] = {
    "mlx_lm_baseline": CandidateStatus.CONTROL,
    "mlx_lm_quantized_kv_b8": CandidateStatus.CONTROL,
    "rfsn_v10_k8_v5_gs32": CandidateStatus.BASELINE,
    "rfsn_v10_k8_v5_gs64": CandidateStatus.BASELINE,
    "rfsn_v11_offline_asymmetric_kv_k8v4_gs64": CandidateStatus.OFFLINE_ONLY,
    "turboquant_v2_b4_gs64_rot": CandidateStatus.EXPERIMENTAL,
    "turboquant_v2_b4_gs64_norot": CandidateStatus.EXPERIMENTAL,
    "polar_reference_offline_b4_d128": CandidateStatus.REFERENCE_ONLY,
}


def get_status_for_name(name: str) -> CandidateStatus:
    """Return the canonical status for a candidate name.

    Falls back to EXPERIMENTAL if unknown.
    """
    return CANDIDATE_STATUSES.get(name, CandidateStatus.EXPERIMENTAL)
