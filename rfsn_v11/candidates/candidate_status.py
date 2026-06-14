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
    LEGACY = "LEGACY"
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
    "legacy_k8_v5_gs32": CandidateStatus.LEGACY,
    "rfsn_v10_k8_v5_gs64": CandidateStatus.REFERENCE_ONLY,  # Dense reconstruction reference
    "rfsn_v10_legacy_k8_v5_gs32": CandidateStatus.LEGACY,
    "rfsn_v11_offline_asymmetric_kv_k8v5_gs64": CandidateStatus.OFFLINE_ONLY,
    "turboquant_v2_b4_gs64_rot": CandidateStatus.EXPERIMENTAL,
    "turboquant_v2_b4_gs64_norot": CandidateStatus.EXPERIMENTAL,
    "polar_reference_offline_b4_d128": CandidateStatus.REFERENCE_ONLY,
    # Alpha 9 TurboPolar — all experimental until gates pass
    "turbo_polar_offline": CandidateStatus.OFFLINE_ONLY,
    "turbo_polar_qjl": CandidateStatus.EXPERIMENTAL,
    "turbo_polar_metal_qk": CandidateStatus.EXPERIMENTAL,
    "turbo_polar_online_attention": CandidateStatus.EXPERIMENTAL,
}


def get_status_for_name(name: str) -> CandidateStatus:
    """Return the canonical status for a candidate name.

    Falls back to EXPERIMENTAL if unknown.
    """
    return CANDIDATE_STATUSES.get(name, CandidateStatus.EXPERIMENTAL)
