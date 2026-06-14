"""Verify candidate names and gate statuses are honest."""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_rfsn_v11_name_is_offline():
    """RFSN v11 must be labeled as offline until real cache injection exists."""
    from rfsn_v11.candidates.rfsn_v11_adapter import RFSNV11Candidate
    c = RFSNV11Candidate()
    assert "offline" in c.name
    assert "real_generation" not in c.name


@pytest.mark.unit
def test_polar_reference_name_is_reference():
    """Polar reference must be labeled as reference, not a speed winner."""
    from rfsn_v11.candidates.polar_reference_adapter import PolarReferenceAdapter
    c = PolarReferenceAdapter()
    assert "reference" in c.name


@pytest.mark.unit
def test_turboquant_v2_name_includes_config():
    """TurboQuant V2 name must include bits and group_size."""
    from rfsn_v11.candidates.turboquant_v2_adapter import TurboQuantV2Candidate
    c = TurboQuantV2Candidate(bits=4, group_size=64)
    assert "turboquant_v2" in c.name
    assert "b4" in c.name
    assert "gs64" in c.name


@pytest.mark.unit
def test_build_candidates_registry_valid():
    """Verify _build_candidates() only uses valid config names.

    This test ensures that invalid candidate names like "k8_v5_gs32" cannot
    silently enter the registry and cause runtime failures. The gs32 path
    was explicitly moved to legacy status as "legacy_k8_v5_gs32".
    """
    import sys
    from pathlib import Path

    # Add benchmarks to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))

    from benchmarks.kv_shootout import _build_candidates

    # Mock availability to test registry construction without MLX
    # This should not raise ValueError for invalid config names
    try:
        candidates = _build_candidates(quick=False, include_legacy=False)
        candidate_names = [c.name for c in candidates]

        # Verify gs32 is NOT in the active registry
        assert "rfsn_v10_k8_v5_gs32" not in candidate_names, (
            "Invalid gs32 config should not be in active registry"
        )

        # Verify gs64 IS in the active registry
        assert "rfsn_v10_k8_v5_gs64" in candidate_names, (
            "Canonical gs64 config should be in active registry"
        )

        # Verify legacy gs32 only appears when explicitly requested
        candidates_with_legacy = _build_candidates(quick=False, include_legacy=True)
        legacy_names = [c.name for c in candidates_with_legacy]
        assert "rfsn_v10_legacy_k8_v5_gs32" in legacy_names, (
            "Legacy gs32 should appear when --include-legacy is set"
        )

    except ValueError as e:
        pytest.fail(f"_build_candidates() raised ValueError for valid registry: {e}")
