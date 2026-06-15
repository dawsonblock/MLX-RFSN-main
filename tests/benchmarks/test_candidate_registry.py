"""Verify candidate names and gate statuses are honest."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.portable


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
    """Verify _build_candidates() only uses valid config names (Phase 0 scope freeze).

    This test ensures that invalid candidate names like "k8_v5_gs32" cannot
    silently enter the registry and cause runtime failures. The gs32 path
    was explicitly moved to legacy status as "legacy_k8_v5_gs32".

    Phase 0: Only direct-packed candidate is active for correctness validation.
    """
    import sys
    from pathlib import Path

    # Add benchmarks to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))

    from benchmarks.kv_shootout import _build_candidates

    # P0 Fix: Use check_available=False to test registry structure without requiring MLX
    # This ensures the test is truly portable and validates declared candidates
    try:
        candidates = _build_candidates(quick=False, include_legacy=False, check_available=False)
        candidate_names = [c.name for c in candidates]

        # Verify gs32 is NOT in the active registry
        assert "rfsn_v10_k8_v5_gs32" not in candidate_names, (
            "Invalid gs32 config should not be in active registry"
        )

        # Phase 0: Verify direct-packed candidate IS in the active registry
        assert "rfsn_direct_packed_k8v8_gs64" in candidate_names, (
            "Direct-packed K8/V8 should be in active registry for correctness validation"
        )

        # Verify baseline is always present (canonical name is dense_mlx_baseline)
        assert "dense_mlx_baseline" in candidate_names, (
            "Baseline should be in active registry for comparison"
        )

    except ValueError as e:
        pytest.fail(f"_build_candidates() raised ValueError for valid registry: {e}")


@pytest.mark.unit
def test_declared_vs_available_candidates():
    """Verify declared_candidates works without MLX, available_candidates requires MLX.
    
    This test ensures the separation between declared and available candidates:
    - declared_candidates: returns all registered candidates (portable)
    - available_candidates: returns only candidates with satisfied dependencies
    """
    import sys
    from pathlib import Path

    # Add benchmarks to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))

    from benchmarks.candidate_registry import get_registry

    registry = get_registry()

    # declared_candidates should work without MLX (portable)
    declared = registry.declared_candidates()
    assert isinstance(declared, list)
    assert len(declared) > 0, "Should have declared candidates"

    # Verify baseline is always declared
    assert "dense_mlx_baseline" in declared

    # available_candidates may return fewer candidates if MLX is not installed
    available = registry.available_candidates()
    assert isinstance(available, list)

    # available should be a subset of declared
    for name in available:
        assert name in declared, f"Available candidate {name} not in declared list"

    # If MLX is not installed, available may be empty or only baseline
    # If MLX is installed, available should include MLX-dependent candidates
    # This test is portable and works in both cases
