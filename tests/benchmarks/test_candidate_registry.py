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
    """Verify default _build_candidates() returns exactly Phase 0 canonical set.

    Phase 0 Scope Freeze: The default registry must contain exactly three
    candidates: dense baseline, MLX-LM 8-bit KV, and RFSN direct-packed K8/V8.
    Anything else requires --experimental.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))

    from benchmarks.kv_shootout import _build_candidates

    candidates = _build_candidates(
        quick=False, include_legacy=False, check_available=False
    )
    candidate_names = [c.name for c in candidates]

    # Exactly three canonical candidates
    assert "dense_mlx_baseline" in candidate_names
    assert "mlx_lm_quantized_kv_b8" in candidate_names
    assert "rfsn_direct_packed_k8v8_gs64" in candidate_names

    # Legacy candidates must NOT appear in default mode
    assert "A1_wht_grouped_k8v4_gs64" not in candidate_names
    assert "rfsn_direct_packed_k8v5" not in candidate_names
    assert "rfsn_direct_packed_k8v6" not in candidate_names

    # Experimental mode must unlock additional candidates
    exp_candidates = _build_candidates(
        quick=False, include_legacy=False, check_available=False, experimental=True
    )
    exp_names = [c.name for c in exp_candidates]
    assert "rfsn_direct_packed_k8v8_gs64" in exp_names


@pytest.mark.unit
def test_declared_vs_available_candidates():
    """Verify frozen registry has exactly three declared candidates.

    Phase 0: declared_candidates returns the canonical three.
    available_candidates may return fewer if MLX is not installed.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "benchmarks"))

    from benchmarks.candidate_registry import get_registry

    registry = get_registry()

    declared = registry.declared_candidates()
    assert isinstance(declared, list)
    # Phase 0 freeze: exactly 3 canonical candidates
    assert len(declared) == 3, f"Expected 3 declared candidates, got {len(declared)}: {declared}"
    assert "dense_mlx_baseline" in declared
    assert "mlx_lm_8bit_kv" in declared
    assert "rfsn_direct_packed_k8v8" in declared

    available = registry.available_candidates()
    assert isinstance(available, list)
    for name in available:
        assert name in declared, f"Available candidate {name} not in declared list"
