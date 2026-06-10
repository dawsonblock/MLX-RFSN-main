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
