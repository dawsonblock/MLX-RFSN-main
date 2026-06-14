"""Tests for MLX-LM version compatibility."""
from __future__ import annotations

import pytest

from rfsn_v10.integrations.mlx_lm_adapter.compatibility import (
    PINNED_MLX_LM_VERSION,
    PINNED_MLX_VERSION,
    check_mlx_lm_version,
    require_pinned_versions,
)


def test_pinned_constants_are_exact() -> None:
    assert PINNED_MLX_VERSION == "0.21.1"
    assert PINNED_MLX_LM_VERSION == "0.20.6"


def test_check_version_passes_with_installed() -> None:
    try:
        ok, msg = check_mlx_lm_version()
        # On this machine, the pinned versions are installed
        assert ok is True, msg
        assert "pinned pair verified" in msg
    except ImportError:
        pytest.skip("MLX not available on this platform")


def test_require_pinned_versions_passes() -> None:
    try:
        # Should not raise when versions match
        require_pinned_versions()
    except ImportError:
        pytest.skip("MLX not available on this platform")
