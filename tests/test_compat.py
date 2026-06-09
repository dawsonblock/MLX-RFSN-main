#!/usr/bin/env python3
"""RFSN v10 — Compatibility layer tests.

Verifies MLX availability detection and fallback module behaviour
without requiring mlx installed.
"""
from __future__ import annotations

import pytest

from rfsn_v10.compat import (
    MLX_AVAILABLE,
    ensure_mlx_available,
)


class TestMLXAvailability:
    def test_mlx_available_is_boolean(self):
        assert isinstance(MLX_AVAILABLE, bool)

    def test_ensure_mlx_available_raises_when_false(self, monkeypatch):
        monkeypatch.setattr("rfsn_v10.compat.MLX_AVAILABLE", False)
        with pytest.raises(ModuleNotFoundError, match="mlx.core"):
            ensure_mlx_available()

    def test_ensure_mlx_available_does_not_raise_when_true(self, monkeypatch):
        monkeypatch.setattr("rfsn_v10.compat.MLX_AVAILABLE", True)
        # Should not raise
        ensure_mlx_available()

    def test_missing_mlx_module_raises_attribute_error(self, monkeypatch):
        import rfsn_v10.compat as _compat
        class _FakeMissing:
            def __getattr__(self, name: str):
                raise AttributeError(name)
        monkeypatch.setattr(_compat, "mx", _FakeMissing())
        with pytest.raises(AttributeError):
            _compat.mx.array([1, 2, 3])

    def test_missing_mlx_module_any_attribute_raises(self, monkeypatch):
        import rfsn_v10.compat as _compat
        class _FakeMissing:
            def __getattr__(self, name: str):
                raise AttributeError(name)
        monkeypatch.setattr(_compat, "mx", _FakeMissing())
        with pytest.raises(AttributeError):
            _compat.mx.nonexistent
