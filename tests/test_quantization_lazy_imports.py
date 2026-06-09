#!/usr/bin/env python3
"""RFSN v10 — Quantization package lazy-import tests.

Verifies that MLX-dependent symbols are loaded lazily and that
dir() / getattr() work correctly without requiring mlx installed.
"""
from __future__ import annotations

import importlib.util

import pytest

# layer_policy is eagerly imported and should always be available
from rfsn_v10.quantization import (  # noqa: I001
    KNOWN_MODES,
    LayerPolicy,
    __all__,
    load_policy,
    validate_layer_policy,
)


class TestLazyImportDir:
    def test_all_contains_lazy_names(self):
        assert "PolarQuantizer" in __all__
        assert "QuantizedKVManager" in __all__
        # Explicitly reference eagerly-imported symbols to satisfy lint
        assert LayerPolicy is not None
        assert load_policy is not None
        assert validate_layer_policy is not None
        assert isinstance(KNOWN_MODES, frozenset)

    def test_dir_returns_sorted_all(self):
        names = __import__(
            "rfsn_v10.quantization",
            fromlist=["__dir__"],
        ).__dir__()
        assert isinstance(names, list)
        assert "LayerPolicy" in names
        assert "PolarQuantizer" in names
        assert names == sorted(names)


class TestLazyGetattr:
    def test_getattr_raises_for_unknown(self):
        mod = __import__("rfsn_v10.quantization", fromlist=["__getattr__"])
        with pytest.raises(
            AttributeError, match="has no attribute 'NonExistent'",
        ):
            mod.__getattr__("NonExistent")

    @pytest.mark.skipif(
        importlib.util.find_spec("mlx") is None,
        reason="mlx not installed",
    )
    def test_lazy_import_resolves_when_mlx_present(self):
        # When mlx is present, lazy imports should resolve
        from rfsn_v10.quantization import PolarQuantizer  # noqa: F811
        assert PolarQuantizer is not None
