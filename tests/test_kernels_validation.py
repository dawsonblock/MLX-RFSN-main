#!/usr/bin/env python3
"""RFSN v10 — Kernel route validation and helper tests.

Covers sequential_reference_route_supported, KernelRouteError,
and kernel helper validation without requiring Metal kernel execution.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.kernels import (
    KernelRouteError,
    maybe_supports_metal_kernels,
    sequential_reference_route_supported,
)


class TestSequentialReferenceRouteSupported:
    def test_requires_wht(self):
        ok, reason = sequential_reference_route_supported(
            shape=(1, 4, 64, 64),
            out_dtype=mx.float32,
            use_wht=False,
            use_incoherent_signs=True,
        )
        assert ok is False
        assert "requires_wht" in reason

    def test_requires_incoherent_signs(self):
        ok, reason = sequential_reference_route_supported(
            shape=(1, 4, 64, 64),
            out_dtype=mx.float32,
            use_wht=True,
            use_incoherent_signs=False,
        )
        assert ok is False
        assert "requires_incoherent_signs" in reason

    def test_requires_supported_dtype(self):
        ok, reason = sequential_reference_route_supported(
            shape=(1, 4, 64, 64),
            out_dtype=mx.int32,
            use_wht=True,
            use_incoherent_signs=True,
        )
        assert ok is False
        assert "out_dtype_unsupported" in reason

    def test_requires_rank_4(self):
        ok, reason = sequential_reference_route_supported(
            shape=(64,),
            out_dtype=mx.float32,
            use_wht=True,
            use_incoherent_signs=True,
        )
        assert ok is False
        assert "shape_rank_unsupported" in reason

    def test_requires_head_dim_multiple_of_64(self):
        ok, reason = sequential_reference_route_supported(
            shape=(1, 4, 64, 32),
            out_dtype=mx.float32,
            use_wht=True,
            use_incoherent_signs=True,
        )
        assert ok is False
        assert "head_dim_unsupported" in reason

    def test_all_conditions_met(self):
        ok, reason = sequential_reference_route_supported(
            shape=(1, 4, 64, 64),
            out_dtype=mx.float16,
            use_wht=True,
            use_incoherent_signs=True,
        )
        assert ok is True
        assert "supported" in reason


class TestMaybeSupportsMetalKernels:
    def test_returns_boolean(self):
        result = maybe_supports_metal_kernels()
        assert isinstance(result, bool)

    def test_consistent_on_repeated_calls(self):
        r1 = maybe_supports_metal_kernels()
        r2 = maybe_supports_metal_kernels()
        assert r1 == r2


class TestKernelRouteError:
    def test_is_runtime_error(self):
        assert issubclass(KernelRouteError, RuntimeError)

    def test_can_be_raised(self):
        with pytest.raises(KernelRouteError, match="unavailable"):
            raise KernelRouteError("metal_kernel_api_unavailable")
