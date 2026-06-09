#!/usr/bin/env python3
"""RFSN v10 — Extended Memory Guard tests.

Covers pressure thresholds, emergency mode transitions, eviction callbacks,
and status reporting with MLX-aware memory tracking.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")

from rfsn_v10.memory_guard import MemoryGuard


class TestMemoryGuardInit:
    def test_default_init(self):
        mg = MemoryGuard()
        assert mg.safety_margin_gb == 0.5
        assert mg.soft_limit_gb is None
        assert mg.hard_limit_gb is None
        assert mg.eviction_callback is None

    def test_custom_limits(self):
        mg = MemoryGuard(
            safety_margin_gb=1.0,
            soft_limit_gb=8.0,
            hard_limit_gb=12.0,
        )
        assert mg.safety_margin_gb == 1.0
        assert mg.soft_limit_gb == 8.0
        assert mg.hard_limit_gb == 12.0


class TestCheckPressure:
    def test_no_pressure_with_no_limits(self):
        mg = MemoryGuard()
        assert mg.check_pressure(0) is False
        assert mg.check_pressure(1_000_000_000) is False

    def test_hard_limit_pressure(self):
        mg = MemoryGuard(hard_limit_gb=1.0)
        assert mg.check_pressure(2 * 1024 ** 3) is True

    def test_soft_limit_pressure(self):
        mg = MemoryGuard(soft_limit_gb=1.0)
        assert mg.check_pressure(2 * 1024 ** 3) is True

    def test_pressure_state_tracked(self):
        mg = MemoryGuard(hard_limit_gb=1.0)
        assert mg._pressure_active is False
        mg.check_pressure(2 * 1024 ** 3)
        assert mg._pressure_active is True

    def test_pressure_resets_when_under(self):
        mg = MemoryGuard(soft_limit_gb=10.0)
        mg.check_pressure(20 * 1024 ** 3)
        assert mg._pressure_active is True
        mg.check_pressure(1 * 1024 ** 3)
        assert mg._pressure_active is False


class TestEnforceSafety:
    def test_no_action_when_under_soft(self):
        mg = MemoryGuard(soft_limit_gb=10.0)
        freed = mg.enforce_safety(1 * 1024 ** 3)
        assert freed == 0
        assert mg._pressure_active is False

    def test_triggers_eviction_callback(self):
        callback_calls = []

        def callback(target_bytes):
            callback_calls.append(target_bytes)
            return target_bytes // 2

        mg = MemoryGuard(soft_limit_gb=1.0, eviction_callback=callback)
        freed = mg.enforce_safety(2 * 1024 ** 3)
        assert len(callback_calls) == 1
        assert freed == callback_calls[0] // 2
        assert mg._pressure_active is True

    def test_callback_exception_handled(self):
        def bad_callback(_):
            raise RuntimeError("boom")

        mg = MemoryGuard(soft_limit_gb=1.0, eviction_callback=bad_callback)
        freed = mg.enforce_safety(2 * 1024 ** 3)
        assert freed == 0
        assert mg._pressure_active is True

    def test_negative_cache_bytes_normalized(self):
        mg = MemoryGuard(soft_limit_gb=10.0)
        freed = mg.enforce_safety(-100)
        assert freed == 0

    def test_hard_limit_enters_emergency(self):
        mg = MemoryGuard(hard_limit_gb=1.0)
        mg.enforce_safety(2 * 1024 ** 3)
        assert mg._sparse_disabled is True
        assert mg._quantized_disabled is True


class TestEmergencyMode:
    def test_enter_emergency(self):
        mg = MemoryGuard()
        mg.enter_emergency_mode()
        assert mg._sparse_disabled is True
        assert mg._quantized_disabled is True
        assert mg._pressure_active is True

    def test_exit_emergency(self):
        mg = MemoryGuard()
        mg.enter_emergency_mode()
        mg.exit_emergency_mode()
        assert mg._sparse_disabled is False
        assert mg._quantized_disabled is False
        assert mg._pressure_active is False


class TestDisableFlags:
    def test_should_disable_sparse(self):
        mg = MemoryGuard()
        assert mg.should_disable_sparse() is False
        mg.enter_emergency_mode()
        assert mg.should_disable_sparse() is True

    def test_should_disable_quantized(self):
        mg = MemoryGuard()
        assert mg.should_disable_quantized() is False
        mg.enter_emergency_mode()
        assert mg.should_disable_quantized() is True

    def test_pressure_alone_disables(self):
        mg = MemoryGuard(soft_limit_gb=1.0)
        mg.check_pressure(10 * 1024 ** 3)
        assert mg.should_disable_sparse() is True
        assert mg.should_disable_quantized() is True


class TestStatus:
    def test_get_status_structure(self):
        mg = MemoryGuard(
            soft_limit_gb=4.0,
            hard_limit_gb=8.0,
            safety_margin_gb=0.5,
        )
        status = mg.get_status()
        assert "has_mlx_memory_api" in status
        assert "active_memory_bytes" in status
        assert "peak_memory_bytes" in status
        assert "pressure_active" in status
        assert "sparse_disabled" in status
        assert "quantized_disabled" in status
        assert status["safety_margin_gb"] == 0.5
        assert status["soft_limit_gb"] == 4.0
        assert status["hard_limit_gb"] == 8.0

    def test_status_reflects_emergency(self):
        mg = MemoryGuard()
        mg.enter_emergency_mode()
        status = mg.get_status()
        assert status["pressure_active"] is True
        assert status["sparse_disabled"] is True
        assert status["quantized_disabled"] is True


class TestMLXMemoryAPI:
    def test_check_mlx_api(self):
        mg = MemoryGuard()
        # Should not raise regardless of API availability
        assert isinstance(mg._has_mlx_memory_api, bool)

    def test_get_active_memory_returns_int(self):
        mg = MemoryGuard()
        mem = mg.get_active_memory_bytes()
        assert isinstance(mem, int)
        assert mem >= 0

    def test_get_peak_memory_returns_int(self):
        mg = MemoryGuard()
        mem = mg.get_peak_memory_bytes()
        assert isinstance(mem, int)
        assert mem >= 0
