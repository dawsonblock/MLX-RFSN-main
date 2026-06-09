#!/usr/bin/env python3
"""RFSN v10 — Health check tests.

Covers health status aggregation, checker registration, and default
health check behaviour without requiring MLX or external services.
"""
from __future__ import annotations

from rfsn_v10.health import (
    HealthChecker,
    HealthCheckResult,
    HealthStatus,
    check_disk_space,
    check_memory_usage,
    get_health_checker,
    setup_default_health_checker,
)

# ------------------------------------------------------------------
# HealthStatus
# ------------------------------------------------------------------

class TestHealthStatus:
    def test_enum_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


# ------------------------------------------------------------------
# HealthCheckResult
# ------------------------------------------------------------------

class TestHealthCheckResult:
    def test_to_dict(self):
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="all good",
            duration_ms=1.5,
            metadata={"extra": "data"},
        )
        d = result.to_dict()
        assert d["name"] == "test_check"
        assert d["status"] == "healthy"
        assert d["message"] == "all good"
        assert d["duration_ms"] == 1.5
        assert d["metadata"] == {"extra": "data"}

    def test_default_metadata(self):
        result = HealthCheckResult(
            name="minimal",
            status=HealthStatus.DEGRADED,
            message="meh",
            duration_ms=0.0,
        )
        assert result.metadata == {}


# ------------------------------------------------------------------
# HealthChecker
# ------------------------------------------------------------------

class TestHealthChecker:
    def test_register_and_run(self):
        checker = HealthChecker()
        def good_check() -> HealthCheckResult:
            return HealthCheckResult(
                name="good", status=HealthStatus.HEALTHY, message="ok", duration_ms=0.0
            )
        checker.register("good", good_check)
        result = checker.run_check("good")
        assert result is not None
        assert result.status == HealthStatus.HEALTHY

    def test_run_unknown_returns_none(self):
        checker = HealthChecker()
        assert checker.run_check("unknown") is None

    def test_failing_check_captured_as_unhealthy(self):
        checker = HealthChecker()
        def bad_check() -> HealthCheckResult:
            raise RuntimeError("boom")
        checker.register("bad", bad_check)
        result = checker.run_check("bad")
        assert result is not None
        assert result.status == HealthStatus.UNHEALTHY
        assert "boom" in result.message

    def test_overall_status_healthy(self):
        checker = HealthChecker()
        checker.last_results["a"] = HealthCheckResult(
            "a", HealthStatus.HEALTHY, "ok", 0.0
        )
        checker.last_results["b"] = HealthCheckResult(
            "b", HealthStatus.HEALTHY, "ok", 0.0
        )
        assert checker.get_overall_status() == HealthStatus.HEALTHY

    def test_overall_status_degraded(self):
        checker = HealthChecker()
        checker.last_results["a"] = HealthCheckResult(
            "a", HealthStatus.HEALTHY, "ok", 0.0
        )
        checker.last_results["b"] = HealthCheckResult(
            "b", HealthStatus.DEGRADED, "slow", 0.0
        )
        assert checker.get_overall_status() == HealthStatus.DEGRADED

    def test_overall_status_unhealthy(self):
        checker = HealthChecker()
        checker.last_results["a"] = HealthCheckResult(
            "a", HealthStatus.HEALTHY, "ok", 0.0
        )
        checker.last_results["b"] = HealthCheckResult(
            "b", HealthStatus.UNHEALTHY, "down", 0.0
        )
        assert checker.get_overall_status() == HealthStatus.UNHEALTHY

    def test_overall_status_no_results(self):
        """A checker that has not yet run any checks must report UNHEALTHY.

        A system that has not been checked cannot be declared healthy — this
        was a false-healthy bug that allowed the service to report 'healthy'
        before any health checks ran.
        """
        checker = HealthChecker()
        assert checker.get_overall_status() == HealthStatus.UNHEALTHY

    def test_run_all_checks(self):
        checker = HealthChecker()
        checker.register("c1", lambda: HealthCheckResult("c1", HealthStatus.HEALTHY, "", 0.0))
        checker.register("c2", lambda: HealthCheckResult("c2", HealthStatus.HEALTHY, "", 0.0))
        results = checker.run_all_checks()
        assert len(results) == 2
        assert results["c1"] is not None
        assert results["c2"] is not None

    def test_health_report_structure(self):
        checker = HealthChecker()
        checker.register("c1", lambda: HealthCheckResult("c1", HealthStatus.HEALTHY, "", 0.0))
        checker.run_all_checks()
        report = checker.get_health_report()
        assert "overall_status" in report
        assert "timestamp" in report
        assert "checks" in report
        assert report["overall_status"] == "healthy"


# ------------------------------------------------------------------
# Default health checks
# ------------------------------------------------------------------

class TestDiskSpaceCheck:
    def test_returns_result(self):
        result = check_disk_space("/tmp", threshold_gb=0.001)
        assert isinstance(result, HealthCheckResult)
        assert result.name == "disk_space"
        # Should be either healthy or degraded depending on actual disk space
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED)

    def test_empty_cache_dir_defaults_to_tmp(self):
        result = check_disk_space("", threshold_gb=10000.0)
        assert result.name == "disk_space"
        assert result.status is not None


class TestMemoryUsageCheck:
    def test_returns_result(self):
        result = check_memory_usage(threshold_gb=100.0)
        assert isinstance(result, HealthCheckResult)
        assert result.name == "memory_usage"
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)


# ------------------------------------------------------------------
# Singleton helpers
# ------------------------------------------------------------------

class TestSingletonHelpers:
    def test_get_health_checker_is_same_instance(self):
        h1 = get_health_checker()
        h2 = get_health_checker()
        assert h1 is h2

    def test_setup_default_health_checker_registers_checks(self):
        checker = setup_default_health_checker(cache_dir="/tmp")
        assert "metal_availability" in checker.checks
        assert "memory_usage" in checker.checks
        assert "disk_space" in checker.checks
