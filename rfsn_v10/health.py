#!/usr/bin/env python3
"""Health check endpoints for RFSN v10.

Provides health status monitoring and readiness checks for production deployment.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


class HealthChecker:
    """Health check manager."""

    def __init__(self):
        self.checks: dict[str, Callable[[], HealthCheckResult]] = {}
        self.last_results: dict[str, HealthCheckResult] = {}

    def register(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Register a health check."""
        self.checks[name] = check_func

    def run_check(self, name: str) -> HealthCheckResult | None:
        """Run a specific health check."""
        if name not in self.checks:
            return None

        try:
            result = self.checks[name]()
            self.last_results[name] = result
            return result
        except Exception as e:
            result = HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                duration_ms=0.0,
            )
            self.last_results[name] = result
            return result

    def run_all_checks(self) -> dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall health status.

        Returns UNHEALTHY when no checks have been run yet, because a system
        that has not been checked cannot be declared healthy.
        """
        if not self.last_results:
            return HealthStatus.UNHEALTHY

        statuses = [r.status for r in self.last_results.values()]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.DEGRADED

    def get_health_report(self) -> dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "overall_status": self.get_overall_status().value,
            "timestamp": time.time(),
            "checks": {name: result.to_dict() for name, result in self.last_results.items()},
        }


# Default health checks
def check_metal_availability() -> HealthCheckResult:
    """Check if Metal is available."""
    start = time.perf_counter()

    try:
        from .compat import MLX_AVAILABLE, mx

        if not MLX_AVAILABLE:
            return HealthCheckResult(
                name="metal_availability",
                status=HealthStatus.DEGRADED,
                message="MLX not available",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        if not hasattr(mx, "fast") or not hasattr(mx.fast, "metal_kernel"):
            return HealthCheckResult(
                name="metal_availability",
                status=HealthStatus.DEGRADED,
                message="Metal kernels not available",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        return HealthCheckResult(
            name="metal_availability",
            status=HealthStatus.HEALTHY,
            message="Metal kernels available",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as e:
        return HealthCheckResult(
            name="metal_availability",
            status=HealthStatus.UNHEALTHY,
            message=f"Metal check failed: {str(e)}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


def check_memory_usage(threshold_gb: float = 8.0) -> HealthCheckResult:
    """Check memory usage."""
    start = time.perf_counter()

    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / 1024 / 1024 / 1024

        if memory_gb > threshold_gb:
            return HealthCheckResult(
                name="memory_usage",
                status=HealthStatus.DEGRADED,
                message=f"Memory usage high: {memory_gb:.2f}GB",
                duration_ms=(time.perf_counter() - start) * 1000,
                metadata={"memory_gb": memory_gb, "threshold_gb": threshold_gb},
            )

        return HealthCheckResult(
            name="memory_usage",
            status=HealthStatus.HEALTHY,
            message=f"Memory usage normal: {memory_gb:.2f}GB",
            duration_ms=(time.perf_counter() - start) * 1000,
            metadata={"memory_gb": memory_gb, "threshold_gb": threshold_gb},
        )
    except Exception as e:
        return HealthCheckResult(
            name="memory_usage",
            status=HealthStatus.UNHEALTHY,
            message=f"Memory check failed: {str(e)}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


def check_disk_space(cache_dir: str, threshold_gb: float = 1.0) -> HealthCheckResult:
    """Check disk space for cache."""
    start = time.perf_counter()

    try:
        import shutil

        cache_path = cache_dir if cache_dir else "/tmp"
        total, used, free = shutil.disk_usage(cache_path)
        free_gb = free / 1024 / 1024 / 1024

        if free_gb < threshold_gb:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.DEGRADED,
                message=f"Low disk space: {free_gb:.2f}GB free",
                duration_ms=(time.perf_counter() - start) * 1000,
                metadata={"free_gb": free_gb, "threshold_gb": threshold_gb},
            )

        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.HEALTHY,
            message=f"Disk space adequate: {free_gb:.2f}GB free",
            duration_ms=(time.perf_counter() - start) * 1000,
            metadata={"free_gb": free_gb, "threshold_gb": threshold_gb},
        )
    except Exception as e:
        return HealthCheckResult(
            name="disk_space",
            status=HealthStatus.UNHEALTHY,
            message=f"Disk space check failed: {str(e)}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


def setup_default_health_checker(cache_dir: str | None = None) -> HealthChecker:
    """Setup health checker with default checks."""
    checker = HealthChecker()

    checker.register("metal_availability", check_metal_availability)
    checker.register("memory_usage", lambda: check_memory_usage())
    checker.register("disk_space", lambda: check_disk_space(cache_dir or ""))

    return checker


# Singleton instance
_health_checker: HealthChecker | None = None


def get_health_checker(cache_dir: str | None = None) -> HealthChecker:
    """Get the global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = setup_default_health_checker(cache_dir)
    return _health_checker
