#!/usr/bin/env python3
"""Advanced memory management for RFSN v10.

Implements predictive eviction, multi-tenant isolation, quota enforcement,
and memory leak detection for production deployment.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TenantId(str):
    """Tenant identifier for multi-tenant isolation."""


class MemoryRegion(Enum):
    """Memory region types."""

    KV_CACHE = "kv_cache"
    ATTENTION = "attention"
    TEMPORARY = "temporary"


@dataclass
class MemoryAllocation:
    """A memory allocation record."""

    tenant_id: TenantId
    region: MemoryRegion
    size_bytes: int
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class PredictiveEvictionPolicy:
    """Predictive eviction policy based on access patterns."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.access_history: dict[str, list[float]] = defaultdict(list)

    def record_access(self, allocation_id: str, timestamp: float) -> None:
        """Record an access event."""
        self.access_history[allocation_id].append(timestamp)
        if len(self.access_history[allocation_id]) > self.window_size:
            self.access_history[allocation_id] = self.access_history[allocation_id][
                -self.window_size :
            ]

    def predict_future_access(self, allocation_id: str, horizon_seconds: float = 60.0) -> float:
        """Predict probability of future access."""
        history = self.access_history.get(allocation_id, [])
        if len(history) < 2:
            return 0.5  # Neutral prediction

        # Calculate access rate
        time_span = history[-1] - history[0]
        if time_span == 0:
            return 0.5

        access_rate = len(history) / time_span
        expected_accesses = access_rate * horizon_seconds

        # Normalize to probability
        return min(1.0, expected_accesses / 10.0)

    def get_eviction_candidates(
        self, allocations: dict[str, MemoryAllocation], n: int = 10
    ) -> list[str]:
        """Get top n candidates for eviction."""
        scores = []
        current_time = time.time()

        for alloc_id, alloc in allocations.items():
            # Score combines recency, access frequency, and predicted future access
            recency_score = 1.0 / (current_time - alloc.last_access + 1.0)
            freq_score = alloc.access_count / max(1, alloc.access_count)
            pred_score = self.predict_future_access(alloc_id)

            # Lower score = better eviction candidate
            score = 0.3 * recency_score + 0.3 * freq_score + 0.4 * pred_score
            scores.append((alloc_id, score))

        scores.sort(key=lambda x: x[1])
        return [alloc_id for alloc_id, _ in scores[:n]]


class MemoryQuotaManager:
    """Manages memory quotas per tenant."""

    def __init__(self, default_quota_gb: float = 4.0):
        self.default_quota = default_quota_gb * 1024 * 1024 * 1024
        self.quotas: dict[TenantId, int] = {}
        self.usage: dict[TenantId, int] = defaultdict(int)
        self.lock = threading.Lock()

    def set_quota(self, tenant_id: TenantId, quota_gb: float) -> None:
        """Set quota for a tenant."""
        with self.lock:
            self.quotas[tenant_id] = int(quota_gb * 1024 * 1024 * 1024)

    def get_quota(self, tenant_id: TenantId) -> int:
        """Get quota for a tenant."""
        return self.quotas.get(tenant_id, self.default_quota)

    def allocate(self, tenant_id: TenantId, size_bytes: int) -> bool:
        """Attempt to allocate memory for a tenant."""
        with self.lock:
            quota = self.get_quota(tenant_id)
            current_usage = self.usage[tenant_id]

            if current_usage + size_bytes <= quota:
                self.usage[tenant_id] = current_usage + size_bytes
                return True
            return False

    def release(self, tenant_id: TenantId, size_bytes: int) -> None:
        """Release memory for a tenant."""
        with self.lock:
            self.usage[tenant_id] = max(0, self.usage[tenant_id] - size_bytes)

    def get_usage(self, tenant_id: TenantId) -> int:
        """Get current usage for a tenant."""
        with self.lock:
            return self.usage[tenant_id]

    def get_utilization(self, tenant_id: TenantId) -> float:
        """Get utilization ratio for a tenant."""
        quota = self.get_quota(tenant_id)
        usage = self.get_usage(tenant_id)
        return usage / quota if quota > 0 else 0.0


class MemoryLeakDetector:
    """Detects potential memory leaks."""

    def __init__(self, threshold_mb: float = 100.0, window_minutes: int = 10):
        self.threshold = threshold_mb * 1024 * 1024
        self.window = window_minutes * 60
        self.snapshots: list[tuple[float, dict[str, int]]] = []

    def take_snapshot(self, allocations: dict[str, MemoryAllocation]) -> None:
        """Take a snapshot of current memory state."""
        current_time = time.time()
        usage_by_tenant: dict[str, int] = defaultdict(int)

        for alloc in allocations.values():
            usage_by_tenant[alloc.tenant_id] += alloc.size_bytes

        self.snapshots.append((current_time, dict(usage_by_tenant)))

        # Keep only recent snapshots
        cutoff = current_time - self.window
        self.snapshots = [(t, u) for t, u in self.snapshots if t > cutoff]

    def detect_leaks(self) -> dict[str, Any]:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 2:
            return {"leaks": []}

        oldest_snapshot = self.snapshots[0]
        newest_snapshot = self.snapshots[-1]

        time_delta = newest_snapshot[0] - oldest_snapshot[0]
        if time_delta < self.window / 2:
            return {"leaks": []}

        leaks = []
        for tenant_id in set(oldest_snapshot[1].keys()) | set(newest_snapshot[1].keys()):
            old_usage = oldest_snapshot[1].get(tenant_id, 0)
            new_usage = newest_snapshot[1].get(tenant_id, 0)
            growth = new_usage - old_usage

            if growth > self.threshold:
                growth_rate = growth / time_delta if time_delta > 0 else 0
                leaks.append(
                    {
                        "tenant_id": tenant_id,
                        "growth_bytes": growth,
                        "growth_rate_bytes_per_sec": growth_rate,
                        "old_usage_mb": old_usage / 1024 / 1024,
                        "new_usage_mb": new_usage / 1024 / 1024,
                    }
                )

        return {"leaks": leaks, "time_window_seconds": time_delta}


class MultiTenantMemoryManager:
    """Manages memory across multiple tenants with isolation."""

    def __init__(self, default_quota_gb: float = 4.0):
        self.allocations: dict[str, MemoryAllocation] = {}
        self.allocation_id_counter = 0
        self.lock = threading.Lock()

        self.quota_manager = MemoryQuotaManager(default_quota_gb)
        self.eviction_policy = PredictiveEvictionPolicy()
        self.leak_detector = MemoryLeakDetector()

    def allocate(
        self,
        tenant_id: TenantId,
        region: MemoryRegion,
        size_bytes: int,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """Allocate memory for a tenant."""
        with self.lock:
            # Check quota
            if not self.quota_manager.allocate(tenant_id, size_bytes):
                return None

            # Create allocation
            alloc_id = f"alloc_{self.allocation_id_counter}"
            self.allocation_id_counter += 1

            allocation = MemoryAllocation(
                tenant_id=tenant_id,
                region=region,
                size_bytes=size_bytes,
                timestamp=time.time(),
                last_access=time.time(),
                metadata=metadata or {},
            )

            self.allocations[alloc_id] = allocation
            return alloc_id

    def access(self, alloc_id: str) -> bool:
        """Record access to an allocation."""
        with self.lock:
            if alloc_id not in self.allocations:
                return False

            alloc = self.allocations[alloc_id]
            alloc.access_count += 1
            alloc.last_access = time.time()
            self.eviction_policy.record_access(alloc_id, alloc.last_access)
            return True

    def release(self, alloc_id: str) -> bool:
        """Release an allocation."""
        with self.lock:
            if alloc_id not in self.allocations:
                return False

            alloc = self.allocations[alloc_id]
            self.quota_manager.release(alloc.tenant_id, alloc.size_bytes)
            del self.allocations[alloc_id]
            return True

    def evict(self, n: int = 10) -> list[str]:
        """Evict n allocations based on policy."""
        with self.lock:
            candidates = self.eviction_policy.get_eviction_candidates(
                self.allocations, n
            )
            evicted = []
            for alloc_id in candidates:
                if self.release(alloc_id):
                    evicted.append(alloc_id)
            return evicted

    def enforce_quotas(self) -> dict[str, Any]:
        """Enforce quotas and evict if necessary."""
        violations = []
        for tenant_id in self.quota_manager.usage.keys():
            utilization = self.quota_manager.get_utilization(tenant_id)
            if utilization > 1.0:
                violations.append(
                    {"tenant_id": tenant_id, "utilization": utilization}
                )

        # Evict from over-quota tenants
        for violation in violations:
            tenant_id = TenantId(violation["tenant_id"])
            # Find allocations for this tenant
            tenant_allocs = [
                alloc_id
                for alloc_id, alloc in self.allocations.items()
                if alloc.tenant_id == tenant_id
            ]
            # Evict oldest
            for alloc_id in tenant_allocs[:5]:  # Evict up to 5
                self.release(alloc_id)

        return {"violations": violations, "evicted_count": len(violations)}

    def detect_leaks(self) -> dict[str, Any]:
        """Detect memory leaks."""
        with self.lock:
            self.leak_detector.take_snapshot(self.allocations)
            return self.leak_detector.detect_leaks()

    def get_stats(self) -> dict[str, Any]:
        """Get memory statistics."""
        with self.lock:
            total_usage = sum(alloc.size_bytes for alloc in self.allocations.values())
            usage_by_region = defaultdict(int)
            usage_by_tenant = defaultdict(int)

            for alloc in self.allocations.values():
                usage_by_region[alloc.region.value] += alloc.size_bytes
                usage_by_tenant[alloc.tenant_id] += alloc.size_bytes

            return {
                "total_allocations": len(self.allocations),
                "total_usage_bytes": total_usage,
                "usage_by_region": dict(usage_by_region),
                "usage_by_tenant": {str(k): v for k, v in usage_by_tenant.items()},
            }
