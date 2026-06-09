#!/usr/bin/env python3
"""RFSN v10 — Memory manager tests.

Covers quota management, predictive eviction, leak detection,
and multi-tenant allocation without MLX.
"""
from __future__ import annotations

import time

from rfsn_v10.memory_manager import (
    MemoryAllocation,
    MemoryLeakDetector,
    MemoryQuotaManager,
    MemoryRegion,
    MultiTenantMemoryManager,
    PredictiveEvictionPolicy,
    TenantId,
)

# ------------------------------------------------------------------
# TenantId & MemoryRegion
# ------------------------------------------------------------------

class TestEnums:
    def test_tenant_id_is_str(self):
        t = TenantId("tenant_1")
        assert isinstance(t, str)
        assert t == "tenant_1"

    def test_memory_region_values(self):
        assert MemoryRegion.KV_CACHE.value == "kv_cache"
        assert MemoryRegion.ATTENTION.value == "attention"
        assert MemoryRegion.TEMPORARY.value == "temporary"


# ------------------------------------------------------------------
# MemoryAllocation
# ------------------------------------------------------------------

class TestMemoryAllocation:
    def test_creation(self):
        alloc = MemoryAllocation(
            tenant_id=TenantId("t1"),
            region=MemoryRegion.KV_CACHE,
            size_bytes=1024,
            timestamp=time.time(),
        )
        assert alloc.tenant_id == "t1"
        assert alloc.region == MemoryRegion.KV_CACHE
        assert alloc.size_bytes == 1024
        assert alloc.metadata == {}


# ------------------------------------------------------------------
# PredictiveEvictionPolicy
# ------------------------------------------------------------------

class TestPredictiveEvictionPolicy:
    def test_record_and_predict(self):
        policy = PredictiveEvictionPolicy()
        now = time.time()
        policy.record_access("a", now)
        policy.record_access("a", now + 1)
        prob = policy.predict_future_access("a")
        assert 0.0 <= prob <= 1.0

    def test_neutral_prediction_for_single_access(self):
        policy = PredictiveEvictionPolicy()
        policy.record_access("a", time.time())
        assert policy.predict_future_access("a") == 0.5

    def test_neutral_prediction_for_no_history(self):
        policy = PredictiveEvictionPolicy()
        assert policy.predict_future_access("unknown") == 0.5

    def test_window_size_limit(self):
        policy = PredictiveEvictionPolicy(window_size=5)
        for i in range(10):
            policy.record_access("a", time.time() + i)
        assert len(policy.access_history["a"]) == 5

    def test_get_eviction_candidates(self):
        policy = PredictiveEvictionPolicy()
        now = time.time()
        allocs = {
            "old": MemoryAllocation(TenantId("t"), MemoryRegion.KV_CACHE, 100, now - 10, last_access=now - 10, access_count=1),
            "new": MemoryAllocation(TenantId("t"), MemoryRegion.KV_CACHE, 100, now, last_access=now, access_count=10),
        }
        candidates = policy.get_eviction_candidates(allocs, n=2)
        assert len(candidates) == 2
        # "old" should be first candidate (lower score = better candidate)
        assert candidates[0] == "old"


# ------------------------------------------------------------------
# MemoryQuotaManager
# ------------------------------------------------------------------

class TestMemoryQuotaManager:
    def test_default_quota(self):
        mgr = MemoryQuotaManager()
        assert mgr.default_quota == 4.0 * 1024 * 1024 * 1024

    def test_set_and_get_quota(self):
        mgr = MemoryQuotaManager()
        mgr.set_quota(TenantId("t1"), 8.0)
        assert mgr.get_quota(TenantId("t1")) == 8 * 1024 * 1024 * 1024

    def test_allocate_within_quota(self):
        mgr = MemoryQuotaManager(default_quota_gb=1.0)
        assert mgr.allocate(TenantId("t1"), 100) is True
        assert mgr.get_usage(TenantId("t1")) == 100

    def test_allocate_exceeds_quota(self):
        mgr = MemoryQuotaManager(default_quota_gb=1.0)
        quota = mgr.get_quota(TenantId("t1"))
        assert mgr.allocate(TenantId("t1"), quota + 1) is False

    def test_release(self):
        mgr = MemoryQuotaManager(default_quota_gb=1.0)
        mgr.allocate(TenantId("t1"), 1000)
        mgr.release(TenantId("t1"), 500)
        assert mgr.get_usage(TenantId("t1")) == 500

    def test_release_does_not_go_negative(self):
        mgr = MemoryQuotaManager(default_quota_gb=1.0)
        mgr.allocate(TenantId("t1"), 100)
        mgr.release(TenantId("t1"), 200)
        assert mgr.get_usage(TenantId("t1")) == 0

    def test_utilization(self):
        mgr = MemoryQuotaManager(default_quota_gb=1.0)
        quota = mgr.get_quota(TenantId("t1"))
        mgr.allocate(TenantId("t1"), quota // 2)
        assert mgr.get_utilization(TenantId("t1")) == 0.5

    def test_utilization_zero_quota(self):
        mgr = MemoryQuotaManager(default_quota_gb=0.0)
        assert mgr.get_utilization(TenantId("t1")) == 0.0


# ------------------------------------------------------------------
# MemoryLeakDetector
# ------------------------------------------------------------------

class TestMemoryLeakDetector:
    def test_no_leaks_with_few_snapshots(self):
        detector = MemoryLeakDetector()
        allocs = {"a": MemoryAllocation(TenantId("t"), MemoryRegion.KV_CACHE, 100, time.time())}
        detector.take_snapshot(allocs)
        result = detector.detect_leaks()
        assert result["leaks"] == []

    def test_detects_growth(self, monkeypatch):
        detector = MemoryLeakDetector(threshold_mb=1.0, window_minutes=60)
        now = 10000.0
        # Mock time.time so snapshots are far apart
        monkeypatch.setattr("time.time", lambda: now)
        # First snapshot: small usage
        allocs1 = {"a": MemoryAllocation(TenantId("t"), MemoryRegion.KV_CACHE, 100, now)}
        detector.take_snapshot(allocs1)
        # Second snapshot: large growth (must be > window/2 = 1800s apart to count)
        monkeypatch.setattr("time.time", lambda: now + 2000)
        allocs2 = {"a": MemoryAllocation(TenantId("t"), MemoryRegion.KV_CACHE, 10 * 1024 * 1024, now + 2000)}
        detector.take_snapshot(allocs2)
        result = detector.detect_leaks()
        assert len(result["leaks"]) >= 1
        assert result["leaks"][0]["tenant_id"] == "t"

    def test_old_snapshots_pruned(self, monkeypatch):
        detector = MemoryLeakDetector(window_minutes=1)
        now = 1000.0
        allocs = {"a": MemoryAllocation(TenantId("t"), MemoryRegion.KV_CACHE, 100, now)}
        monkeypatch.setattr("time.time", lambda: now)
        detector.take_snapshot(allocs)
        assert len(detector.snapshots) == 1
        # Simulate time passing beyond window
        monkeypatch.setattr("time.time", lambda: now + 120)
        future_allocs = {"a": MemoryAllocation(TenantId("t"), MemoryRegion.KV_CACHE, 100, now + 120)}
        detector.take_snapshot(future_allocs)
        # Old snapshot should be pruned
        assert len(detector.snapshots) == 1


# ------------------------------------------------------------------
# MultiTenantMemoryManager
# ------------------------------------------------------------------

class TestMultiTenantMemoryManager:
    def test_allocate_success(self):
        mgr = MultiTenantMemoryManager(default_quota_gb=1.0)
        alloc_id = mgr.allocate(TenantId("t1"), MemoryRegion.KV_CACHE, 1024)
        assert alloc_id is not None
        assert alloc_id.startswith("alloc_")

    def test_allocate_exceeds_quota(self):
        mgr = MultiTenantMemoryManager(default_quota_gb=0.000001)
        alloc_id = mgr.allocate(TenantId("t1"), MemoryRegion.KV_CACHE, 1024 * 1024)
        assert alloc_id is None

    def test_access_recorded(self):
        mgr = MultiTenantMemoryManager(default_quota_gb=1.0)
        alloc_id = mgr.allocate(TenantId("t1"), MemoryRegion.KV_CACHE, 100)
        mgr.access(alloc_id)
        alloc = mgr.allocations[alloc_id]
        assert alloc.access_count == 1
        assert alloc.last_access > 0

    def test_access_unknown_ignored(self):
        mgr = MultiTenantMemoryManager()
        result = mgr.access("nonexistent")
        assert result is False

    def test_get_usage_by_tenant(self):
        mgr = MultiTenantMemoryManager(default_quota_gb=1.0)
        mgr.allocate(TenantId("t1"), MemoryRegion.KV_CACHE, 100)
        mgr.allocate(TenantId("t1"), MemoryRegion.KV_CACHE, 200)
        mgr.allocate(TenantId("t2"), MemoryRegion.KV_CACHE, 50)
        stats = mgr.get_stats()
        assert stats["usage_by_tenant"]["t1"] == 300
        assert stats["usage_by_tenant"]["t2"] == 50
        assert stats["usage_by_tenant"].get("t3", 0) == 0
