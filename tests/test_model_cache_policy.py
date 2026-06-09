#!/usr/bin/env python3
"""RFSN v10 — Model cache policy tests.

Covers eviction strategies (LRU, LFU, FIFO, adaptive), policy management,
and cache statistics without external dependencies.
"""
from __future__ import annotations

import time

from rfsn_v10.model_cache_policy import (
    CachePolicy,
    EvictionStrategy,
    ModelCachePolicyManager,
    get_default_policy_for_model,
)

# ------------------------------------------------------------------
# CachePolicy
# ------------------------------------------------------------------

class TestCachePolicy:
    def test_default_values(self):
        p = CachePolicy()
        assert p.max_size_gb == 2.0
        assert p.eviction_strategy == EvictionStrategy.LRU
        assert p.ttl_seconds is None
        assert p.min_access_count == 0

    def test_custom_values(self):
        p = CachePolicy(
            max_size_gb=4.0,
            eviction_strategy=EvictionStrategy.LFU,
            ttl_seconds=3600.0,
            min_access_count=5,
        )
        assert p.max_size_gb == 4.0
        assert p.eviction_strategy == EvictionStrategy.LFU
        assert p.ttl_seconds == 3600.0
        assert p.min_access_count == 5


# ------------------------------------------------------------------
# EvictionStrategy
# ------------------------------------------------------------------

class TestEvictionStrategy:
    def test_enum_values(self):
        assert EvictionStrategy.LRU.value == "lru"
        assert EvictionStrategy.LFU.value == "lfu"
        assert EvictionStrategy.FIFO.value == "fifo"
        assert EvictionStrategy.ADAPTIVE.value == "adaptive"


# ------------------------------------------------------------------
# ModelCachePolicyManager
# ------------------------------------------------------------------

class TestModelCachePolicyManager:
    def test_set_and_get_policy(self):
        mgr = ModelCachePolicyManager()
        policy = CachePolicy(max_size_gb=8.0)
        mgr.set_policy("model_a", policy)
        assert mgr.get_policy("model_a") == policy
        assert mgr.get_policy("model_b") is None

    def test_record_access(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy())
        mgr.record_access("m", "k1")
        mgr.record_access("m", "k1")
        assert mgr.access_counts["m"]["k1"] == 2

    def test_record_access_unknown_model_ignored(self):
        mgr = ModelCachePolicyManager()
        mgr.record_access("unknown", "k")
        # Should not raise

    def test_record_insertion(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy())
        mgr.record_insertion("m", "k1")
        mgr.record_insertion("m", "k2")
        assert mgr.insertion_order["m"] == ["k1", "k2"]
        # Duplicate insertion should be ignored
        mgr.record_insertion("m", "k1")
        assert mgr.insertion_order["m"] == ["k1", "k2"]

    def test_lru_eviction(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy(max_size_gb=1.0, eviction_strategy=EvictionStrategy.LRU))
        mgr.record_access("m", "old")
        time.sleep(0.01)
        mgr.record_access("m", "new")
        candidates = mgr.get_eviction_candidates("m", current_size_gb=2.0, required_gb=1.0)
        assert candidates[0] == "old"

    def test_lfu_eviction(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy(max_size_gb=1.0, eviction_strategy=EvictionStrategy.LFU))
        mgr.record_access("m", "rare")
        for _ in range(5):
            mgr.record_access("m", "popular")
        candidates = mgr.get_eviction_candidates("m", current_size_gb=2.0, required_gb=1.0)
        assert candidates[0] == "rare"

    def test_fifo_eviction(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy(max_size_gb=1.0, eviction_strategy=EvictionStrategy.FIFO))
        mgr.record_insertion("m", "first")
        mgr.record_insertion("m", "second")
        candidates = mgr.get_eviction_candidates("m", current_size_gb=2.0, required_gb=1.0)
        assert candidates[0] == "first"

    def test_adaptive_eviction(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy(max_size_gb=1.0, eviction_strategy=EvictionStrategy.ADAPTIVE))
        mgr.record_access("m", "a")
        time.sleep(0.01)
        for _ in range(10):
            mgr.record_access("m", "b")
        candidates = mgr.get_eviction_candidates("m", current_size_gb=2.0, required_gb=1.0)
        # "a" is older and less frequent, so should be evicted first
        assert "a" in candidates

    def test_no_eviction_when_under_quota(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy(max_size_gb=10.0))
        mgr.record_access("m", "k")
        candidates = mgr.get_eviction_candidates("m", current_size_gb=2.0, required_gb=1.0)
        assert candidates == []

    def test_unknown_policy_returns_empty(self):
        mgr = ModelCachePolicyManager()
        candidates = mgr.get_eviction_candidates("unknown", current_size_gb=2.0, required_gb=1.0)
        assert candidates == []

    def test_ttl_filtering(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy(max_size_gb=1.0, ttl_seconds=0.001))
        mgr.record_access("m", "expired")
        time.sleep(0.01)
        mgr.record_access("m", "fresh")
        candidates = mgr.get_eviction_candidates("m", current_size_gb=2.0, required_gb=1.0)
        assert "expired" in candidates
        assert "fresh" not in candidates

    def test_min_access_count_filtering(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy(max_size_gb=1.0, min_access_count=3))
        mgr.record_access("m", "few")
        for _ in range(5):
            mgr.record_access("m", "many")
        candidates = mgr.get_eviction_candidates("m", current_size_gb=2.0, required_gb=1.0)
        assert "few" in candidates
        assert "many" not in candidates

    def test_cache_stats(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy())
        mgr.record_access("m", "k1")
        mgr.record_access("m", "k1")
        mgr.record_access("m", "k2")
        stats = mgr.get_cache_stats("m")
        assert stats["total_keys"] == 2
        assert stats["total_accesses"] == 3
        assert stats["avg_access_count"] == 1.5

    def test_clear_model_cache(self):
        mgr = ModelCachePolicyManager()
        mgr.set_policy("m", CachePolicy())
        mgr.record_access("m", "k")
        mgr.clear_model_cache("m")
        assert mgr.access_counts.get("m") is None
        assert mgr.last_access.get("m") is None


# ------------------------------------------------------------------
# get_default_policy_for_model
# ------------------------------------------------------------------

class TestDefaultPolicyForModel:
    def test_gpt_models_get_larger_cache(self):
        policy = get_default_policy_for_model("gpt-4")
        assert policy.max_size_gb == 4.0

    def test_non_gpt_models_get_standard_cache(self):
        policy = get_default_policy_for_model("qwen-7b")
        assert policy.max_size_gb == 2.0

    def test_gpt_case_insensitive(self):
        policy = get_default_policy_for_model("GPT-3")
        assert policy.max_size_gb == 4.0
