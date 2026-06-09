#!/usr/bin/env python3
"""Model-specific cache policies for RFSN v10.

Provides cache policies tailored to specific models
for optimal performance and resource utilization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class EvictionStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    ADAPTIVE = "adaptive"


@dataclass
class CachePolicy:
    """Cache policy configuration."""

    max_size_gb: float = 2.0
    eviction_strategy: EvictionStrategy = EvictionStrategy.LRU
    ttl_seconds: float | None = None
    min_access_count: int = 0
    priority_boost: float = 1.0


class ModelCachePolicyManager:
    """Manages cache policies for different models."""

    def __init__(self):
        """Initialize model cache policy manager."""
        self.policies: dict[str, CachePolicy] = {}
        self.access_counts: dict[str, dict[str, int]] = {}
        self.last_access: dict[str, dict[str, float]] = {}
        self.insertion_order: dict[str, list[str]] = {}

    def set_policy(self, model_id: str, policy: CachePolicy) -> None:
        """Set cache policy for a model.

        Args:
            model_id: Model identifier
            policy: Cache policy
        """
        self.policies[model_id] = policy
        self.access_counts[model_id] = {}
        self.last_access[model_id] = {}
        self.insertion_order[model_id] = []

    def get_policy(self, model_id: str) -> CachePolicy | None:
        """Get cache policy for a model.

        Args:
            model_id: Model identifier

        Returns:
            Cache policy or None if not found
        """
        return self.policies.get(model_id)

    def record_access(self, model_id: str, cache_key: str) -> None:
        """Record cache access for a model.

        Args:
            model_id: Model identifier
            cache_key: Cache key
        """
        if model_id not in self.access_counts:
            return

        self.access_counts[model_id][cache_key] = (
            self.access_counts[model_id].get(cache_key, 0) + 1
        )
        self.last_access[model_id][cache_key] = time.time()

    def record_insertion(self, model_id: str, cache_key: str) -> None:
        """Record cache insertion for a model.

        Args:
            model_id: Model identifier
            cache_key: Cache key
        """
        if model_id not in self.insertion_order:
            return

        if cache_key not in self.insertion_order[model_id]:
            self.insertion_order[model_id].append(cache_key)

    def get_eviction_candidates(
        self,
        model_id: str,
        current_size_gb: float,
        required_gb: float,
    ) -> list[str]:
        """Get cache keys to evict based on policy.

        Args:
            model_id: Model identifier
            current_size_gb: Current cache size in GB
            required_gb: Required space in GB

        Returns:
            List of cache keys to evict
        """
        policy = self.policies.get(model_id)
        if not policy:
            return []

        if current_size_gb + required_gb <= policy.max_size_gb:
            return []

        strategy = policy.eviction_strategy
        candidates = []

        if strategy == EvictionStrategy.LRU:
            candidates = self._get_lru_candidates(model_id)
        elif strategy == EvictionStrategy.LFU:
            candidates = self._get_lfu_candidates(model_id)
        elif strategy == EvictionStrategy.FIFO:
            candidates = self._get_fifo_candidates(model_id)
        elif strategy == EvictionStrategy.ADAPTIVE:
            candidates = self._get_adaptive_candidates(model_id)

        # Filter by minimum access count
        if policy.min_access_count > 0:
            access_counts = self.access_counts.get(model_id, {})
            candidates = [
                k for k in candidates if access_counts.get(k, 0) < policy.min_access_count
            ]

        # Filter by TTL
        if policy.ttl_seconds:
            last_access = self.last_access.get(model_id, {})
            current_time = time.time()
            candidates = [
                k
                for k in candidates
                if current_time - last_access.get(k, 0) > policy.ttl_seconds
            ]

        return candidates

    def _get_lru_candidates(self, model_id: str) -> list[str]:
        """Get LRU eviction candidates.

        Args:
            model_id: Model identifier

        Returns:
            List of cache keys sorted by least recently used
        """
        last_access = self.last_access.get(model_id, {})
        return sorted(last_access.keys(), key=lambda k: last_access.get(k, 0))

    def _get_lfu_candidates(self, model_id: str) -> list[str]:
        """Get LFU eviction candidates.

        Args:
            model_id: Model identifier

        Returns:
            List of cache keys sorted by least frequently used
        """
        access_counts = self.access_counts.get(model_id, {})
        return sorted(access_counts.keys(), key=lambda k: access_counts.get(k, 0))

    def _get_fifo_candidates(self, model_id: str) -> list[str]:
        """Get FIFO eviction candidates.

        Args:
            model_id: Model identifier

        Returns:
            List of cache keys in insertion order
        """
        return self.insertion_order.get(model_id, [])

    def _get_adaptive_candidates(self, model_id: str) -> list[str]:
        """Get adaptive eviction candidates.

        Combines LRU and LFU for adaptive eviction.

        Args:
            model_id: Model identifier

        Returns:
            List of cache keys sorted by adaptive score
        """
        last_access = self.last_access.get(model_id, {})
        access_counts = self.access_counts.get(model_id, {})

        def adaptive_score(key: str) -> float:
            # Lower score = higher eviction priority
            recency = last_access.get(key, 0)
            frequency = access_counts.get(key, 0)
            current_time = time.time()

            # Score combines recency and frequency
            recency_score = (current_time - recency) / 3600.0  # Hours since access
            frequency_score = 1.0 / (frequency + 1)  # Inverse frequency

            return recency_score * frequency_score

        return sorted(self.last_access.get(model_id, {}).keys(), key=adaptive_score)

    def get_cache_stats(self, model_id: str) -> dict[str, any]:
        """Get cache statistics for a model.

        Args:
            model_id: Model identifier

        Returns:
            Dictionary of cache statistics
        """
        access_counts = self.access_counts.get(model_id, {})
        self.last_access.get(model_id, {})

        total_accesses = sum(access_counts.values())
        avg_access_count = (
            total_accesses / len(access_counts) if access_counts else 0
        )

        return {
            "total_keys": len(access_counts),
            "total_accesses": total_accesses,
            "avg_access_count": avg_access_count,
            "policy": str(self.policies.get(model_id, EvictionStrategy.LRU)),
        }

    def clear_model_cache(self, model_id: str) -> None:
        """Clear cache for a model.

        Args:
            model_id: Model identifier
        """
        if model_id in self.access_counts:
            del self.access_counts[model_id]
        if model_id in self.last_access:
            del self.last_access[model_id]
        if model_id in self.insertion_order:
            del self.insertion_order[model_id]


def get_default_policy_for_model(model_id: str) -> CachePolicy:
    """Get default cache policy for a model.

    Args:
        model_id: Model identifier

    Returns:
        Default cache policy
    """
    # Adjust policy based on model characteristics
    if "gpt" in model_id.lower():
        # GPT models benefit from larger cache
        return CachePolicy(max_size_gb=4.0, eviction_strategy=EvictionStrategy.LRU)
    elif "llama" in model_id.lower():
        # LLaMA models benefit from adaptive eviction
        return CachePolicy(
            max_size_gb=3.0, eviction_strategy=EvictionStrategy.ADAPTIVE
        )
    else:
        # Default policy
        return CachePolicy(max_size_gb=2.0, eviction_strategy=EvictionStrategy.LRU)
