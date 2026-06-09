#!/usr/bin/env python3
"""Multi-model support for RFSN v10.

Provides model isolation, model-specific cache policies,
and model switching optimization for serving multiple models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from rfsn_v10.memory_manager import MemoryRegion, MultiTenantMemoryManager, TenantId


class ModelPriority(Enum):
    """Model priority for resource allocation."""

    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    priority: ModelPriority = ModelPriority.NORMAL
    max_cache_gb: float = 2.0
    sparse_ratio: float = 0.3
    quantization_bits: int = 8
    enable_wht: bool = True
    enable_incoherent_signs: bool = True
    block_size: int = 64


@dataclass
class ModelStats:
    """Statistics for a model."""

    model_id: str
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_latency_ms: float = 0.0
    last_access: float = field(default_factory=time.time)
    memory_usage_bytes: int = 0


class ModelIsolationManager:
    """Manages isolation between different models."""

    def __init__(self, memory_manager: MultiTenantMemoryManager | None = None):
        """Initialize model isolation manager.

        Args:
            memory_manager: Optional memory manager for tenant isolation
        """
        self.memory_manager = memory_manager or MultiTenantMemoryManager()
        self.models: dict[str, ModelConfig] = {}
        self.stats: dict[str, ModelStats] = {}
        self.active_model: str | None = None

    def register_model(self, config: ModelConfig) -> None:
        """Register a model with its configuration.

        Args:
            config: Model configuration
        """
        self.models[config.model_id] = config
        self.stats[config.model_id] = ModelStats(model_id=config.model_id)

        # Create tenant for model
        tenant_id = TenantId(f"model_{config.model_id}")
        self.memory_manager.set_tenant_quota(tenant_id, config.max_cache_gb * 1024**3)

    def get_model_config(self, model_id: str) -> ModelConfig | None:
        """Get configuration for a model.

        Args:
            model_id: Model identifier

        Returns:
            Model configuration or None if not found
        """
        return self.models.get(model_id)

    def get_model_stats(self, model_id: str) -> ModelStats | None:
        """Get statistics for a model.

        Args:
            model_id: Model identifier

        Returns:
            Model statistics or None if not found
        """
        return self.stats.get(model_id)

    def switch_model(self, model_id: str) -> bool:
        """Switch to a different model.

        Args:
            model_id: Model identifier to switch to

        Returns:
            True if switch successful, False otherwise
        """
        if model_id not in self.models:
            return False

        self.active_model = model_id
        self.stats[model_id].last_access = time.time()
        return True

    def record_request(self, model_id: str, latency_ms: float, cache_hit: bool) -> None:
        """Record a request for a model.

        Args:
            model_id: Model identifier
            latency_ms: Request latency in milliseconds
            cache_hit: Whether the request was a cache hit
        """
        if model_id not in self.stats:
            return

        stats = self.stats[model_id]
        stats.total_requests += 1

        if cache_hit:
            stats.cache_hits += 1
        else:
            stats.cache_misses += 1

        # Update average latency
        stats.avg_latency_ms = (
            stats.avg_latency_ms * (stats.total_requests - 1) + latency_ms
        ) / stats.total_requests

        stats.last_access = time.time()

    def allocate_model_memory(
        self,
        model_id: str,
        region: MemoryRegion,
        size_bytes: int,
    ) -> str | None:
        """Allocate memory for a model.

        Args:
            model_id: Model identifier
            region: Memory region
            size_bytes: Size in bytes

        Returns:
            Allocation ID or None if allocation failed
        """
        if model_id not in self.models:
            return None

        tenant_id = TenantId(f"model_{model_id}")
        alloc_id = self.memory_manager.allocate(tenant_id, region, size_bytes)

        if alloc_id:
            self.stats[model_id].memory_usage_bytes += size_bytes

        return alloc_id

    def release_model_memory(self, model_id: str, alloc_id: str) -> bool:
        """Release memory for a model.

        Args:
            model_id: Model identifier
            alloc_id: Allocation ID

        Returns:
            True if release successful
        """
        if model_id not in self.models:
            return False

        success = self.memory_manager.release(alloc_id)

        if success:
            # Update memory usage (approximate)
            alloc = self.memory_manager.allocations.get(alloc_id)
            if alloc:
                self.stats[model_id].memory_usage_bytes -= alloc.size_bytes

        return success

    def get_cache_hit_rate(self, model_id: str) -> float:
        """Get cache hit rate for a model.

        Args:
            model_id: Model identifier

        Returns:
            Cache hit rate (0-1)
        """
        stats = self.stats.get(model_id)
        if not stats or stats.total_requests == 0:
            return 0.0

        return stats.cache_hits / stats.total_requests

    def get_all_stats(self) -> dict[str, ModelStats]:
        """Get statistics for all models.

        Returns:
            Dictionary of model statistics
        """
        return self.stats.copy()

    def cleanup_inactive_models(self, inactive_threshold_seconds: float = 3600.0) -> int:
        """Cleanup models that have been inactive for a threshold.

        Args:
            inactive_threshold_seconds: Inactivity threshold in seconds

        Returns:
            Number of models cleaned up
        """
        current_time = time.time()
        cleaned = 0

        for model_id, stats in list(self.stats.items()):
            if current_time - stats.last_access > inactive_threshold_seconds:
                # Remove model
                del self.models[model_id]
                del self.stats[model_id]
                cleaned += 1

        return cleaned


class ModelSwitchingOptimizer:
    """Optimizes model switching for reduced latency."""

    def __init__(self, isolation_manager: ModelIsolationManager):
        """Initialize model switching optimizer.

        Args:
            isolation_manager: Model isolation manager
        """
        self.isolation_manager = isolation_manager
        self.switch_history: list[tuple[str, str, float]] = []

    def predict_switch_cost(self, from_model: str, to_model: str) -> float:
        """Predict the cost of switching between models.

        Args:
            from_model: Source model
            to_model: Target model

        Returns:
            Predicted cost in milliseconds
        """
        from_stats = self.isolation_manager.get_model_stats(from_model)
        to_stats = self.isolation_manager.get_model_stats(to_model)

        if not from_stats or not to_stats:
            return 100.0  # Default cost

        # Cost based on memory usage difference
        from_memory = from_stats.memory_usage_bytes
        to_memory = to_stats.memory_usage_bytes

        memory_diff = abs(to_memory - from_memory)
        memory_cost = memory_diff / (1024**3) * 50.0  # 50ms per GB

        # Cost based on cache hit rate (lower hit rate = higher cost)
        to_hit_rate = self.isolation_manager.get_cache_hit_rate(to_model)
        cache_cost = (1.0 - to_hit_rate) * 20.0  # Up to 20ms

        return memory_cost + cache_cost

    def record_switch(self, from_model: str, to_model: str, cost_ms: float) -> None:
        """Record a model switch.

        Args:
            from_model: Source model
            to_model: Target model
            cost_ms: Actual switch cost in milliseconds
        """
        self.switch_history.append((from_model, to_model, cost_ms))

        # Keep only last 100 switches
        if len(self.switch_history) > 100:
            self.switch_history = self.switch_history[-100:]

    def get_average_switch_cost(self) -> float:
        """Get average switch cost from history.

        Returns:
            Average switch cost in milliseconds
        """
        if not self.switch_history:
            return 0.0

        total_cost = sum(cost for _, _, cost in self.switch_history)
        return total_cost / len(self.switch_history)

    def recommend_model(self, context: dict[str, any]) -> str | None:
        """Recommend a model based on context.

        Args:
            context: Context information (e.g., request type, priority)

        Returns:
            Recommended model ID or None
        """
        # Simple recommendation based on priority
        high_priority_models = [
            model_id
            for model_id, config in self.isolation_manager.models.items()
            if config.priority == ModelPriority.HIGH
        ]

        if high_priority_models:
            # Return most recently accessed high priority model
            return max(
                high_priority_models,
                key=lambda m: self.isolation_manager.stats[m].last_access,
            )

        # Return most recently accessed model
        if self.isolation_manager.models:
            return max(
                self.isolation_manager.models.keys(),
                key=lambda m: self.isolation_manager.stats[m].last_access,
            )

        return None


def get_model_isolation_manager() -> ModelIsolationManager:
    """Get the global model isolation manager instance.

    Returns:
        ModelIsolationManager instance
    """
    global _model_isolation_manager
    if _model_isolation_manager is None:
        _model_isolation_manager = ModelIsolationManager()
    return _model_isolation_manager


_model_isolation_manager: ModelIsolationManager | None = None
