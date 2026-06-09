"""RFSN v11 Telemetry dataclasses.

Ported from rfsn_v10/runtime/engine.py TelemetryEvent.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TelemetryEvent:
    """Single decode-step telemetry record."""

    task_id: str
    model_id: str
    layer_id: str
    batch_id: str
    skill_pattern: str
    seq_len: int
    head_count: int
    head_dim: int
    top_k_ratio: float
    block_size: int
    num_active_blocks: int
    effective_sparsity: float
    kv_cache_hit: bool
    kv_cache_store_latency_ms: float
    kv_cache_retrieve_latency_ms: float
    attention_latency_ms: float
    total_latency_ms: float
    fallback_used: bool
    sparse_success: bool
    dense_success: bool
    audit_enabled: bool
    audit_cosine: float | None
    audit_rel_mae: float | None
    audit_max_abs_error: float | None
    quant_audit_cosine: float | None
    quant_audit_rel_mae: float | None
    quant_audit_max_abs_error: float | None
    sparse_audit_cosine: float | None
    sparse_audit_rel_mae: float | None
    sparse_audit_max_abs_error: float | None
    execution_mode: str
    termination_reason: str
