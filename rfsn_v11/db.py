"""rfsn_v11.db — Minimal SQLAlchemy metadata stub.

Defines the ``telemetry_events`` table that mirrors every field of
:class:`rfsn_v11.telemetry.TelemetryEvent`.  This module is the single source
of truth consumed by:

- ``rfsn_v11/alembic/env.py``  (target_metadata for autogenerate)
- ``rfsn_v11/alembic/versions/0001_create_telemetry_events.py``  (initial migration)

Column type mapping:
  str fields          → String
  int fields          → Integer
  float fields        → Float  (nullable where annotated ``float | None``)
  bool fields         → Boolean
  auto PK             → Integer (autoincrement)
  record timestamp    → DateTime
"""
from __future__ import annotations

import sqlalchemy as sa

metadata = sa.MetaData()

telemetry_events = sa.Table(
    "telemetry_events",
    metadata,
    # --- Primary key (synthetic, not part of TelemetryEvent dataclass) ---
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),

    # --- Identity / routing strings ---
    sa.Column("task_id",        sa.String, nullable=False),
    sa.Column("model_id",       sa.String, nullable=False),
    sa.Column("layer_id",       sa.String, nullable=False),
    sa.Column("batch_id",       sa.String, nullable=False),
    sa.Column("skill_pattern",  sa.String, nullable=False),

    # --- Sequence / head dimensions ---
    sa.Column("seq_len",           sa.Integer, nullable=False),
    sa.Column("head_count",        sa.Integer, nullable=False),
    sa.Column("head_dim",          sa.Integer, nullable=False),
    sa.Column("block_size",        sa.Integer, nullable=False),
    sa.Column("num_active_blocks", sa.Integer, nullable=False),

    # --- Sparsity / ratio scalars ---
    sa.Column("top_k_ratio",         sa.Float, nullable=False),
    sa.Column("effective_sparsity",  sa.Float, nullable=False),

    # --- Cache flags and latencies ---
    sa.Column("kv_cache_hit",                  sa.Boolean, nullable=False),
    sa.Column("kv_cache_store_latency_ms",     sa.Float,   nullable=False),
    sa.Column("kv_cache_retrieve_latency_ms",  sa.Float,   nullable=False),
    sa.Column("attention_latency_ms",          sa.Float,   nullable=False),
    sa.Column("total_latency_ms",              sa.Float,   nullable=False),

    # --- Execution outcome flags ---
    sa.Column("fallback_used",  sa.Boolean, nullable=False),
    sa.Column("sparse_success", sa.Boolean, nullable=False),
    sa.Column("dense_success",  sa.Boolean, nullable=False),

    # --- Audit gate and dense-path audit metrics (nullable) ---
    sa.Column("audit_enabled",       sa.Boolean, nullable=False),
    sa.Column("audit_cosine",        sa.Float,   nullable=True),
    sa.Column("audit_rel_mae",       sa.Float,   nullable=True),
    sa.Column("audit_max_abs_error", sa.Float,   nullable=True),

    # --- Quantization audit metrics (nullable) ---
    sa.Column("quant_audit_cosine",        sa.Float, nullable=True),
    sa.Column("quant_audit_rel_mae",       sa.Float, nullable=True),
    sa.Column("quant_audit_max_abs_error", sa.Float, nullable=True),

    # --- Sparse-path audit metrics (nullable) ---
    sa.Column("sparse_audit_cosine",        sa.Float, nullable=True),
    sa.Column("sparse_audit_rel_mae",       sa.Float, nullable=True),
    sa.Column("sparse_audit_max_abs_error", sa.Float, nullable=True),

    # --- Execution metadata ---
    sa.Column("execution_mode",    sa.String, nullable=False),
    sa.Column("termination_reason", sa.String, nullable=False),
)

__all__ = ["metadata", "telemetry_events"]
