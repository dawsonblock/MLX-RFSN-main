"""Create telemetry_events table.

Revision ID: 0001
Revises: (none — initial migration)
Create Date: 2025-01-01 00:00:00.000000

Creates the ``telemetry_events`` table, which stores every field of
:class:`rfsn_v11.telemetry.TelemetryEvent` plus a synthetic auto-increment
primary key ``id``.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# ---------------------------------------------------------------------------
# Alembic revision metadata
# ---------------------------------------------------------------------------
revision: str = "0001"
down_revision: str | None = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the telemetry_events table."""
    op.create_table(
        "telemetry_events",

        # --- Primary key (synthetic) ---
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),

        # --- Identity / routing strings ---
        sa.Column("task_id",        sa.String(), nullable=False),
        sa.Column("model_id",       sa.String(), nullable=False),
        sa.Column("layer_id",       sa.String(), nullable=False),
        sa.Column("batch_id",       sa.String(), nullable=False),
        sa.Column("skill_pattern",  sa.String(), nullable=False),

        # --- Sequence / head dimensions ---
        sa.Column("seq_len",           sa.Integer(), nullable=False),
        sa.Column("head_count",        sa.Integer(), nullable=False),
        sa.Column("head_dim",          sa.Integer(), nullable=False),
        sa.Column("block_size",        sa.Integer(), nullable=False),
        sa.Column("num_active_blocks", sa.Integer(), nullable=False),

        # --- Sparsity / ratio scalars ---
        sa.Column("top_k_ratio",        sa.Float(), nullable=False),
        sa.Column("effective_sparsity", sa.Float(), nullable=False),

        # --- Cache flags and latencies ---
        sa.Column("kv_cache_hit",                 sa.Boolean(), nullable=False),
        sa.Column("kv_cache_store_latency_ms",    sa.Float(),   nullable=False),
        sa.Column("kv_cache_retrieve_latency_ms", sa.Float(),   nullable=False),
        sa.Column("attention_latency_ms",         sa.Float(),   nullable=False),
        sa.Column("total_latency_ms",             sa.Float(),   nullable=False),

        # --- Execution outcome flags ---
        sa.Column("fallback_used",  sa.Boolean(), nullable=False),
        sa.Column("sparse_success", sa.Boolean(), nullable=False),
        sa.Column("dense_success",  sa.Boolean(), nullable=False),

        # --- Audit gate and dense-path audit metrics (nullable) ---
        sa.Column("audit_enabled",       sa.Boolean(), nullable=False),
        sa.Column("audit_cosine",        sa.Float(),   nullable=True),
        sa.Column("audit_rel_mae",       sa.Float(),   nullable=True),
        sa.Column("audit_max_abs_error", sa.Float(),   nullable=True),

        # --- Quantization audit metrics (nullable) ---
        sa.Column("quant_audit_cosine",        sa.Float(), nullable=True),
        sa.Column("quant_audit_rel_mae",       sa.Float(), nullable=True),
        sa.Column("quant_audit_max_abs_error", sa.Float(), nullable=True),

        # --- Sparse-path audit metrics (nullable) ---
        sa.Column("sparse_audit_cosine",        sa.Float(), nullable=True),
        sa.Column("sparse_audit_rel_mae",       sa.Float(), nullable=True),
        sa.Column("sparse_audit_max_abs_error", sa.Float(), nullable=True),

        # --- Execution metadata ---
        sa.Column("execution_mode",     sa.String(), nullable=False),
        sa.Column("termination_reason", sa.String(), nullable=False),
    )


def downgrade() -> None:
    """Drop the telemetry_events table."""
    op.drop_table("telemetry_events")
