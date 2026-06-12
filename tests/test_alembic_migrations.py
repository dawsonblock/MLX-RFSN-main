"""Ticket 4-4: Alembic migration smoke tests.

Verifies that:
- ``alembic upgrade head`` runs against a fresh SQLite DB without error.
- The expected ``telemetry_events`` table and columns are created.
- ``alembic downgrade base`` rolls back cleanly (table removed).
- ``alembic check`` detects no unmigrated model changes (schema drift gate).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

alembic = pytest.importorskip("alembic", reason="alembic not installed")
sqlalchemy = pytest.importorskip("sqlalchemy", reason="sqlalchemy not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
_ALEMBIC_INI = _REPO_ROOT / "rfsn_v11" / "alembic.ini"
_SCRIPT_LOCATION = _REPO_ROOT / "rfsn_v11" / "alembic"

pytestmark = [pytest.mark.integration, pytest.mark.db]


def _run_alembic(cmd: list[str], db_url: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "RFSN_DB_URL": db_url}
    return subprocess.run(
        [sys.executable, "-m", "alembic", "-c", str(_ALEMBIC_INI)] + cmd,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_alembic_upgrade_head(tmp_path):
    """alembic upgrade head must succeed against a fresh SQLite database."""
    db_file = tmp_path / "test_telemetry.db"
    db_url = f"sqlite:///{db_file}"

    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, (
        f"alembic upgrade head failed:\nSTDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def test_telemetry_events_table_exists(tmp_path):
    """After upgrade head the telemetry_events table must exist with all columns.
    """
    import sqlalchemy as sa

    db_file = tmp_path / "test_schema.db"
    db_url = f"sqlite:///{db_file}"

    result = _run_alembic(["upgrade", "head"], db_url)
    assert result.returncode == 0, result.stderr

    engine = sa.create_engine(db_url)
    inspector = sa.inspect(engine)
    tables = inspector.get_table_names()
    assert "telemetry_events" in tables, (
        f"telemetry_events not found in tables: {tables}"
    )

    columns = {col["name"] for col in inspector.get_columns("telemetry_events")}
    required = {
        "id", "task_id", "model_id", "layer_id", "batch_id", "skill_pattern",
        "seq_len", "head_count", "head_dim", "block_size", "num_active_blocks",
        "top_k_ratio", "effective_sparsity",
        "kv_cache_hit", "kv_cache_store_latency_ms", "kv_cache_retrieve_latency_ms",
        "attention_latency_ms", "total_latency_ms",
        "fallback_used", "sparse_success", "dense_success",
        "audit_enabled", "audit_cosine", "audit_rel_mae", "audit_max_abs_error",
        "quant_audit_cosine", "quant_audit_rel_mae", "quant_audit_max_abs_error",
        "sparse_audit_cosine", "sparse_audit_rel_mae", "sparse_audit_max_abs_error",
        "execution_mode", "termination_reason",
    }
    missing = required - columns
    assert not missing, f"Missing columns in telemetry_events: {missing}"
    engine.dispose()


def test_alembic_downgrade_base(tmp_path):
    """alembic downgrade base must roll back cleanly (table removed)."""
    import sqlalchemy as sa

    db_file = tmp_path / "test_downgrade.db"
    db_url = f"sqlite:///{db_file}"

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, up.stderr

    down = _run_alembic(["downgrade", "base"], db_url)
    assert down.returncode == 0, (
        f"alembic downgrade base failed:\nSTDOUT:\n{down.stdout}\nSTDERR:\n{down.stderr}"
    )

    engine = sa.create_engine(db_url)
    inspector = sa.inspect(engine)
    tables = inspector.get_table_names()
    assert "telemetry_events" not in tables, (
        f"telemetry_events still present after downgrade: {tables}"
    )
    engine.dispose()


def test_alembic_check_no_drift(tmp_path):
    """alembic check must detect no unmigrated schema changes."""
    db_file = tmp_path / "test_drift.db"
    db_url = f"sqlite:///{db_file}"

    up = _run_alembic(["upgrade", "head"], db_url)
    assert up.returncode == 0, up.stderr

    check = _run_alembic(["check"], db_url)
    assert check.returncode == 0, (
        f"alembic check detected schema drift:\nSTDOUT:\n{check.stdout}\nSTDERR:\n{check.stderr}"
    )
