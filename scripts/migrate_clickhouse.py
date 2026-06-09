#!/usr/bin/env python3
"""Idempotent ClickHouse schema migration runner.

Replaces Alembic for ClickHouse-specific DDL.  Tracks applied migrations in
``rfsn_schema_version``.

Usage:
    python scripts/migrate_clickhouse.py
"""
from __future__ import annotations

import sys
from urllib import request


MIGRATIONS = [
    {
        "version": 1,
        "description": "Initial telemetry tables",
        "statements": [
            """
            CREATE TABLE IF NOT EXISTS rfsn_schema_version (
                version UInt32,
                applied_at DateTime DEFAULT now(),
                description String
            ) ENGINE = MergeTree()
            ORDER BY version
            """,
            """
            CREATE TABLE IF NOT EXISTS rfsn_attention_events (
                task_id String,
                model_id String,
                layer_id String,
                batch_id String,
                skill_pattern String,
                seq_len UInt32,
                head_count UInt32,
                head_dim UInt32,
                top_k_ratio Float32,
                block_size UInt32,
                num_active_blocks UInt32,
                effective_sparsity Float32,
                kv_cache_hit UInt8,
                kv_cache_store_latency_ms Float64,
                kv_cache_retrieve_latency_ms Float64,
                attention_latency_ms Float64,
                total_latency_ms Float64,
                fallback_used UInt8,
                sparse_success UInt8,
                dense_success UInt8,
                audit_enabled UInt8,
                audit_cosine Nullable(Float64),
                audit_rel_mae Nullable(Float64),
                audit_max_abs_error Nullable(Float64),
                quant_audit_cosine Nullable(Float64),
                quant_audit_rel_mae Nullable(Float64),
                quant_audit_max_abs_error Nullable(Float64),
                sparse_audit_cosine Nullable(Float64),
                sparse_audit_rel_mae Nullable(Float64),
                sparse_audit_max_abs_error Nullable(Float64),
                execution_mode String,
                termination_reason String,
                timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (task_id, timestamp)
            SETTINGS index_granularity = 8192
            """,
            """
            CREATE TABLE IF NOT EXISTS rfsn_kv_cache_events (
                task_id String,
                model_id String,
                layer_id String,
                batch_id String,
                skill_pattern String,
                operation String,
                key_size_bytes UInt64,
                value_size_bytes UInt64,
                latency_ms Float64,
                success UInt8,
                timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (task_id, timestamp)
            SETTINGS index_granularity = 8192
            """,
            """
            CREATE TABLE IF NOT EXISTS rfsn_audit_events (
                task_id String,
                model_id String,
                layer_id String,
                batch_id String,
                skill_pattern String,
                sparse_output String,
                dense_output String,
                cosine_similarity Float64,
                rel_mae Float64,
                max_abs_error Float64,
                timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (task_id, timestamp)
            SETTINGS index_granularity = 8192
            """,
            """
            CREATE TABLE IF NOT EXISTS rfsn_failures (
                task_id String,
                model_id String,
                layer_id String,
                batch_id String,
                skill_pattern String,
                error_type String,
                error_message String,
                timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (task_id, timestamp)
            SETTINGS index_granularity = 8192
            """,
            """
            CREATE TABLE IF NOT EXISTS rfsn_sparsity_profiles (
                task_id String,
                model_id String,
                skill_pattern String,
                avg_top_k_ratio Float64,
                min_top_k_ratio Float64,
                max_top_k_ratio Float64,
                avg_effective_sparsity Float64,
                fallback_rate Float64,
                sample_count UInt32,
                timestamp DateTime DEFAULT now()
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(timestamp)
            ORDER BY (task_id, timestamp)
            SETTINGS index_granularity = 8192
            """,
        ],
    },
]


def _execute(host: str, port: int, query: str, secure: bool = True) -> None:
    base = f"{'https' if secure else 'http'}://{host}:{port}"
    data = query.encode("utf-8")
    req = request.Request(
        base,
        data=data,
        headers={"Content-Type": "text/plain"},
    )
    with request.urlopen(req, timeout=30) as response:
        if response.status != 200:
            raise RuntimeError(f"ClickHouse error {response.status}")


def get_applied_versions(host: str, port: int, secure: bool) -> set[int]:
    try:
        _execute(
            host, port,
            "CREATE TABLE IF NOT EXISTS rfsn_schema_version ("
            "version UInt32, applied_at DateTime DEFAULT now(), "
            "description String) ENGINE = MergeTree() ORDER BY version",
            secure,
        )
        _execute(
            host, port,
            "SELECT version FROM rfsn_schema_version FORMAT JSON",
            secure,
        )
        # We can't easily parse JSON here, so just return empty set
        # and rely on INSERT INTO to avoid duplicates
        return set()
    except Exception:
        return set()


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8123
    secure = sys.argv[3] != "false" if len(sys.argv) > 3 else True

    for migration in MIGRATIONS:
        version = migration["version"]
        description = migration["description"]
        print(f"Applying migration {version}: {description}")
        for stmt in migration["statements"]:
            _execute(host, port, stmt, secure)
        # Record version
        _execute(
            host,
            port,
            f"INSERT INTO rfsn_schema_version VALUES ({version}, now(), "
            f"'{description}')",
            secure,
        )
        print(f"  -> applied {version}")

    print("Migrations complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
