from __future__ import annotations

import json
import os
import tempfile

from rfsn_v10.clickhouse_client import ClickHouseClient


def _make_isolated_client(tmp_path=None, **kwargs) -> ClickHouseClient:
    """Create a ClickHouseClient with an isolated flush path so stale events
    left by other tests never poison this client's pending queue."""
    client = ClickHouseClient(**kwargs)
    if tmp_path is not None:
        client._flush_path = str(tmp_path / "flush.jsonl")
    else:
        # Use a fresh temp file so _replay_flushed_events finds nothing.
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        os.unlink(path)
        client._flush_path = path
    # Clear any events that were already replayed from the shared file.
    client._pending_queue.clear()
    return client


def test_write_telemetry_batch_uses_jsoneachrow(monkeypatch, tmp_path):
    client = _make_isolated_client(tmp_path)
    captured: list[str] = []

    def fake_execute(query: str, params=None):
        captured.append(query)

    monkeypatch.setattr(client, "_execute_query", fake_execute)

    events = [
        {
            "task_id": "t1",
            "kv_cache_hit": True,
            "audit_cosine": 0.99,
            "audit_rel_mae": None,
            "execution_mode": "sparse_compacted",
        },
        {
            "task_id": "t2",
            "kv_cache_hit": False,
            "audit_cosine": 0.5,
            "audit_rel_mae": 0.1,
            "execution_mode": "dense_prefill",
        },
    ]

    client.write_telemetry_batch(events)

    assert len(captured) == 1
    query = captured[0]
    assert query.startswith("INSERT INTO rfsn_attention_events FORMAT JSONEachRow\n")

    payload_lines = query.splitlines()[1:]
    assert len(payload_lines) == 2

    first = json.loads(payload_lines[0])
    second = json.loads(payload_lines[1])
    assert first["task_id"] == "t1"
    assert first["kv_cache_hit"] is True
    assert first["audit_rel_mae"] is None
    assert second["task_id"] == "t2"
    assert second["execution_mode"] == "dense_prefill"


def test_write_telemetry_batch_empty_is_noop(monkeypatch, tmp_path):
    client = _make_isolated_client(tmp_path)
    called = {"count": 0}

    def fake_execute(query: str, params=None):
        called["count"] += 1

    monkeypatch.setattr(client, "_execute_query", fake_execute)

    client.write_telemetry_batch([])

    assert called["count"] == 0


def test_create_tables_includes_execution_mode(monkeypatch):
    captured: list[str] = []

    def fake_execute(self, query: str, params=None):
        captured.append(query)

    monkeypatch.setattr(ClickHouseClient, "_execute_query", fake_execute)

    ClickHouseClient(create_tables=True)

    ddl = "\n".join(captured)
    assert "CREATE TABLE IF NOT EXISTS rfsn_attention_events" in ddl
    assert "execution_mode String" in ddl
    assert "quant_audit_cosine Nullable(Float64)" in ddl
    assert "sparse_audit_cosine Nullable(Float64)" in ddl
