from __future__ import annotations

import json

import pytest

from rfsn_v10.clickhouse_client import ClickHouseClient


def test_attention_writer_targets_attention_table(monkeypatch):
    client = ClickHouseClient()
    captured: list[str] = []

    def fake_execute(query: str, params=None):
        captured.append(query)

    monkeypatch.setattr(client, "_execute_query", fake_execute)
    client.write_attention_events([{"task_id": "a1", "execution_mode": "dense"}])

    assert len(captured) == 1
    assert captured[0].startswith("INSERT INTO rfsn_attention_events FORMAT JSONEachRow\n")


def test_kv_writer_targets_kv_table(monkeypatch):
    client = ClickHouseClient()
    captured: list[str] = []

    def fake_execute(query: str, params=None):
        captured.append(query)

    monkeypatch.setattr(client, "_execute_query", fake_execute)
    client.write_kv_cache_events([{"task_id": "k1", "operation": "store"}])

    assert len(captured) == 1
    assert captured[0].startswith("INSERT INTO rfsn_kv_cache_events FORMAT JSONEachRow\n")


def test_invalid_table_rejected():
    client = ClickHouseClient()
    with pytest.raises(ValueError, match="not allowed"):
        client._write_events("not_allowed_table", [{"task_id": "x"}])


def test_jsoneachrow_payload_is_preserved(monkeypatch):
    client = ClickHouseClient()
    captured: list[str] = []

    def fake_execute(query: str, params=None):
        captured.append(query)

    monkeypatch.setattr(client, "_execute_query", fake_execute)

    events = [
        {"task_id": "x1", "audit_rel_mae": None, "ok": True},
        {"task_id": "x2", "audit_rel_mae": 0.1, "ok": False},
    ]
    client.write_audit_events(events)

    query = captured[0]
    lines = query.splitlines()[1:]
    assert len(lines) == 2

    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0["audit_rel_mae"] is None
    assert row1["audit_rel_mae"] == 0.1
    assert row0["ok"] is True
    assert row1["ok"] is False
