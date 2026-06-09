"""Shared pytest fixtures and hooks for the rfsn_v10 test suite."""
from __future__ import annotations

import os

import pytest

from rfsn_v10.clickhouse_client import ClickHouseClient


@pytest.fixture(autouse=True)
def isolate_clickhouse_flush_path(tmp_path, monkeypatch):
    """Redirect the ClickHouseClient shared flush file to a per-test temp path.

    Without this, retry tests that write to /tmp/rfsn_telemetry_flush.jsonl
    leave stale events on disk that get replayed by the next ClickHouseClient
    constructor, causing spurious extra _execute_query calls in other tests.
    """
    isolated_path = str(tmp_path / "rfsn_telemetry_flush.jsonl")
    monkeypatch.setattr(ClickHouseClient, "_FLUSH_PATH", isolated_path)
    # Also remove any leftover shared file from previous runs outside pytest.
    shared = "/tmp/rfsn_telemetry_flush.jsonl"
    if os.path.exists(shared):
        os.unlink(shared)
    yield
