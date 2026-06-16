"""Shared pytest fixtures and hooks for the rfsn_v10 test suite."""
from __future__ import annotations

import os
import tempfile

import pytest

from rfsn_v10.clickhouse_client import ClickHouseClient


def pytest_addoption(parser: pytest.Parser) -> None:
    """Phase 1: Add explicit portable / native-release test modes."""
    parser.addoption(
        "--rfsn-portable",
        action="store_true",
        default=False,
        help="Portable CI mode: verify schema, structure, and safe defaults only",
    )
    parser.addoption(
        "--rfsn-native-release",
        action="store_true",
        default=False,
        help="Native release mode: require real execution evidence and provenance",
    )


@pytest.fixture
def rfsn_portable(request: pytest.FixtureRequest) -> bool:
    """Return True if --rfsn-portable was passed."""
    return bool(request.config.getoption("--rfsn-portable"))


@pytest.fixture
def rfsn_native_release(request: pytest.FixtureRequest) -> bool:
    """Return True if --rfsn-native-release was passed."""
    return bool(request.config.getoption("--rfsn-native-release"))


@pytest.fixture(autouse=True)
def isolate_clickhouse_flush_path(tmp_path, monkeypatch):
    """Redirect the ClickHouseClient flush file to a per-test temp path.

    Without this, retry tests that write to the shared flush file leave stale
    events on disk that get replayed by the next ClickHouseClient constructor,
    causing spurious extra _execute_query calls in other tests.
    """
    isolated_path = str(tmp_path / "rfsn_telemetry_flush.jsonl")
    original_init = ClickHouseClient.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Override the instance flush path after init
        self._flush_path = isolated_path

    monkeypatch.setattr(ClickHouseClient, "__init__", patched_init)
    # Also remove any leftover shared files from previous runs outside pytest.
    for shared in [
        "/tmp/rfsn_telemetry_flush.jsonl",
        os.path.join(tempfile.gettempdir(), "rfsn_telemetry_flush.jsonl"),
    ]:
        if os.path.exists(shared):
            os.unlink(shared)
    yield
