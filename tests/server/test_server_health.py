"""Server health and models endpoint tests."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient


@pytest.mark.server
@pytest.mark.unit
def test_health_returns_ok(monkeypatch):
    """GET /health returns status=ok without a model loaded."""
    monkeypatch.setenv("RFSN_MODEL_ID", "")
    # Import after env is set
    import importlib
    import sys
    for key in list(sys.modules):
        if "rfsn_v10.server.app" in key:
            del sys.modules[key]
    from rfsn_v10.server.app import app
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "backend" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is False
    assert "kv_compression" in data
    assert "sparse_decode" in data
    assert "telemetry" in data


@pytest.mark.server
@pytest.mark.unit
def test_models_returns_empty_list(monkeypatch):
    """GET /v1/models returns empty list when no model is loaded."""
    monkeypatch.setenv("RFSN_MODEL_ID", "")
    monkeypatch.setenv("RFSN_REQUIRE_API_KEY", "false")
    import sys
    for key in list(sys.modules):
        if "rfsn_v10.server.app" in key:
            del sys.modules[key]
    from rfsn_v10.server.app import app
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
