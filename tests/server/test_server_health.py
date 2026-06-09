"""Server health and models endpoint tests."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from rfsn_v10.config import RFSNConfig
from rfsn_v10.server.app import create_app


def _make_client(**overrides) -> TestClient:
    """Create an isolated test client with custom config."""
    cfg = RFSNConfig.from_env()
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    return TestClient(create_app(cfg), raise_server_exceptions=False)


@pytest.mark.server
@pytest.mark.unit
def test_health_returns_ok():
    """GET /health returns status=ok without a model loaded."""
    client = _make_client(**{"model.id": ""})
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
def test_models_returns_empty_list():
    """GET /v1/models returns empty list when no model is loaded."""
    client = _make_client(
        **{"model.id": "", "server.require_api_key": False},
    )
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert isinstance(data["data"], list)
