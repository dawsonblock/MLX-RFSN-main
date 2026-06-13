"""Real server endpoint tests with fake generator.

Tests actual HTTP endpoints with proper request/response validation.
Uses create_app() factory for per-test isolation.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from rfsn_v10.config import RFSNConfig
from rfsn_v10.server.app import ServerState, create_app


def _app_and_client(**overrides):
    """Build an app + TestClient with custom config fields."""
    cfg = RFSNConfig.from_env()
    cfg.model.id = overrides.pop("model_id", "")
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    application = create_app(cfg)
    return application, TestClient(
        application, raise_server_exceptions=False,
    )


@pytest.mark.server
class TestServerEndpoints:
    """Test server endpoints with realistic requests."""

    def test_health_endpoint(self):
        """Test /health endpoint returns proper structure."""
        _, client = _app_and_client()
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "backend" in data
        assert "model_loaded" in data
        assert "api_key_required" in data
        assert "max_concurrent_requests" in data

    def test_v1_models_endpoint_no_auth_required(self):
        """Test /v1/models works without auth when not required."""
        _, client = _app_and_client(
            **{"server.require_api_key": False},
        )
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_v1_models_endpoint_with_auth(self):
        """Test /v1/models requires auth when enabled."""
        _, client = _app_and_client(
            **{
                "server.require_api_key": True,
                "server.api_key": "test-key",
            },
        )
        # Without auth key should fail
        response = client.get("/v1/models")
        assert response.status_code == 401
        # With auth key should succeed
        headers = {"Authorization": "Bearer test-key"}
        response = client.get("/v1/models", headers=headers)
        assert response.status_code == 200

    def test_metrics_endpoint_auth(self):
        """Test /metrics endpoint requires auth when enabled."""
        _, client = _app_and_client(
            **{
                "server.require_api_key": True,
                "server.api_key": "test-key",
            },
        )
        # Without auth key should fail
        response = client.get("/metrics")
        assert response.status_code == 401
        # With auth key should succeed
        headers = {"Authorization": "Bearer test-key"}
        response = client.get("/metrics", headers=headers)
        assert response.status_code == 200

    def test_dashboard_endpoint_auth(self):
        """Test /dashboard requires auth when enabled."""
        _, client = _app_and_client(
            **{
                "server.require_api_key": True,
                "server.api_key": "test-key",
                "server.enable_dashboard": True,
            },
        )
        # Without auth key should fail
        response = client.get("/dashboard")
        assert response.status_code == 401
        # With auth key should succeed
        headers = {"Authorization": "Bearer test-key"}
        response = client.get("/dashboard", headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/html",
        )

    def test_v1_models_shows_configured_model(self):
        """Test /v1/models shows configured model before load."""
        _, client = _app_and_client(
            model_id="test-model",
            **{"server.require_api_key": False},
        )
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert len(data["data"]) > 0
        model = data["data"][0]
        assert model["id"] == "test-model"
        assert model["object"] == "model"
        assert "loaded" in model
        assert model["loaded"] is False

    def test_chat_completions_missing_auth(self):
        """Test chat completions requires auth when enabled."""
        _, client = _app_and_client(
            **{
                "server.require_api_key": True,
                "server.api_key": "test-key",
            },
        )
        request_data = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False,
        }
        response = client.post(
            "/v1/chat/completions", json=request_data,
        )
        assert response.status_code == 401

    def test_docs_endpoint_enabled(self):
        """Test that docs endpoint works when enabled."""
        _, client = _app_and_client(
            **{"server.enable_docs": True},
        )
        response = client.get("/docs")
        assert response.status_code == 200

    def test_docs_endpoint_disabled(self):
        """Test that docs endpoint is 404 when disabled."""
        _, client = _app_and_client(
            **{"server.enable_docs": False},
        )
        response = client.get("/docs")
        assert response.status_code == 404

    def test_dashboard_endpoint_disabled(self):
        """Test that dashboard is 404 when disabled."""
        _, client = _app_and_client(
            **{"server.enable_dashboard": False},
        )
        response = client.get("/dashboard")
        assert response.status_code in [404, 405]


@pytest.mark.server
class TestServerConfiguration:
    """Test server configuration and behavior."""

    def test_concurrency_limit_enforced(self):
        """Verify the semaphore is configured from config."""
        cfg = RFSNConfig.from_env()
        cfg.server.max_concurrent_requests = 3
        state = ServerState(cfg=cfg)
        assert state.semaphore._value == 3  # noqa: SLF001

    def test_metrics_initial_state(self):
        """Test that metrics start at zero."""
        cfg = RFSNConfig.from_env()
        state = ServerState(cfg=cfg)
        assert state.metrics["requests_total"] == 0
        assert state.metrics["model_loaded"] is False
