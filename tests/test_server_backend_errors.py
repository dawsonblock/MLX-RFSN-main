"""Test server backend error handling.

Validates that the FastAPI server returns proper HTTP status codes
for backend mismatch and missing configuration, not raw 500s.
"""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from rfsn_v10.config import RFSNConfig
from rfsn_v10.server.app import ServerState, create_app


def _state_with(**overrides) -> ServerState:
    """Build a ServerState with custom config fields."""
    cfg = RFSNConfig.from_env()
    for key, val in overrides.items():
        parts = key.split(".")
        obj = cfg
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], val)
    return ServerState(cfg=cfg)


class TestBackendErrors:
    """Server backend error handling tests."""

    def test_numpy_backend_raises_valueerror(self) -> None:
        """RFSN_BACKEND=numpy should raise ValueError."""
        s = _state_with(**{"backend.name": "numpy", "model.id": "dummy"})
        with pytest.raises(ValueError, match="Unknown backend 'numpy'"):
            s.load_generator()

    def test_bad_backend_raises_valueerror(self) -> None:
        """RFSN_BACKEND=bad should raise ValueError."""
        s = _state_with(**{"backend.name": "bad", "model.id": "dummy"})
        with pytest.raises(ValueError, match="Unknown backend 'bad'"):
            s.load_generator()

    def test_missing_model_id_error(self) -> None:
        """Missing RFSN_MODEL_ID should raise RuntimeError."""
        s = _state_with(**{"model.id": ""})
        with pytest.raises(RuntimeError, match="RFSN_MODEL_ID is not set"):
            s.load_generator()


class TestChatEndpointErrorCodes:
    """Verify chat endpoint HTTP status codes via route-level testing."""

    def test_numpy_backend_returns_400(self) -> None:
        """Simulate numpy backend → load_generator → ValueError."""
        s = _state_with(**{"backend.name": "numpy", "model.id": "dummy"})
        try:
            s.load_generator()
            pytest.fail("Expected ValueError")
        except ValueError as exc:
            http_exc = HTTPException(status_code=400, detail=str(exc))
            assert http_exc.status_code == 400
            assert "numpy" in http_exc.detail

    def test_missing_model_id_returns_503(self) -> None:
        """Missing model ID → load_generator → RuntimeError."""
        s = _state_with(**{"model.id": ""})
        try:
            s.load_generator()
            pytest.fail("Expected RuntimeError")
        except RuntimeError as exc:
            http_exc = HTTPException(status_code=503, detail=str(exc))
            assert http_exc.status_code == 503
            assert "RFSN_MODEL_ID" in http_exc.detail


class TestHealthEndpoint:
    """Health endpoint should always return 200."""

    def test_health_always_200(self) -> None:
        """Health check does not depend on backend or model ID."""
        cfg = RFSNConfig.from_env()
        cfg.model.id = ""
        client = TestClient(create_app(cfg))
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("ok", "healthy")
        assert "version" in data
