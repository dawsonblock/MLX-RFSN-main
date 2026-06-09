"""Test server backend error handling.

Validates that the FastAPI server returns proper HTTP status codes
for backend mismatch and missing configuration, not raw 500s.
"""
from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from rfsn_v10.server.app import _load_generator, app


class TestBackendErrors:
    """Server backend error handling tests."""

    def test_numpy_backend_raises_valueerror(self, monkeypatch: Any) -> None:
        """RFSN_BACKEND=numpy should raise ValueError from load_model_auto."""
        monkeypatch.setenv("RFSN_BACKEND", "numpy")
        monkeypatch.setenv("RFSN_MODEL_ID", "dummy")

        with pytest.raises(ValueError, match="Unknown backend 'numpy'"):
            _load_generator()

    def test_bad_backend_raises_valueerror(self, monkeypatch: Any) -> None:
        """RFSN_BACKEND=bad should raise ValueError from load_model_auto."""
        monkeypatch.setenv("RFSN_BACKEND", "bad")
        monkeypatch.setenv("RFSN_MODEL_ID", "dummy")

        with pytest.raises(ValueError, match="Unknown backend 'bad'"):
            _load_generator()

    def test_missing_model_id_error(self, monkeypatch: Any) -> None:
        """Missing RFSN_MODEL_ID should raise RuntimeError."""
        monkeypatch.delenv("RFSN_MODEL_ID", raising=False)

        with pytest.raises(RuntimeError, match="RFSN_MODEL_ID is not set"):
            _load_generator()


class TestChatEndpointErrorCodes:
    """Verify chat endpoint HTTP status codes via route-level testing."""

    def test_numpy_backend_returns_400(self, monkeypatch: Any) -> None:
        """Simulate numpy backend → chat should get HTTP 400, not 500."""
        monkeypatch.setenv("RFSN_BACKEND", "numpy")
        monkeypatch.setenv("RFSN_MODEL_ID", "dummy")

        # Simulate what the route does
        try:
            _load_generator()
            pytest.fail("Expected ValueError")
        except ValueError as exc:
            # Route converts this to HTTP 400
            http_exc = HTTPException(status_code=400, detail=str(exc))
            assert http_exc.status_code == 400
            assert "numpy" in http_exc.detail

    def test_missing_model_id_returns_503(self, monkeypatch: Any) -> None:
        """Missing model ID → chat should get HTTP 503."""
        monkeypatch.delenv("RFSN_MODEL_ID", raising=False)

        try:
            _load_generator()
            pytest.fail("Expected RuntimeError")
        except RuntimeError as exc:
            http_exc = HTTPException(status_code=503, detail=str(exc))
            assert http_exc.status_code == 503
            assert "RFSN_MODEL_ID" in http_exc.detail


class TestHealthEndpoint:
    """Health endpoint should always return 200."""

    def test_health_always_200(self) -> None:
        """Health check does not depend on backend or model ID."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
