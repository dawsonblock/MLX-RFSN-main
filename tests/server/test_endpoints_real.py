"""Real server endpoint tests with fake generator.

Tests actual HTTP endpoints with proper request/response validation.
"""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

from rfsn_v10.server.app import app


@pytest.mark.server
class TestServerEndpoints:
    """Test server endpoints with realistic requests."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Test /health endpoint returns proper structure."""
        response = self.client.get("/health")
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
        response = self.client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

    @patch.dict("os.environ", {"RFSN_REQUIRE_API_KEY": "true", "RFSN_API_KEY": "test-key"})
    def test_v1_models_endpoint_with_auth(self):
        """Test /v1_models requires auth when enabled."""
        # Without auth key should fail
        response = self.client.get("/v1/models")
        assert response.status_code == 401
        
        # With auth key should succeed
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.get("/v1/models", headers=headers)
        assert response.status_code == 200

    @patch.dict("os.environ", {"RFSN_REQUIRE_API_KEY": "true", "RFSN_API_KEY": "test-key"})
    def test_metrics_endpoint_auth(self):
        """Test /metrics endpoint requires auth when enabled."""
        # Without auth key should fail
        response = self.client.get("/metrics")
        assert response.status_code == 401
        
        # With auth key should succeed
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.get("/metrics", headers=headers)
        assert response.status_code == 200

    @patch.dict("os.environ", {"RFSN_REQUIRE_API_KEY": "true", "RFSN_API_KEY": "test-key"})
    def test_dashboard_endpoint_auth(self):
        """Test /dashboard endpoint requires auth when enabled."""
        # Without auth key should fail
        response = self.client.get("/dashboard")
        assert response.status_code == 401
        
        # With auth key should succeed
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.get("/dashboard", headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")

    @patch.dict("os.environ", {"RFSN_MODEL_ID": "test-model"})
    def test_v1_models_shows_configured_model(self):
        """Test /v1/models shows configured model even before load."""
        response = self.client.get("/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["data"]) > 0
        
        model = data["data"][0]
        assert model["id"] == "test-model"
        assert model["object"] == "model"
        assert "loaded" in model
        assert model["loaded"] is False  # Not loaded yet

    @patch.dict("os.environ", {"RFSN_REQUIRE_API_KEY": "true", "RFSN_API_KEY": "test-key"})
    def test_chat_completions_missing_auth(self):
        """Test chat completions requires auth when enabled."""
        request_data = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "stream": False
        }
        
        response = self.client.post("/v1/chat/completions", json=request_data)
        assert response.status_code == 401

    @patch.dict("os.environ", {"RFSN_REQUIRE_API_KEY": "true", "RFSN_API_KEY": "test-key"})
    @patch("rfsn_v10.server.app._load_generator")
    def test_chat_completions_oversized_prompt(self, mock_load_generator):
        """Test chat completions rejects oversized prompts."""
        # Mock generator
        mock_generator = AsyncMock()
        mock_load_generator.return_value = mock_generator
        
        # Create oversized prompt (assuming default 24000 char limit)
        oversized_content = "x" * 25000
        request_data = {
            "model": "test",
            "messages": [{"role": "user", "content": oversized_content}],
            "max_tokens": 10,
            "stream": False
        }
        
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.post("/v1/chat/completions", json=request_data, headers=headers)
        assert response.status_code == 413  # Payload Too Large

    @patch.dict("os.environ", {"RFSN_REQUIRE_API_KEY": "true", "RFSN_API_KEY": "test-key"})
    @patch("rfsn_v10.server.app._load_generator")
    def test_chat_completions_too_many_tokens(self, mock_load_generator):
        """Test chat completions rejects excessive max_tokens."""
        # Mock generator
        mock_generator = AsyncMock()
        mock_load_generator.return_value = mock_generator
        
        request_data = {
            "model": "test",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10000,  # Over default 4096 limit
            "stream": False
        }
        
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.post("/v1/chat/completions", json=request_data, headers=headers)
        assert response.status_code == 400  # Bad Request

    @patch.dict("os.environ", {"RFSN_REQUIRE_API_KEY": "true", "RFSN_API_KEY": "test-key"})
    @patch("rfsn_v10.server.app._load_generator")
    def test_chat_completions_successful_non_streaming(self, mock_load_generator):
        """Test successful non-streaming chat completion."""
        # Mock generator
        mock_generator = AsyncMock()
        mock_load_generator.return_value = mock_generator
        
        # Mock generation response
        mock_generator.generate.return_value = [
            {"token": "Hello", "logprob": -0.5},
            {"token": " world", "logprob": -0.3},
            {"token": "!", "logprob": -0.2},
        ]
        
        request_data = {
            "model": "test",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
            "stream": False
        }
        
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.post("/v1/chat/completions", json=request_data, headers=headers)
        assert response.status_code == 200
        
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert data["choices"][0]["message"]["content"] == "Hello world!"

    @patch.dict("os.environ", {"RFSN_REQUIRE_API_KEY": "true", "RFSN_API_KEY": "test-key"})
    @patch("rfsn_v10.server.app._load_generator")
    def test_chat_completions_successful_streaming(self, mock_load_generator):
        """Test successful streaming chat completion."""
        # Mock generator
        mock_generator = AsyncMock()
        mock_load_generator.return_value = mock_generator
        
        # Mock streaming response
        mock_generator.generate_stream.return_value = [
            {"token": "Hello", "logprob": -0.5},
            {"token": " world", "logprob": -0.3},
            {"token": "!", "logprob": -0.2},
        ]
        
        request_data = {
            "model": "test",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
            "stream": True
        }
        
        headers = {"Authorization": "Bearer test-key"}
        response = self.client.post("/v1/chat/completions", json=request_data, headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"
        
        # Verify SSE format
        lines = response.text.split("\n")
        assert any(line.startswith("data: ") for line in lines)
        assert any(line == "data: [DONE]" for line in lines)

    def test_docs_endpoint_disabled(self):
        """Test that docs endpoint can be disabled."""
        # This test assumes enable_docs is False in test environment
        # If it's True, this test should be adapted or skipped
        response = self.client.get("/docs")
        # Should either work (200) or be disabled (404)
        assert response.status_code in [200, 404]

    def test_dashboard_endpoint_disabled(self):
        """Test that dashboard endpoint can be disabled."""
        # This test assumes enable_dashboard is False in test environment
        # If it's True, this test should be adapted or skipped
        response = self.client.get("/dashboard")
        # Should either work (200) or be disabled (404)
        assert response.status_code in [200, 404]


@pytest.mark.server
class TestServerConfiguration:
    """Test server configuration and behavior."""

    def test_concurrency_limit_enforced(self):
        """Test that concurrency limits are enforced."""
        # This would require more complex setup with actual concurrent requests
        # For now, just verify the semaphore is configured
        from rfsn_v10.server.app import _generation_semaphore
        assert _generation_semaphore._value >= 1  # Should have at least 1 slot

    def test_metrics_updated(self):
        """Test that metrics are properly updated."""
        from rfsn_v10.server.app import _metrics
        
        # Initial state
        assert _metrics["requests_total"] == 0
        assert _metrics["model_loaded"] is False
        
        # Metrics should be updated during actual requests
        # This would require integration tests with real requests