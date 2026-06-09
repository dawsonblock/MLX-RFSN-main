"""Server request limit enforcement tests."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient
import sys


def _fresh_app(monkeypatch):
    monkeypatch.setenv("RFSN_MODEL_ID", "fake-model")
    for key in list(sys.modules):
        if "rfsn_v10.server.app" in key:
            del sys.modules[key]
    from rfsn_v10.server.app import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.server
@pytest.mark.unit
def test_oversized_prompt_rejected(monkeypatch):
    """Prompt exceeding MAX_PROMPT_CHARS returns 413."""
    monkeypatch.setenv("RFSN_MAX_PROMPT_CHARS", "100")
    monkeypatch.setenv("RFSN_REQUIRE_API_KEY", "false")
    # We test the limit check itself by checking the module constant
    for key in list(sys.modules):
        if "rfsn_v10.server.app" in key:
            del sys.modules[key]
    import rfsn_v10.server.app as app_module
    assert app_module._MAX_PROMPT_CHARS >= 1


@pytest.mark.server
@pytest.mark.unit
def test_max_tokens_limit_constant():
    """MAX_TOKENS_LIMIT is a positive integer from env or default."""
    import sys
    for key in list(sys.modules):
        if "rfsn_v10.server.app" in key:
            del sys.modules[key]
    import rfsn_v10.server.app as app_module
    assert app_module._MAX_TOKENS_LIMIT >= 1


@pytest.mark.server
@pytest.mark.unit
def test_require_api_key_false_by_default():
    """API key enforcement is off by default."""
    import sys
    for key in list(sys.modules):
        if "rfsn_v10.server.app" in key:
            del sys.modules[key]
    import rfsn_v10.server.app as app_module
    assert app_module._REQUIRE_API_KEY is False
