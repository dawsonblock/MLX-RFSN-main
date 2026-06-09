"""Server request limit enforcement tests."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from rfsn_v10.config import RFSNConfig
from rfsn_v10.server.app import ServerState


@pytest.mark.server
@pytest.mark.unit
def test_max_prompt_chars_from_config():
    """ServerState respects max_prompt_chars from config."""
    cfg = RFSNConfig.from_env()
    cfg.server.max_prompt_chars = 100
    state = ServerState(cfg=cfg)
    assert state.cfg.server.max_prompt_chars == 100


@pytest.mark.server
@pytest.mark.unit
def test_max_tokens_limit_positive():
    """max_tokens_limit defaults to a positive integer."""
    cfg = RFSNConfig.from_env()
    assert cfg.server.max_tokens_limit >= 1


@pytest.mark.server
@pytest.mark.unit
def test_require_api_key_false_by_default():
    """API key enforcement is off by default."""
    cfg = RFSNConfig.from_env()
    assert cfg.server.require_api_key is False
