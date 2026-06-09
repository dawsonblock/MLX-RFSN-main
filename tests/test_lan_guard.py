"""Tests for LAN guard and concurrency config validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


@pytest.mark.unit
def test_localhost_no_api_key_allowed():
    from rfsn_v10.config import ServerConfig
    cfg = ServerConfig(host="127.0.0.1", require_api_key=False)
    assert cfg.host == "127.0.0.1"


@pytest.mark.unit
def test_lan_host_without_api_key_raises():
    from rfsn_v10.config import ServerConfig
    with pytest.raises(ValidationError) as exc_info:
        ServerConfig(host="0.0.0.0", require_api_key=False)
    assert "LAN mode" in str(exc_info.value)


@pytest.mark.unit
def test_lan_host_with_api_key_allowed():
    from rfsn_v10.config import ServerConfig
    cfg = ServerConfig(
        host="0.0.0.0",
        require_api_key=True,
        api_key="secret-token",
    )
    assert cfg.host == "0.0.0.0"
    assert cfg.require_api_key is True


@pytest.mark.unit
def test_require_api_key_without_key_raises():
    from rfsn_v10.config import ServerConfig
    with pytest.raises(ValidationError) as exc_info:
        ServerConfig(require_api_key=True, api_key="")
    assert "RFSN_API_KEY" in str(exc_info.value)


@pytest.mark.unit
def test_default_max_concurrent_is_one():
    from rfsn_v10.config import ServerConfig
    cfg = ServerConfig()
    assert cfg.max_concurrent_requests == 1


@pytest.mark.unit
def test_max_concurrent_configurable():
    from rfsn_v10.config import ServerConfig
    cfg = ServerConfig(max_concurrent_requests=4)
    assert cfg.max_concurrent_requests == 4


@pytest.mark.unit
def test_max_concurrent_minimum_one():
    from rfsn_v10.config import ServerConfig
    with pytest.raises(ValidationError):
        ServerConfig(max_concurrent_requests=0)
