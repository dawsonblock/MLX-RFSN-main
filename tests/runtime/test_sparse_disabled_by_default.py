"""Verify sparse decode is disabled by default in RFSNConfig."""
from __future__ import annotations

import os

import pytest


@pytest.mark.unit
def test_sparse_decode_off_by_default():
    """RuntimeConfig.sparse_decode_enabled must default to False."""
    import sys
    for key in list(sys.modules):
        if key.startswith("rfsn_v10.config"):
            del sys.modules[key]
    os.environ.pop("RFSN_SPARSE_DECODE_ENABLED", None)
    from rfsn_v10.config import RFSNConfig
    cfg = RFSNConfig()
    assert cfg.runtime.sparse_decode_enabled is False


@pytest.mark.unit
def test_qjl_off_by_default():
    """RuntimeConfig.qjl_enabled must default to False."""
    import sys
    for key in list(sys.modules):
        if key.startswith("rfsn_v10.config"):
            del sys.modules[key]
    os.environ.pop("RFSN_QJL_ENABLED", None)
    from rfsn_v10.config import RFSNConfig
    cfg = RFSNConfig()
    assert cfg.runtime.qjl_enabled is False


@pytest.mark.unit
def test_experimental_all_off_by_default():
    """All experimental flags must default to False."""
    import sys
    for key in list(sys.modules):
        if key.startswith("rfsn_v10.config"):
            del sys.modules[key]
    for env in ["RFSN_EXPERIMENTAL_QJL", "RFSN_EXPERIMENTAL_POLAR", "RFSN_EXPERIMENTAL_ADAPTIVE"]:
        os.environ.pop(env, None)
    from rfsn_v10.config import RFSNConfig
    cfg = RFSNConfig()
    assert cfg.experimental.enable_qjl is False
    assert cfg.experimental.enable_polar is False
    assert cfg.experimental.enable_adaptive is False
