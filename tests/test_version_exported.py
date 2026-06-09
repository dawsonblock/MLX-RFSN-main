"""Test that rfsn_v10 exports __version__."""
from __future__ import annotations

import rfsn_v10


def test_version_is_string() -> None:
    assert isinstance(rfsn_v10.__version__, str)


def test_version_not_empty() -> None:
    assert rfsn_v10.__version__ != ""
    assert rfsn_v10.__version__ != "unknown"


def test_version_not_zero() -> None:
    assert rfsn_v10.__version__ != "0.0.0"
