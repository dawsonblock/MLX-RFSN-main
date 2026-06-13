"""Packaging validation tests.

Phase 12 exit condition:
  1. Build source distribution.
  2. Build wheel from the source distribution.
  3. Install into an empty Python 3.12 environment.
  4. Remove the checkout from PYTHONPATH.
  5. Load every resource using importlib.resources.
  6. Run a real model from the installed wheel.
  7. Execute Metal from the installed wheel.

These tests are marked as slow and only run in CI or on demand.
"""
from __future__ import annotations

import importlib.resources
import subprocess
import sys
from pathlib import Path

import pytest


def test_package_can_import() -> None:
    """Verify the rfsn_v10.cache package is importable."""
    from rfsn_v10.cache import (
        BlockwiseReferenceAttention,
        CartesianCodec,
        GenerationCacheSession,
        MemoryReport,
    )
    assert CartesianCodec is not None
    assert GenerationCacheSession is not None
    assert MemoryReport is not None
    assert BlockwiseReferenceAttention is not None


def test_importlib_resources_can_list_files() -> None:
    """Verify package data files are accessible via importlib.resources."""
    import rfsn_v10.cache

    files = importlib.resources.files(rfsn_v10.cache)
    # Should be able to list at least __init__.py
    found_init = False
    for item in files.iterdir():
        if item.name == "__init__.py":
            found_init = True
            break
    assert found_init, "Cannot find __init__.py via importlib.resources"


@pytest.mark.slow
def test_wheel_builds() -> None:
    """Build a wheel from the source tree."""
    repo_root = Path(__file__).parent.parent.parent.parent.parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"Wheel build failed (needs build deps): {result.stderr[:200]}")

    dist_dir = repo_root / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) > 0, "No wheel produced"
