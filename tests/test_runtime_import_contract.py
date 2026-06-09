"""Runtime import contract regression tests.

Verifies that the module structure fixed in the beta-repair remains stable:
1. RFSNRuntime is importable from both rfsn_v10 and rfsn_v10.runtime
2. The old rfsn_v10/runtime.py file does not exist (collision was removed)
3. The runtime package lives at rfsn_v10/runtime/__init__.py
4. Required subpackages are importable

These tests exist to catch regressions if someone accidentally re-introduces
the runtime.py vs runtime/ collision or breaks subpackage installation.
"""
from __future__ import annotations

from pathlib import Path


def test_public_rfsn_v10_import():
    """Top-level package must import without error."""
    import rfsn_v10  # noqa: F401
    assert rfsn_v10 is not None


def test_runtime_import_from_top_level():
    """RFSNRuntime must be importable from rfsn_v10 directly."""
    from rfsn_v10 import RFSNRuntime
    assert RFSNRuntime is not None


def test_runtime_import_from_package():
    """RFSNRuntime must be importable from rfsn_v10.runtime subpackage."""
    from rfsn_v10.runtime import RFSNRuntime
    assert RFSNRuntime is not None


def test_runtime_imports_are_same_class():
    """Both import paths must resolve to the same class."""
    from rfsn_v10 import RFSNRuntime as A
    from rfsn_v10.runtime import RFSNRuntime as B
    assert A is B, (
        "rfsn_v10.RFSNRuntime and rfsn_v10.runtime.RFSNRuntime are different objects. "
        "Ensure runtime/__init__.py re-exports from engine.py and __init__.py imports from runtime."
    )


def test_kernels_subpackage_importable():
    """rfsn_v10.kernels must be importable (subpackage discovery must be correct)."""
    import rfsn_v10.kernels  # noqa: F401


def test_quantization_subpackage_importable():
    """rfsn_v10.quantization must be importable."""
    import rfsn_v10.quantization  # noqa: F401


def test_runtime_subpackage_importable():
    """rfsn_v10.runtime must be importable as a package (not a module)."""
    import rfsn_v10.runtime  # noqa: F401
    import importlib
    spec = importlib.util.find_spec("rfsn_v10.runtime")
    assert spec is not None, "rfsn_v10.runtime not found"
    # The spec origin should point to runtime/__init__.py, not runtime.py
    assert spec.origin is not None
    assert "runtime/__init__.py" in spec.origin.replace("\\", "/"), (
        f"Expected runtime/__init__.py but got: {spec.origin}\n"
        "This indicates the runtime.py vs runtime/ collision has returned."
    )


def test_no_runtime_py_collision():
    """rfsn_v10/runtime.py must not exist — it collides with the runtime/ package."""
    root = Path(__file__).resolve().parents[1]
    collision = root / "rfsn_v10" / "runtime.py"
    assert not collision.exists(), (
        f"{collision} exists — this causes a module/package collision. "
        "The runtime must live entirely under rfsn_v10/runtime/ (package), "
        "not as a top-level rfsn_v10/runtime.py module."
    )


def test_runtime_engine_exists():
    """rfsn_v10/runtime/engine.py must exist (canonical location after Phase 2 repair)."""
    root = Path(__file__).resolve().parents[1]
    engine = root / "rfsn_v10" / "runtime" / "engine.py"
    assert engine.exists(), (
        f"{engine} not found. The runtime engine should live at "
        "rfsn_v10/runtime/engine.py (moved from rfsn_v10/runtime.py in Phase 2)."
    )
