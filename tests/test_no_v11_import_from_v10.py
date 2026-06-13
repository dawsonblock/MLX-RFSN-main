"""Boundary test: rfsn_v10 stable runtime must not import from rfsn_v11 or external/.

This enforces the separation between stable and experimental code.
If this test fails, a stable module has grown an import dependency on
experimental code — that must be fixed before releasing.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
V10_DIR = REPO_ROOT / "rfsn_v10"

FORBIDDEN_PREFIXES = (
    "rfsn_v11",
    "external",
    "research",
    "agent_core",
)


def _collect_imports(path: Path) -> list[str]:
    """Return all top-level module names imported by a Python file."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


@pytest.mark.unit
def test_rfsn_v10_has_no_forbidden_imports():
    """No file in rfsn_v10/ should import from rfsn_v11, external, research, or agent_core."""
    violations: list[str] = []
    for py_file in sorted(V10_DIR.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        for imp in _collect_imports(py_file):
            for prefix in FORBIDDEN_PREFIXES:
                if imp == prefix or imp.startswith(prefix + "."):
                    rel = py_file.relative_to(REPO_ROOT)
                    violations.append(f"{rel}: imports '{imp}'")
    assert not violations, (
        "rfsn_v10 stable runtime has forbidden imports:\n"
        + "\n".join(f"  {v}" for v in violations)
    )
