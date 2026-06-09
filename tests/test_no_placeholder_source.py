"""Guard test: no Python source file may contain literal placeholder text.

This test catches generated-code debris — files that contain prose like
"FULL COMPLETE CODE WITH FUNCTIONAL INTERFACE HERE" instead of real Python.
Such files compile as SyntaxErrors and must never reach the package.

If this test fails, locate the offending file and either implement the
feature or replace the file with a valid disabled stub.
"""
from __future__ import annotations

import ast
from pathlib import Path

# Patterns that indicate a file is a placeholder and not real Python.
# These are split to avoid the test matching itself.
_P = "FULL"
FORBIDDEN_PATTERNS = [
    _P + " COMPLETE CODE",
    _P + " CODE FOR",
    "FUNCTIONAL INTERFACE HERE",
    "PLACEHOLDER ONLY",
    "PLACEHOLDER CODE",
    "INSERT CODE HERE",
    "TODO GENERATED",
    "TODO: IMPLEMENT",
]

# Directories to skip (build artifacts, dependencies, etc.)
SKIP_DIRS = {
    ".venv", "venv", "dist", "build", ".eggs", "__pycache__", "node_modules",
}


def test_no_generated_placeholder_text_in_python_sources():
    """Every .py file must be valid Python, not placeholder text."""
    root = Path(__file__).resolve().parents[1]
    this_file = Path(__file__).resolve()
    offenders = []

    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)

        # Skip excluded directories (anchored to repo root)
        # and this file itself
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path == this_file:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            offenders.append(f"{rel}: unreadable ({exc})")
            continue

        # Validate that the file is syntactically valid Python
        try:
            ast.parse(text)
        except SyntaxError as exc:
            offenders.append(f"{rel}:{exc.lineno}: SyntaxError: {exc.msg}")
            continue

        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                # Find the first matching line for context
                for lineno, line in enumerate(text.splitlines(), 1):
                    if pattern in line:
                        offenders.append(
                            f"{rel}:{lineno}: contains {pattern!r}"
                        )
                        break

    assert not offenders, (
        f"Found {len(offenders)} source file(s) with placeholder text "
        f"(replace with valid Python or a disabled stub):\n"
        + "\n".join(f"  {o}" for o in offenders)
    )
