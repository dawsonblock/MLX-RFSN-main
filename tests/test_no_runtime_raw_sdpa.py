"""Enforcement test: runtime and audit paths must not call raw SDPA directly.

Any dense attention in runtime/audit/scoring paths must go through
:func:`rfsn_v10.attention_reference.causal_attention_dense` so that causal
masking is always applied for multi-token prefill.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN_FILES = [
    "rfsn_v10/runtime/engine.py",
    "rfsn_v10/runtime/scoring_modes.py",
    "rfsn_v10/runtime/audit.py",
    "rfsn_v10/attention.py",
]

_FORBIDDEN_PATTERN = "mx.fast.scaled_dot_product_attention"


def _code_lines_without_comments(source: str) -> list[str]:
    """Return non-empty, non-comment, non-docstring code lines."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fall back to raw line check if parsing fails
        return [
            line
            for line in source.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    # Collect line numbers that are part of string constants (docstrings)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                # Mark every line in this string expression
                for lineno in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                    docstring_lines.add(lineno)

    lines = source.splitlines()
    result = []
    for i, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if i in docstring_lines:
            continue
        result.append(line)
    return result


def test_runtime_does_not_bypass_causal_reference():
    """Verify that forbidden files do not call mx.fast.scaled_dot_product_attention."""
    for rel_path in FORBIDDEN_FILES:
        file_path = REPO_ROOT / rel_path
        assert file_path.exists(), f"Expected file not found: {file_path}"
        source = file_path.read_text(encoding="utf-8")
        code_lines = _code_lines_without_comments(source)
        for line in code_lines:
            assert _FORBIDDEN_PATTERN not in line, (
                f"Forbidden raw SDPA call found in {rel_path}:\n  {line.strip()}\n"
                "Use causal_attention_dense from rfsn_v10.attention_reference instead."
            )
