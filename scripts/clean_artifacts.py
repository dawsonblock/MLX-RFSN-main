#!/usr/bin/env python3
"""Clean build artifacts and cache files from the repository.

Usage::
    python scripts/clean_artifacts.py

Removes common build artifacts, cache files, and temporary files
to prepare for clean release builds.
"""
from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    """Clean all build artifacts and cache files."""
    root = Path(__file__).resolve().parent.parent

    # Directories to remove completely
    cleanup_dirs = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".rfsn_cache",
        ".tmp",
        ".venv",
        "dist",
        "build",
        "*.egg-info",
    ]

    # File patterns to remove
    cleanup_patterns = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".DS_Store",
        "*.log",
        "*.swp",
        "*.swo",
        "*~",
        "=*",
    ]

    removed_count = 0
    removed_size = 0

    print(f"Cleaning artifacts in: {root}")
    print()

    # Remove directories
    for pattern in cleanup_dirs:
        for path in root.rglob(pattern):
            if path.is_dir() and path.name == pattern.replace("*", ""):
                try:
                    size = sum(
                        f.stat().st_size
                        for f in path.rglob("*") if f.is_file()
                    )
                    shutil.rmtree(path)
                    print(
                        f"Removed directory: {path.relative_to(root)} "
                        f"({size / 1024:.1f} KB)"
                    )
                    removed_count += 1
                    removed_size += size
                except OSError as e:
                    print(f"Failed to remove {path}: {e}")

    # Remove files matching patterns
    for pattern in cleanup_patterns:
        for path in root.rglob(pattern):
            if path.is_file():
                try:
                    size = path.stat().st_size
                    path.unlink()
                    print(
                        f"Removed file: {path.relative_to(root)} ({size} bytes)"
                    )
                    removed_count += 1
                    removed_size += size
                except OSError as e:
                    print(f"Failed to remove {path}: {e}")

    print()
    print(f"Cleanup complete: {removed_count} items removed ({removed_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()