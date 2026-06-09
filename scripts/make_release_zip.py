#!/usr/bin/env python3
"""Create a clean release ZIP for RFSN v10.

Usage::

    python scripts/make_release_zip.py [--out dist/] [--version 10.2.0]

The ZIP includes only the source tree tracked by git (respects .gitignore),
and excludes a fixed set of development / runtime artifacts regardless of
git status.

Always verify the output with::

    unzip -l dist/rfsn-v10-<version>.zip | head -40
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Files and directory patterns that must NEVER appear in a release ZIP.
_EXCLUDE_PATTERNS: frozenset[str] = frozenset({
    ".git",
    ".env",
    ".env.local",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "*.egg-info",
    "dist",
    "build",
    "artifacts",
    ".devin",
    ".windsurf",
    "node_modules",
    ".DS_Store",
    "*.log",
    "*.jsonl",
    "progress.txt",
    ".venv",
    ".rfsn_cache",
    "=*",
})


def _is_excluded(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    for part in parts:
        if part in _EXCLUDE_PATTERNS:
            return True
        for pat in _EXCLUDE_PATTERNS:
            if pat.startswith("*") and part.endswith(pat[1:]):
                return True
            if pat.endswith("*") and part.startswith(pat[:-1]):
                return True
    return False


def _git_tracked_files(root: Path) -> list[Path]:
    """Return all files tracked by git in *root*."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        files = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            p = root / line
            if p.is_file():
                files.append(p)
        return files
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Git not available or not a git repo - fall back to filesystem scan
        return _filesystem_scan_files(root)


def _filesystem_scan_files(root: Path) -> list[Path]:
    """Scan filesystem for files, respecting _EXCLUDE_PATTERNS."""
    files = []
    for file_path in root.rglob("*"):
        if file_path.is_file():
            rel = str(file_path.relative_to(root))
            if not _is_excluded(rel):
                files.append(file_path)
    return files


def _read_version(root: Path) -> str:
    version_file = root / "rfsn_v10" / "_version.py"
    if not version_file.exists():
        return "unknown"
    ns: dict = {}
    exec(version_file.read_text(), ns)  # noqa: S102
    return ns.get("__version__", "unknown")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a clean release ZIP for RFSN v10"
    )
    parser.add_argument(
        "--out",
        default="dist",
        help="Output directory for the ZIP (default: dist/)",
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Version string (default: read from rfsn_v10/_version.py)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be included without writing the ZIP",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    version = args.version or _read_version(root)
    zip_name = f"rfsn-v10-{version}.zip"
    out_dir = root / args.out
    zip_path = out_dir / zip_name

    print(f"Root:    {root}")
    print(f"Version: {version}")
    print(f"Output:  {zip_path}")
    print()

    tracked = _git_tracked_files(root)
    if not tracked:
        sys.exit("No files found to include in release ZIP")

    included: list[tuple[str, Path]] = []
    excluded: list[str] = []

    for abs_path in sorted(tracked):
        rel = str(abs_path.relative_to(root))
        if _is_excluded(rel):
            excluded.append(rel)
        else:
            included.append((rel, abs_path))

    print(f"Included: {len(included)} files")
    print(f"Excluded: {len(excluded)} files")
    print()

    if args.dry_run:
        print("=== DRY RUN — files that would be included ===")
        for rel, _ in included:
            print(f"  {rel}")
        print()
        print("=== Excluded ===")
        for rel in excluded:
            print(f"  {rel}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Guard: never overwrite an existing release ZIP
    if zip_path.exists():
        sys.exit(
            f"ZIP already exists: {zip_path}\n"
            "Delete it or change --version before re-running."
        )

    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for rel, abs_path in included:
            arcname = os.path.join(f"rfsn-v10-{version}", rel)
            zf.write(abs_path, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created: {zip_path}  ({size_mb:.2f} MB)")

    # Sanity guard: verify no excluded patterns slipped in
    bad: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            # Strip the top-level prefix before checking
            rel = "/".join(name.split("/")[1:])
            if rel and _is_excluded(rel):
                bad.append(name)
    if bad:
        zip_path.unlink()
        sys.exit(
            "ZIP contained excluded files and was deleted:\n"
            + "\n".join(f"  {b}" for b in bad)
        )

    print("Verification passed.")


if __name__ == "__main__":
    main()
