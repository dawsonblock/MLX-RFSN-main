#!/usr/bin/env python3
"""Create immutable source/artifact bundle for release validation.

This script creates a reproducible bundle containing:
- Source tree SHA256 hash
- Git commit and dirty state
- Release identity from release.toml
- Artifact manifest with SHA256 hashes
- Bundle metadata for validation

The bundle can be used to verify that artifacts were generated from
a specific source state and to validate promotion eligibility.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with file_path.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_directory_hash(root: Path, ignore_patterns: list[str]) -> str:
    """Compute SHA256 hash of all files in a directory."""
    sha256_hash = hashlib.sha256()

    for file_path in sorted(root.rglob("*")):
        if file_path.is_file():
            # Skip ignored patterns
            if any(pattern in str(file_path) for pattern in ignore_patterns):
                continue

            # Include relative path in hash
            rel_path = file_path.relative_to(root)
            sha256_hash.update(str(rel_path).encode())

            # Include file content
            with file_path.open("rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def get_git_state() -> dict[str, Any]:
    """Get Git repository state."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"

    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        is_dirty = bool(dirty)
    except (subprocess.CalledProcessError, FileNotFoundError):
        is_dirty = False

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch = "unknown"

    return {
        "commit": commit,
        "branch": branch,
        "dirty": is_dirty,
    }


def load_release_config(root: Path) -> dict[str, Any]:
    """Load release.toml configuration."""
    config_path = root / "release.toml"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def create_artifact_manifest(artifacts_dir: Path) -> dict[str, Any]:
    """Create manifest of benchmark artifacts with hashes."""
    manifest = {
        "artifacts": {},
        "total_hash": "",
    }

    if not artifacts_dir.exists():
        return manifest

    artifact_hashes = []

    for artifact_path in sorted(artifacts_dir.rglob("*")):
        if artifact_path.is_file():
            rel_path = artifact_path.relative_to(artifacts_dir)
            file_hash = compute_file_hash(artifact_path)
            manifest["artifacts"][str(rel_path)] = {
                "sha256": file_hash,
                "size": artifact_path.stat().st_size,
            }
            artifact_hashes.append(f"{rel_path}:{file_hash}")

    # Compute total hash of all artifacts
    if artifact_hashes:
        combined = "|".join(sorted(artifact_hashes))
        manifest["total_hash"] = hashlib.sha256(combined.encode()).hexdigest()

    return manifest


def create_bundle(
    root: Path,
    output_dir: Path,
    include_artifacts: bool = True,
) -> dict[str, Any]:
    """Create immutable release bundle.

    Args:
        root: Repository root directory.
        output_dir: Directory to write bundle files.
        include_artifacts: Whether to include artifact hashes.

    Returns:
        Bundle metadata dictionary.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get Git state
    git_state = get_git_state()

    # Load release configuration
    release_config = load_release_config(root)

    # Compute source tree hash (excluding common build artifacts)
    ignore_patterns = [
        ".git",
        "__pycache__",
        "*.pyc",
        "dist",
        "build",
        "*.egg-info",
        ".pytest_cache",
        ".mypy_cache",
        "artifacts",
        ".DS_Store",
    ]
    source_hash = compute_directory_hash(root, ignore_patterns)

    # Create artifact manifest if requested
    artifact_manifest = {}
    if include_artifacts:
        artifacts_dir = root / "artifacts"
        artifact_manifest = create_artifact_manifest(artifacts_dir)

    # Build bundle metadata
    bundle_metadata = {
        "release_id": release_config.get("release_id", "unknown"),
        "display_name": release_config.get("display_name", "unknown"),
        "package_version": release_config.get("package_version", "unknown"),
        "git": git_state,
        "source_tree_hash": source_hash,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "artifacts": artifact_manifest,
        "bundle_created_at": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip(),
    }

    # Mark bundle as invalid if source tree is dirty
    if git_state["dirty"]:
        bundle_metadata["validation_status"] = "DIRTY_SOURCE_TREE"
        bundle_metadata["validation_message"] = (
            "Source tree has uncommitted changes. "
            "Bundle is not eligible for promotion."
        )
    else:
        bundle_metadata["validation_status"] = "VALID"
        bundle_metadata["validation_message"] = "Source tree is clean"

    # Write bundle metadata
    bundle_path = output_dir / "bundle.json"
    with bundle_path.open("w", encoding="utf-8") as f:
        json.dump(bundle_metadata, f, indent=2, sort_keys=True)

    # Write bundle hash for quick verification
    bundle_hash = compute_file_hash(bundle_path)
    hash_path = output_dir / "bundle.sha256"
    hash_path.write_text(f"{bundle_hash}  bundle.json\n")

    print(f"Created release bundle: {bundle_path}")
    print(f"Bundle hash: {bundle_hash}")
    print(f"Validation status: {bundle_metadata['validation_status']}")

    return bundle_metadata


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create immutable source/artifact bundle"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/release"),
        help="Output directory for bundle files",
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip artifact hash computation",
    )
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = args.output_dir

    try:
        bundle_metadata = create_bundle(
            root,
            output_dir,
            include_artifacts=not args.no_artifacts,
        )

        # Exit with error if bundle is invalid
        if bundle_metadata["validation_status"] != "VALID":
            print("\nERROR: Bundle validation failed")
            print(f"Reason: {bundle_metadata['validation_message']}")
            return 1

        print("\nBundle created successfully")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
