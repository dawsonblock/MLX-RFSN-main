#!/usr/bin/env python3
"""Sync version from release.toml to pyproject.toml and README.md.

Fix #13: Unify manifest, README, canonical config and package version.

This script ensures that the single source of truth (release.toml) is
reflected in pyproject.toml and README.md.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def load_release_toml(project_root: Path) -> dict[str, Any]:
    """Load release.toml configuration."""
    release_toml_path = project_root / "release.toml"
    if not release_toml_path.exists():
        raise FileNotFoundError(f"release.toml not found: {release_toml_path}")
    
    with release_toml_path.open("rb") as f:
        return tomllib.load(f)


def update_pyproject_toml(project_root: Path, release_config: dict[str, Any]) -> None:
    """Update pyproject.toml fallback_version to match release.toml package_version."""
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found: {pyproject_path}")
    
    with pyproject_path.open("r") as f:
        content = f.read()
    
    package_version = release_config.get("package_version")
    if not package_version:
        raise ValueError("package_version not found in release.toml")
    
    # Update fallback_version line
    pattern = r'fallback_version = ".*?"'
    replacement = f'fallback_version = "{package_version}"  # Synchronized with release.toml package_version'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        with pyproject_path.open("w") as f:
            f.write(content)
        print(f"Updated pyproject.toml fallback_version to {package_version}")
    else:
        print(f"Warning: fallback_version pattern not found in pyproject.toml")


def update_readme(project_root: Path, release_config: dict[str, Any]) -> None:
    """Update README.md to reference release.toml as source of truth."""
    readme_path = project_root / "README.md"
    if not readme_path.exists():
        raise FileNotFoundError(f"README.md not found: {readme_path}")
    
    with readme_path.open("r") as f:
        content = f.read()
    
    release_id = release_config.get("release_id")
    package_version = release_config.get("package_version")
    artifact_schema = release_config.get("artifact_schema")
    
    # Update version references if they exist
    if "**Package Version:**" in content:
        # Check if it already references release.toml
        if "(from release.toml)" not in content:
            # Update to reference release.toml
            pattern = r'\*\*Package Version:\*\* .*?'
            replacement = f'**Package Version:** `{package_version}` (from release.toml)'
            content = re.sub(pattern, replacement, content)
    
    # Add release.toml reference if not present
    if "release.toml" not in content:
        # Add version section after status section
        status_section = r'(\*\*Promotion allowed:\*\* .*)'
        version_section = r'\1\n\n**Release ID:** `{release_id}` (from release.toml)\n**Package Version:** `{package_version}` (from release.toml)\n**Artifact Schema:** `{artifact_schema}` (from release.toml)'
        content = re.sub(status_section, version_section, content)
    
    with readme_path.open("w") as f:
        f.write(content)
    print(f"Updated README.md to reference release.toml")


def main() -> None:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    
    # Load release.toml
    release_config = load_release_toml(project_root)
    
    # Update pyproject.toml
    update_pyproject_toml(project_root, release_config)
    
    # Update README.md
    update_readme(project_root, release_config)
    
    print("Version sync complete. release.toml is the single source of truth.")


if __name__ == "__main__":
    main()
