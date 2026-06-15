#!/usr/bin/env python3
"""Archive current benchmark artifacts to history.

Fix #12: Complete artifact-history migration.

This script moves artifacts from artifacts/bench/current/ to
artifacts/bench/history/ with proper categorization and timestamping.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def archive_current_artifacts(
    current_dir: Path,
    history_dir: Path,
    category: str = "general",
) -> None:
    """Archive current artifacts to history with timestamp.
    
    Args:
        current_dir: Path to current artifacts directory
        history_dir: Path to history directory
        category: Category subdirectory (debug, kernel, legacy, memory, etc.)
    """
    if not current_dir.exists():
        print(f"Current artifacts directory does not exist: {current_dir}")
        return
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_subdir = history_dir / category / timestamp
    archive_subdir.mkdir(parents=True, exist_ok=True)
    
    # Move all files from current to archive
    for item in current_dir.iterdir():
        if item.is_file():
            dest = archive_subdir / item.name
            shutil.move(str(item), str(dest))
            print(f"Archived: {item.name} -> {dest}")
    
    print(f"Archived artifacts to: {archive_subdir}")


def main() -> None:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    current_dir = project_root / "artifacts" / "bench" / "current"
    history_dir = project_root / "artifacts" / "bench" / "history"
    
    # Archive to general category by default
    archive_current_artifacts(current_dir, history_dir, category="general")


if __name__ == "__main__":
    main()
