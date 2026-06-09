"""
Simple patch utility for generating and applying diffs.
"""

from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import difflib

logger = __import__('logging').getLogger(__name__)


@dataclass
class SimpleFileChange:
    """Simple representation of a file change."""
    file_path: Path
    old_content: Optional[str] = None
    new_content: Optional[str] = None


def generate_unified_diff(file_path: Path, old_content: str, new_content: str) -> str:
    """
    Generate a unified diff between two versions of a file.
    
    Args:
        file_path: Path to the file (for header)
        old_content: Original content
        new_content: New content
        
    Returns:
        Unified diff string
    """
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"{file_path} (original)",
        tofile=f"{file_path} (modified)",
        lineterm=""  # We'll add newlines ourselves
    )
    
    return "".join(diff)


def apply_simple_patch(file_path: Path, old_content: str, new_content: str) -> bool:
    """
    Apply a simple patch to a file.
    
    Args:
        file_path: Path to the file to patch
        old_content: Expected original content
        new_content: New content to write
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Verify current content matches expected (if provided)
        if old_content is not None:
            if file_path.exists():
                current_content = file_path.read_text(encoding="utf-8")
                if current_content != old_content:
                    logger.error(f"Current content does not match expected for {file_path}")
                    return False
            else:
                logger.error(f"File does not exist but old content was expected: {file_path}")
                return False
        
        # Write new content
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(new_content, encoding="utf-8")
        logger.info(f"Applied patch to {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply patch to {file_path}: {e}")
        return False


def create_patch_file(changes: List[SimpleFileChange], output_path: Path) -> bool:
    """
    Create a patch file from a list of file changes.
    
    Args:
        changes: List of file changes
        output_path: Path to write the patch file to
        
        Returns:
            True if successful, False otherwise
    """
    try:
        patch_lines = []
        
        for change in changes:
            if change.old_content is not None and change.new_content is not None:
                # Modification
                diff = generate_unified_diff(
                    change.file_path,
                    change.old_content,
                    change.new_content
                )
                patch_lines.append(diff)
            elif change.old_content is None and change.new_content is not None:
                # Addition
                patch_lines.append(f"--- /dev/null")
                patch_lines.append(f"+++ {change.file_path}")
                patch_lines.append("@@ -0,0 +1,@@ ")
                for line in change.new_content.splitlines(keepends=True):
                    patch_lines.append(f"+{line}")
            elif change.old_content is not None and change.new_content is None:
                # Deletion
                patch_lines.append(f"--- {change.file_path}")
                patch_lines.append(f"+++ /dev/null")
                patch_lines.append(f"@@ -1,0 +0,0 @@ ")
                for line in change.old_content.splitlines(keepends=True):
                    patch_lines.append(f"-{line}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(patch_lines), encoding="utf-8")
        logger.info(f"Patch file created at {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create patch file: {e}")
        return False


def apply_patch_file(patch_path: Path, base_directory: Optional[Path] = None) -> bool:
    """
    Apply a patch file to files.
    
    Args:
        patch_path: Path to the patch file
        base_directory: Base directory to apply patch relative to (optional)
        
        Returns:
            True if successful, False otherwise
    """
    # This is a simplified implementation - a full patch applier would be more complex
    logger.warning("apply_patch_file is not fully implemented - use agent_core/patch_manager.py for production use")
    return False


# Convenience functions
def create_diff(old_file: Path, new_file: Path) -> str:
    """Create a diff between two files."""
    try:
        old_content = old_file.read_text(encoding="utf-8") if old_file.exists() else ""
        new_content = new_file.read_text(encoding="utf-8") if new_file.exists() else ""
        return generate_unified_diff(old_file, old_content, new_content)
    except Exception as e:
        logger.error(f"Failed to create diff: {e}")
        return ""


def apply_text_patch(file_path: Path, old_text: str, new_text: str) -> bool:
    """Apply a text patch to a file."""
    return apply_simple_patch(file_path, old_text, new_text)