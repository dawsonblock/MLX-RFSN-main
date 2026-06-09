"""
Patch management for generating and applying diffs.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import difflib
import uuid
import shutil

from .schemas import PatchPlan, FileChange
from .orchestrator import Orchestrator
from .tool_runner import ToolRunner

logger = __import__('logging').getLogger(__name__)


@dataclass
class AppliedPatch:
    """Record of a patch that has been applied."""
    patch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plan_id: str = ""
    file_changes: List[FileChange] = field(default_factory=list)
    applied_at: float = field(default_factory=lambda: __import__('time').time())
    success: bool = False
    error_message: Optional[str] = None
    backup_paths: List[Path] = field(default_factory=list)  # Backups made before applying
    
    def __post_init__(self):
        if not self.plan_id:
            raise ValueError("Patch must be associated with a plan ID")


class PatchManager:
    """
    Manages the generation and application of patches.
    
    The patch manager takes patch plans and applies them to files,
    creating backups and recording what was changed for potential rollback.
    """
    
    def __init__(self, orchestrator: Orchestrator, tool_runner: ToolRunner):
        self.orchestrator = orchestrator
        self.tool_runner = tool_runner
        self.applied_patches: List[AppliedPatch] = []
        self.failed_patches: List[AppliedPatch] = []
    
    def generate_diff(self, file_path: Path, old_content: str, 
                     new_content: str) -> List[str]:
        """
        Generate a unified diff between two versions of a file.
        
        Args:
            file_path: Path to the file (for header)
            old_content: Original content
            new_content: New content
            
        Returns:
            List of diff lines
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
        
        return list(diff)
    
    def apply_file_change(self, file_change: FileChange) -> Tuple[bool, Optional[str]]:
        """
        Apply a single file change.
        
        Args:
            file_change: The change to apply
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            file_path = Path(file_change.file_path)
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_change.change_type == "modified":
                if file_change.old_content is None or file_change.new_content is None:
                    return (False, "Modified file change missing old or new content")
                
                # Verify current content matches expected old content (if provided)
                if file_change.old_content is not None:
                    if file_path.exists():
                        current_content = file_path.read_text(encoding="utf-8")
                        if current_content != file_change.old_content:
                            return (False, f"Current content does not match expected old content for {file_path}")
                    else:
                        return (False, f"File does not exist but old content was expected: {file_path}")
                
                # Write new content
                file_path.write_text(file_change.new_content, encoding="utf-8")
                return (True, None)
                
            elif file_change.change_type == "added":
                if file_change.new_content is None:
                    return (False, "Added file change missing new content")
                
                if file_path.exists():
                    return (False, f"File already exists: {file_path}")
                
                file_path.write_text(file_change.new_content, encoding="utf-8")
                return (True, None)
                
            elif file_change.change_type == "deleted":
                if file_change.old_content is not None:
                    if not file_path.exists():
                        return (False, f"File does not exist to delete: {file_path}")
                    
                    current_content = file_path.read_text(encoding="utf-8")
                    if current_content != file_change.old_content:
                        return (False, f"Current content does not match expected content for deletion: {file_path}")
                
                if file_path.exists():
                    file_path.unlink()
                return (True, None)
                
            else:
                return (False, f"Unknown change type: {file_change.change_type}")
                
        except Exception as e:
            return (False, f"Failed to apply file change: {str(e)}")
    
    def apply_patch_plan(self, plan: PatchPlan, 
                        create_backups: bool = True) -> AppliedPatch:
        """
        Apply a patch plan to the filesystem.
        
        Args:
            plan: The patch plan to apply
            create_backups: Whether to create backups before modifying files
            
        Returns:
            AppliedPatch record of what was done
        """
        applied_patch = AppliedPatch(
            plan_id=plan.plan_id,
            file_changes=plan.file_changes.copy()
        )
        
        if create_backups:
            snapshot_root = self._create_rollback_snapshot(plan.file_changes)
            applied_patch.backup_paths.append(snapshot_root)
        
        all_success = True
        error_messages: list[str] = []

        try:
            self.transactional_apply(plan)
        except Exception as e:
            all_success = False
            error_messages.append(str(e))
            logger.error(f"Failed to apply patch plan {plan.plan_id}: {e}")
        
        # Record the result
        applied_patch.success = all_success
        if error_messages:
            applied_patch.error_message = "; ".join(error_messages)
        
        if all_success:
            self.applied_patches.append(applied_patch)
            logger.info(f"Successfully applied patch plan {plan.plan_id}")
        else:
            self.failed_patches.append(applied_patch)
            logger.error(f"Failed to apply patch plan {plan.plan_id}: {applied_patch.error_message}")
        
        return applied_patch

    def _create_rollback_snapshot(self, file_changes: List[FileChange]) -> Path:
        """Create rollback snapshot for files that currently exist."""
        snapshot_root = Path.cwd() / ".tmp" / "rollback_snapshot" / str(uuid.uuid4())
        snapshot_root.mkdir(parents=True, exist_ok=True)

        for file_change in file_changes:
            file_path = Path(file_change.file_path)
            if file_path.exists() and file_path.is_file():
                backup_path = snapshot_root / file_path.as_posix().lstrip("/")
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file_path, backup_path)

        return snapshot_root

    def _restore_from_snapshot(
        self,
        snapshot_root: Path,
        file_changes: List[FileChange],
        existing_paths_before: set[Path],
    ) -> None:
        """Restore all files from snapshot and remove files created during failed apply."""
        for file_change in file_changes:
            file_path = Path(file_change.file_path)
            backup_path = snapshot_root / file_path.as_posix().lstrip("/")
            if file_path in existing_paths_before:
                if backup_path.exists():
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, file_path)
            elif file_path.exists():
                file_path.unlink()

    def transactional_apply(self, plan: PatchPlan) -> None:
        """Apply all file changes atomically, rolling back on the first failure."""
        snapshot_root = self._create_rollback_snapshot(plan.file_changes)
        existing_paths_before = {
            Path(file_change.file_path)
            for file_change in plan.file_changes
            if Path(file_change.file_path).exists()
        }

        try:
            for file_change in plan.file_changes:
                success, error_message = self.apply_file_change(file_change)
                if not success:
                    raise RuntimeError(f"{file_change.file_path}: {error_message}")
        except Exception:
            self._restore_from_snapshot(snapshot_root, plan.file_changes, existing_paths_before)
            raise
        finally:
            shutil.rmtree(snapshot_root, ignore_errors=True)
    
    def generate_patch_from_plan(self, plan: PatchPlan) -> str:
        """
        Generate a unified diff patch from a patch plan.
        
        Args:
            plan: The patch plan to generate diff for
            
        Returns:
            Unified diff string
        """
        diff_lines = []
        
        for file_change in plan.file_changes:
            if file_change.change_type == "modified" and file_change.old_content is not None and file_change.new_content is not None:
                diff = self.generate_diff(
                    Path(file_change.file_path),
                    file_change.old_content,
                    file_change.new_content
                )
                diff_lines.extend(diff)
            elif file_change.change_type == "added" and file_change.new_content is not None:
                diff_lines.append(f"--- /dev/null")
                diff_lines.append(f"+++ {file_change.file_path}")
                diff_lines.append("@@ -0,0 +1,@@ ")
                for line in file_change.new_content.splitlines(keepends=True):
                    diff_lines.append(f"+{line}")
            elif file_change.change_type == "deleted" and file_change.old_content is not None:
                diff_lines.append(f"--- {file_change.file_path}")
                diff_lines.append(f"+++ /dev/null")
                diff_lines.append(f"@@ -1,0 +0,0 @@ ")
                for line in file_change.old_content.splitlines(keepends=True):
                    diff_lines.append(f"-{line}")
        
        return "\n".join(diff_lines)
    
    def rollback_patch(self, applied_patch: AppliedPatch) -> bool:
        """
        Rollback a previously applied patch.
        
        Args:
            applied_patch: The patch to rollback
            
        Returns:
            True if rollback successful, False otherwise
        """
        if not applied_patch.success:
            logger.warning("Cannot rollback unsuccessful patch")
            return False
        
        # Apply the reverse of each file change
        all_success = True
        error_messages = []
        
        for file_change in reversed(applied_patch.file_changes):  # Reverse order for safety
            try:
                # Create reverse change
                reverse_change = FileChange(
                    file_path=file_change.file_path,
                    old_content=file_change.new_content,
                    new_content=file_change.old_content,
                    change_type="modified" if file_change.change_type == "modified" else
                               "added" if file_change.change_type == "deleted" else
                               "deleted" if file_change.change_type == "added" else
                               file_change.change_type
                )
                
                success, error_message = self.apply_file_change(reverse_change)
                if not success:
                    all_success = False
                    error_messages.append(f"{file_change.file_path}: {error_message}")
                else:
                    logger.info(f"Rolled back change to {file_change.file_path}")
                    
            except Exception as e:
                all_success = False
                error_messages.append(f"{file_change.file_path}: {str(e)}")
        
        if all_success:
            logger.info(f"Successfully rolled back patch {applied_patch.patch_id}")
            return True
        else:
            logger.error(f"Failed to rollback patch {applied_patch.patch_id}: {'; '.join(error_messages)}")
            return False
    
    def get_patch_history(self) -> List[AppliedPatch]:
        """Get history of all applied patches."""
        return self.applied_patches.copy()
    
    def get_failed_patches(self) -> List[AppliedPatch]:
        """Get history of failed patches."""
        return self.failed_patches.copy()