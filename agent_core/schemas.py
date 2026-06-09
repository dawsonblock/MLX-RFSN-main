"""
Core data structures for the Dawson AI Core agent system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pathlib import Path
import uuid


class TaskStatus(Enum):
    """Status of a task or subtask."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TestResultStatus(Enum):
    """Status of test execution."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


TestResultStatus.__test__ = False


@dataclass
class TestResult:
    """Result of executing a single test."""
    test_name: str
    status: TestResultStatus
    duration_ms: float
    message: Optional[str] = None
    traceback: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    
    @property
    def passed(self) -> bool:
        return self.status == TestResultStatus.PASSED


TestResult.__test__ = False


@dataclass
class FileChange:
    """Represents a change to a file."""
    file_path: Path
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    change_type: str = "modified"  # modified, added, deleted
    
    @property
    def is_modified(self) -> bool:
        return self.change_type == "modified" and self.old_content is not None and self.new_content is not None
    
    @property
    def is_added(self) -> bool:
        return self.change_type == "added"
    
    @property
    def is_deleted(self) -> bool:
        return self.change_type == "deleted"


@dataclass
class PatchPlan:
    """A proposed set of changes to fix issues."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    file_changes: List[FileChange] = field(default_factory=list)
    rationale: str = ""
    estimated_impact: str = ""  # e.g., "fixes test failures", "improves performance"
    risks: List[str] = field(default_factory=list)
    
    def add_file_change(self, file_path: str | Path, old_content: Optional[str] = None, 
                        new_content: Optional[str] = None, change_type: str = "modified"):
        """Add a file change to the plan."""
        self.file_changes.append(FileChange(
            file_path=Path(file_path),
            old_content=old_content,
            new_content=new_content,
            change_type=change_type
        ))


@dataclass
class TaskState:
    """State of a task being executed by the agent."""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: Priority = Priority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    subtasks: List['TaskState'] = field(default_factory=list)
    parent_task_id: Optional[str] = None
    patches_applied: List[PatchPlan] = field(default_factory=list)
    test_results: List[TestResult] = field(default_factory=list)
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    def add_subtask(self, description: str) -> 'TaskState':
        """Add a subtask and return it for further configuration."""
        subtask = TaskState(
            description=description,
            parent_task_id=self.task_id
        )
        self.subtasks.append(subtask)
        return subtask
    
    def add_test_result(self, test_name: str, status: TestResultStatus, duration_ms: float,
                       message: Optional[str] = None, traceback: Optional[str] = None,
                       stdout: Optional[str] = None, stderr: Optional[str] = None):
        """Add a test result to this task."""
        self.test_results.append(TestResult(
            test_name=test_name,
            status=status,
            duration_ms=duration_ms,
            message=message,
            traceback=traceback,
            stdout=stdout,
            stderr=stderr
        ))
    
    @property
    def is_complete(self) -> bool:
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED
    
    @property
    def all_tests_passed(self) -> bool:
        return all(t.passed for t in self.test_results) if self.test_results else False


@dataclass
class AgentVerdict:
    """Final verdict from the agent after executing a task."""
    task_id: str
    success: bool
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_task_state(cls, task_state: TaskState) -> 'AgentVerdict':
        """Create a verdict from a task state."""
        return cls(
            task_id=task_state.task_id,
            success=task_state.is_complete and task_state.all_tests_passed,
            summary=f"Task '{task_state.description}' {'completed successfully' if task_state.is_complete and task_state.all_tests_passed else 'failed'}",
            details={
                "status": task_state.status.value,
                "subtasks_count": len(task_state.subtasks),
                "patches_applied": len(task_state.patches_applied),
                "tests_run": len(task_state.test_results),
                "tests_passed": sum(1 for t in task_state.test_results if t.passed),
                "duration_seconds": (
                    (task_state.completed_at or datetime.now()) - 
                    (task_state.started_at or task_state.created_at)
                ).total_seconds() if task_state.started_at else 0
            },
            lessons_learned=[
                f"Test results: {sum(1 for t in task_state.test_results if t.passed)}/{len(task_state.test_results)} passed"
            ] if task_state.test_results else [],
            recommendations=task_state.error_message and [f"Address error: {task_state.error_message}"] or []
        )