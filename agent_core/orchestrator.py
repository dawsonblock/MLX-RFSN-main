"""
Task state machine for orchestrating agent workflows.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable
from datetime import datetime
import asyncio
import logging
import pickle
from pathlib import Path

from .schemas import TaskState, TaskStatus, Priority, AgentVerdict, TestResult

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Orchestrates task execution by managing task state and transitions.
    
    The orchestrator is responsible for:
    - Creating and tracking task states
    - Managing task hierarchies (parent/child relationships)
    - Transitioning tasks between states
    - Executing task callbacks
    - Collecting results and generating verdicts
    """
    
    def __init__(self, auto_save_path: Optional[str | Path] = None):
        self.tasks: Dict[str, TaskState] = {}
        self.root_tasks: List[TaskState] = []
        self.auto_save_path = Path(auto_save_path) if auto_save_path else None
        logger.info("Orchestrator initialized")

    def _persist_if_configured(self) -> None:
        if self.auto_save_path is not None:
            self.save_state(self.auto_save_path)
    
    def create_task(self, description: str, priority: Priority = Priority.MEDIUM,
                   parent_task_id: Optional[str] = None) -> TaskState:
        """
        Create a new task.
        
        Args:
            description: Human-readable description of the task
            priority: Priority level of the task
            parent_task_id: ID of parent task (if this is a subtask)
            
        Returns:
            The created TaskState
        """
        task = TaskState(
            description=description,
            priority=priority,
            parent_task_id=parent_task_id
        )
        
        self.tasks[task.task_id] = task
        
        if parent_task_id:
            parent = self.tasks.get(parent_task_id)
            if parent:
                parent.subtasks.append(task)
            else:
                logger.warning(f"Parent task {parent_task_id} not found")
        else:
            self.root_tasks.append(task)

        self._persist_if_configured()
        
        logger.info(f"Created task {task.task_id}: {description}")
        return task
    
    def start_task(self, task_id: str) -> bool:
        """
        Mark a task as started.
        
        Args:
            task_id: ID of task to start
            
        Returns:
            True if task was started, False if not found or already started
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False
        
        if task.status != TaskStatus.PENDING:
            logger.warning(f"Task {task_id} is not pending (current status: {task.status.value})")
            return False
        
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        self._persist_if_configured()
        logger.info(f"Started task {task_id}: {task.description}")
        return True
    
    def complete_task(self, task_id: str, success: bool = True) -> bool:
        """
        Mark a task as completed.
        
        Args:
            task_id: ID of task to complete
            success: Whether the task completed successfully
            
        Returns:
            True if task was completed, False if not found or already completed
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False
        
        if task.status != TaskStatus.IN_PROGRESS:
            logger.warning(f"Task {task_id} is not in progress (current status: {task.status.value})")
            return False
        
        task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
        task.completed_at = datetime.now()
        self._persist_if_configured()
        logger.info(f"Completed task {task_id}: {task.description} (success: {success})")
        
        # If all subtasks are complete, we could auto-complete parent
        # but we'll leave that to explicit calls for now
        return True
    
    def fail_task(self, task_id: str, error_message: str) -> bool:
        """
        Mark a task as failed.
        
        Args:
            task_id: ID of task to fail
            error_message: Description of the failure
            
        Returns:
            True if task was failed, False if not found
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False
        
        task.status = TaskStatus.FAILED
        task.error_message = error_message
        task.completed_at = datetime.now()
        self._persist_if_configured()
        logger.error(f"Failed task {task_id}: {task.description} - {error_message}")
        return True
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task.
        
        Args:
            task_id: ID of task to cancel
            
        Returns:
            True if task was cancelled, False if not found or already completed
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False
        
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            logger.warning(f"Cannot cancel task {task_id} - already finished")
            return False
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now()
        self._persist_if_configured()
        logger.info(f"Cancelled task {task_id}: {task.description}")
        return True
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def get_root_tasks(self) -> List[TaskState]:
        """Get all root tasks (tasks with no parent)."""
        return self.root_tasks.copy()
    
    def get_pending_tasks(self) -> List[TaskState]:
        """Get all pending tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.PENDING]
    
    def get_in_progress_tasks(self) -> List[TaskState]:
        """Get all in-progress tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.IN_PROGRESS]
    
    def get_completed_tasks(self) -> List[TaskState]:
        """Get all completed tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED]
    
    def get_failed_tasks(self) -> List[TaskState]:
        """Get all failed tasks."""
        return [task for task in self.tasks.values() if task.status == TaskStatus.FAILED]
    
    def get_task_tree(self, task_id: Optional[str] = None) -> List[TaskState]:
        """
        Get a task and all its subtasks.
        
        Args:
            task_id: ID of task to get tree for (None for all root tasks)
            
        Returns:
            List of tasks in the tree
        """
        if task_id is None:
            return self.get_root_tasks()
        
        task = self.tasks.get(task_id)
        if not task:
            return []
        
        # Return task and all descendants
        result = [task]
        for subtask in task.subtasks:
            result.extend(self.get_task_tree(subtask.task_id))
        return result

    def add_test_result(self, task_id: str, test_result: TestResult) -> bool:
        """
        Add a test result to a task.
        
        Args:
            task_id: ID of task to add test result to
            test_result: TestResult object to add
            
        Returns:
            True if test result was added, False if task not found
        """
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False
        
        task.test_results.append(test_result)
        self._persist_if_configured()
        logger.info(f"Added test result '{test_result.test_name}' to task {task_id}")
        return True

    def save_state(self, filepath: str | Path) -> None:
        """Persist orchestrator task state to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tasks": self.tasks,
            "root_task_ids": [task.task_id for task in self.root_tasks],
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def load_state(self, filepath: str | Path) -> None:
        """Load orchestrator task state from disk."""
        path = Path(filepath)
        with path.open("rb") as f:
            payload = pickle.load(f)

        loaded_tasks = payload.get("tasks", {})
        root_task_ids = payload.get("root_task_ids", [])

        if not isinstance(loaded_tasks, dict):
            raise ValueError("Invalid state payload: tasks must be a dictionary")

        self.tasks = loaded_tasks
        self.root_tasks = [
            task for task_id, task in self.tasks.items() if task_id in root_task_ids
        ]

    def reset(self):
        """Reset the orchestrator to initial state."""
        self.tasks.clear()
        self.root_tasks.clear()
        self._persist_if_configured()
        logger.info("Orchestrator reset")