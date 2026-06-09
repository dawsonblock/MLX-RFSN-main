"""
Task planner for breaking down complex tasks into executable steps.
"""

from __future__ import annotations

from typing import List, Callable, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .schemas import TaskState, Priority, TaskStatus
from .orchestrator import Orchestrator


@dataclass
class Step:
    """A single step in a plan."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    action: Optional[Callable[[], Any]] = None  # Function to execute
    success_criteria: Optional[Callable[[Any], bool]] = None  # Function to validate success
    estimated_duration: Optional[float] = None  # In seconds
    dependencies: List[str] = field(default_factory=list)  # Step IDs this depends on
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[float] = None
    
    def __post_init__(self):
        if not self.description:
            raise ValueError("Step description is required")


@dataclass
class Plan:
    """A collection of steps that accomplish a goal."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    steps: List[Step] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    
    def add_step(self, description: str, action: Optional[Callable[[], Any]] = None,
                success_criteria: Optional[Callable[[Any], bool]] = None,
                estimated_duration: Optional[float] = None,
                dependencies: Optional[List[str]] = None) -> Step:
        """Add a step to the plan."""
        step = Step(
            description=description,
            action=action,
            success_criteria=success_criteria,
            estimated_duration=estimated_duration,
            dependencies=dependencies or []
        )
        self.steps.append(step)
        return step
    
    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a step by ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_ready_steps(self, completed_steps: List[str]) -> List[Step]:
        """
        Get steps that are ready to execute (all dependencies met).
        
        Args:
            completed_steps: List of step IDs that have been completed
            
        Returns:
            List of steps ready to execute
        """
        ready = []
        for step in self.steps:
            if step.step_id in completed_steps:
                continue  # Already completed
            
            # Check if all dependencies are met
            deps_met = all(dep in completed_steps for dep in step.dependencies)
            if deps_met:
                ready.append(step)
        
        return ready


class Planner:
    """
    Planner that breaks down complex tasks into executable steps.
    
    The planner takes a high-level goal and breaks it down into
    concrete, executable steps with clear success criteria.
    """
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.plans: dict[str, Plan] = {}
        logger = __import__('logging').getLogger(__name__)
        self.logger = logger
    
    def create_plan(self, goal: str) -> Plan:
        """
        Create a new plan for achieving a goal.
        
        Args:
            goal: High-level goal to achieve
            
        Returns:
            The created Plan
        """
        plan = Plan(goal=goal)
        self.plans[plan.plan_id] = plan
        self.logger.info(f"Created plan {plan.plan_id}: {goal}")
        return plan
    
    def plan_repo_repair_task(self, repo_path: str) -> Plan:
        """
        Create a plan for repairing a repository based on test failures.
        
        This is the core planning logic for the Tool-Verified Repo Agent MVP.
        
        Args:
            repo_path: Path to the repository to repair
            
        Returns:
            Plan for repository repair
        """
        plan = self.create_plan(f"Repair repository at {repo_path}")
        
        # Step 1: Extract repository info and build file tree
        plan.add_step(
            description="Extract repository information and build file tree",
            estimated_duration=5.0
        )
        
        # Step 2: Detect language/runtime and dependencies
        plan.add_step(
            description="Detect language/runtime and identify dependencies",
            estimated_duration=3.0,
            dependencies=[plan.steps[0].step_id]  # Depends on step 1
        )
        
        # Step 3: Install dependencies or identify missing ones
        plan.add_step(
            description="Install dependencies or report missing dependencies",
            estimated_duration=30.0,  # Could take time for npm/pip install
            dependencies=[plan.steps[1].step_id]  # Depends on step 2
        )
        
        # Step 4: Run compile checks
        plan.add_step(
            description="Run compile/check syntax checks",
            estimated_duration=10.0,
            dependencies=[plan.steps[2].step_id]  # Depends on step 3
        )
        
        # Step 5: Run test suite and parse failures
        plan.add_step(
            description="Run test suite and parse failures into structured data",
            estimated_duration=20.0,
            dependencies=[plan.steps[3].step_id]  # Depends on step 4
        )
        
        # Step 6: Generate repair plan based on failures
        plan.add_step(
            description="Generate repair plan based on test failures",
            estimated_duration=10.0,
            dependencies=[plan.steps[4].step_id]  # Depends on step 5
        )
        
        # Step 7: Apply patches
        plan.add_step(
            description="Apply proposed patches to fix issues",
            estimated_duration=15.0,
            dependencies=[plan.steps[5].step_id]  # Depends on step 6
        )
        
        # Step 8: Rerun tests to verify fixes
        plan.add_step(
            description="Rerun test suite to verify fixes work",
            estimated_duration=20.0,
            dependencies=[plan.steps[6].step_id]  # Depends on step 7
        )
        
        # Step 9: Store results in memory
        plan.add_step(
            description="Store repair results in persistent memory",
            estimated_duration=5.0,
            dependencies=[plan.steps[7].step_id]  # Depends on step 8
        )
        
        return plan
    
    def plan_generic_task(self, description: str) -> Plan:
        """
        Create a generic plan for a task description.
        
        Args:
            description: Description of the task
            
        Returns:
            Generic plan with placeholder steps
        """
        plan = self.create_plan(description)
        
        # Add generic steps
        plan.add_step(
            description="Analyze task requirements",
            estimated_duration=5.0
        )
        
        plan.add_step(
            description="Research and gather information",
            estimated_duration=15.0
        )
        
        plan.add_step(
            description="Develop solution approach",
            estimated_duration=10.0
        )
        
        plan.add_step(
            description="Implement solution",
            estimated_duration=30.0
        )
        
        plan.add_step(
            description="Test and validate solution",
            estimated_duration=15.0
        )
        
        plan.add_step(
            description="Document and cleanup",
            estimated_duration=5.0
        )
        
        return plan
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        return self.plans.get(plan_id)