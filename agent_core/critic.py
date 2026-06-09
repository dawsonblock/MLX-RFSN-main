"""
Critic that attacks proposed plans to identify risks and failure modes.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .schemas import PatchPlan, TaskState
from .orchestrator import Orchestrator

logger = __import__('logging').getLogger(__name__)


@dataclass
class Critique:
    """A critique of a plan or proposal."""
    critique_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    plan_id: str = ""
    description: str = ""
    severity: str = "low"  # low, medium, high, critical
    category: str = ""  # e.g., "correctness", "performance", "security", "testability"
    suggestion: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.description:
            raise ValueError("Critique description is required")


class Critic:
    """
    Critic that analyzes plans and identifies risks, flaws, and failure modes.
    
    The critic takes proposed plans and attacks them from multiple angles:
    - Does it actually solve the problem?
    - Does it introduce new problems?
    - Is it testable?
    - Is it performant?
    - Does it follow best practices?
    """
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.critiques: List[Critique] = []
    
    def critique_patch_plan(self, plan: PatchPlan) -> List[Critique]:
        """
        Critique a patch plan for potential issues.
        
        Args:
            plan: The patch plan to critique
            
        Returns:
            List of critiques identifying issues with the plan
        """
        critiques = []
        
        # Check if plan actually addresses the problem
        if not plan.description or len(plan.description.strip()) < 10:
            critiques.append(Critique(
                plan_id=plan.plan_id,
                description="Plan description is too vague or missing",
                severity="medium",
                category="clarity",
                suggestion="Provide a clear, detailed description of what the plan accomplishes"
            ))
        
        # Check for missing rationale
        if not plan.rationale or len(plan.rationale.strip()) < 10:
            critiques.append(Critique(
                plan_id=plan.plan_id,
                description="Plan rationale is missing or insufficient",
                severity="medium",
                category="justification",
                suggestion="Explain why this plan is necessary and what problem it solves"
            ))
        
        # Check file changes
        if not plan.file_changes:
            critiques.append(Critique(
                plan_id=plan.plan_id,
                description="Plan contains no file changes",
                severity="high",
                category="completeness",
                suggestion="Specify what files need to be changed to implement the solution"
            ))
        else:
            # Critique each file change
            for i, change in enumerate(plan.file_changes):
                if change.change_type == "modified":
                    if change.old_content is None:
                        critiques.append(Critique(
                            plan_id=plan.plan_id,
                            description=f"File change {i+1}: Missing old content for modified file",
                            severity="high",
                            category="correctness",
                            suggestion="Provide the original content to ensure correct replacement"
                        ))
                    elif change.new_content is None:
                        critiques.append(Critique(
                            plan_id=plan.plan_id,
                            description=f"File change {i+1}: Missing new content for modified file",
                            severity="high",
                            category="correctness",
                            suggestion="Provide the new content to replace the old content with"
                        ))
                    elif change.old_content == change.new_content:
                        critiques.append(Critique(
                            plan_id=plan.plan_id,
                            description=f"File change {i+1}: Old and new content are identical",
                            severity="low",
                            category="efficiency",
                            suggestion="Skip this change as it makes no modification"
                        ))
                
                elif change.change_type == "added":
                    if change.new_content is None:
                        critiques.append(Critique(
                            plan_id=plan.plan_id,
                            description=f"File change {i+1}: Added file missing content",
                            severity="high",
                            category="completeness",
                            suggestion="Provide the content for the new file"
                        ))
                
                elif change.change_type == "deleted":
                    if change.old_content is None:
                        critiques.append(Critique(
                            plan_id=plan.plan_id,
                            description=f"File change {i+1}: Deleted file missing old content (for verification)",
                            severity="medium",
                            category="safety",
                            suggestion="Provide original content to verify what is being deleted"
                        ))
        
        # Check for risks
        if not plan.risks:
            critiques.append(Critique(
                plan_id=plan.plan_id,
                description="No risks identified in plan",
                severity="low",
                category="risk_assessment",
                suggestion="Consider and document potential risks or downsides of this approach"
            ))
        elif len(plan.risks) == 0 or all("low" in risk.lower() or "minimal" in risk.lower() for risk in plan.risks):
            critiques.append(Critique(
                plan_id=plan.plan_id,
                description="All risks rated as low/minimal - may be underestimating",
                severity="medium",
                category="risk_assessment",
                suggestion="Carefully reconsider potential risks - all changes have some risk"
            ))
        
        # Check impact statement
        if not plan.estimated_impact or len(plan.estimated_impact.strip()) < 5:
            critiques.append(Critique(
                plan_id=plan.plan_id,
                description="Impact statement is missing or too vague",
                severity="low",
                category="measurement",
                suggestion="Describe the expected impact of this change (e.g., 'fixes test X', 'reduces latency by Y%')"
            ))
        
        return critiques
    
    def critique_task_state(self, task_state: TaskState) -> List[Critique]:
        """
        Critique a task state for potential issues in execution.
        
        Args:
            task_state: The task state to critique
            
        Returns:
            List of critiques identifying issues with the task execution
        """
        critiques = []
        
        # Check if task has been running too long without progress
        if task_state.status == TaskState.TaskStatus.IN_PROGRESS:
            if task_state.started_at:
                import datetime
                elapsed = (datetime.datetime.now() - task_state.started_at).total_seconds()
                if elapsed > 300:  # 5 minutes
                    critiques.append(Critique(
                        plan_id="",  # Not plan-specific
                        description=f"Task has been in progress for {elapsed:.0f} seconds without completion",
                        severity="medium",
                        category="performance",
                        suggestion="Consider if the task is stuck or needs to be broken down further"
                    ))
        
        # Check if task has too many failed tests without adjustment
        failed_tests = [t for t in task_state.test_results if not t.passed]
        if len(failed_tests) > 3:
            critiques.append(Critique(
                plan_id="",
                description=f"Task has {len(failed_tests)} failed tests - may need approach adjustment",
                severity="medium",
                category="effectiveness",
                suggestion="Consider if the current approach is working or if a different strategy is needed"
            ))
        
        return critiques
    
    def get_critiques_for_plan(self, plan_id: str) -> List[Critique]:
        """Get all critiques for a specific plan."""
        return [c for c in self.critiques if c.plan_id == plan_id]
    
    def get_all_critiques(self) -> List[Critique]:
        """Get all critiques."""
        return self.critiques.copy()
    
    def clear_critiques(self):
        """Clear all critiques."""
        self.critiques.clear()