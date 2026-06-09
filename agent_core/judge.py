"""
Judge that evaluates evidence and decides pass/fail for tasks.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from .schemas import TaskState, TestResult, PatchPlan, AgentVerdict, TaskStatus, Priority
from .orchestrator import Orchestrator

logger = __import__('logging').getLogger(__name__)


@dataclass
class Evidence:
    """Evidence used to make a judgment."""
    evidence_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    description: str = ""
    type: str = ""  # test_result, performance, correctness, etc.
    value: Any = None
    weight: float = 1.0  # How much this evidence should count in decision
    timestamp: float = field(default_factory=lambda: __import__('time').time())
    
    def __post_init__(self):
        if not self.description:
            raise ValueError("Evidence description is required")


@dataclass
class JudgmentCriteria:
    """Criteria for making a pass/fail judgment."""
    criterion_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    description: str = ""
    required: bool = False  # If true, must pass for overall success
    weight: float = 1.0
    threshold: Optional[float] = None  # For numeric criteria
    
    def __post_init__(self):
        if not self.description:
            raise ValueError("Judgment criteria description is required")


class Judge:
    """
    Judge that evaluates evidence and decides whether a task has succeeded.
    
    The judge looks at:
    - Test results (did tests pass?)
    - Whether the proposed solution actually fixes the problem
    - Whether any new problems were introduced
    - Performance impact
    - Code quality and best practices
    """
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.judgments: List[Dict[str, Any]] = []
    
    def judge_task_completion(self, task_state: TaskState) -> AgentVerdict:
        """
        Judge whether a task has been completed successfully.
        
        Args:
            task_state: The task state to judge
            
        Returns:
            AgentVerdict indicating success/failure and details
        """
        # Start with the basic verdict from task state
        verdict = AgentVerdict.from_task_state(task_state)
        
        # Add more detailed analysis
        evidence = self._gather_evidence(task_state)
        
        # Apply judgment criteria
        criteria_results = self._apply_judgment_criteria(evidence, task_state)
        
        # Determine final success based on criteria
        verdict.success = self._compute_overall_success(criteria_results, task_state)
        
        # Enhance the verdict with detailed reasoning
        verdict.summary = self._generate_judgment_summary(criteria_results, task_state, evidence)
        verdict.lessons_learned = self._extract_lessons_learned(task_state, evidence)
        verdict.recommendations = self._generate_recommendations(criteria_results, task_state, evidence)
        
        # Store judgment for history
        self.judgments.append({
            "task_id": task_state.task_id,
            "verdict": verdict,
            "evidence": evidence,
            "criteria_results": criteria_results,
            "timestamp": __import__('time').time()
        })
        
        logger.info(f"Judged task {task_state.task_id}: {verdict.summary}")
        return verdict
    
    def _gather_evidence(self, task_state: TaskState) -> List[Evidence]:
        """Gather evidence from the task state."""
        evidence = []
        
        # Test results evidence
        if task_state.test_results:
            passed_count = sum(1 for t in task_state.test_results if t.passed)
            total_count = len(task_state.test_results)
            pass_rate = passed_count / total_count if total_count > 0 else 0
            
            evidence.append(Evidence(
                description=f"Test pass rate: {passed_count}/{total_count} ({pass_rate:.1%})",
                type="test_results",
                value={"passed": passed_count, "total": total_count, "rate": pass_rate},
                weight=2.0  # Test results are important
            ))
            
            # Individual test failures
            for test_result in task_state.test_results:
                if not test_result.passed:
                    evidence.append(Evidence(
                        description=f"Failed test: {test_result.test_name}",
                        type="test_failure",
                        value={"test_name": test_result.test_name, "message": test_result.message},
                        weight=1.5
                    ))
        
        # Patches applied evidence
        if task_state.patches_applied:
            evidence.append(Evidence(
                description=f"Patches applied: {len(task_state.patches_applied)}",
                type="patches_applied",
                value={"count": len(task_state.patches_applied)},
                weight=1.0
            ))
        
        # Error evidence
        if task_state.error_message:
            evidence.append(Evidence(
                description=f"Error encountered: {task_state.error_message}",
                type="error",
                value={"message": task_state.error_message},
                weight=2.0  # Errors are significant
            ))
        
        # Duration evidence
        if task_state.started_at and task_state.completed_at:
            import datetime
            duration = (task_state.completed_at - task_state.started_at).total_seconds()
            evidence.append(Evidence(
                description=f"Task duration: {duration:.1f} seconds",
                type="duration",
                value={"duration_seconds": duration},
                weight=0.5
            ))
        
        return evidence
    
    def _apply_judgment_criteria(self, evidence: List[Evidence], 
                               task_state: TaskState) -> List[Dict[str, Any]]:
        """Apply judgment criteria to evidence."""
        criteria_results = []
        
        # Criterion 1: Tests must pass (unless explicitly allowed to fail)
        test_evidence = [e for e in evidence if e.type == "test_results"]
        if test_evidence:
            test_data = test_evidence[0].value
            test_pass_rate = test_data["rate"]
            # Allow some flexibility - if it's a repair task, we expect improvement, not necessarily 100%
            # For now, we'll require all tests to pass for success
            criteria_results.append({
                "criterion_id": "test_pass_rate",
                "description": "All tests must pass",
                "required": True,
                "satisfied": test_pass_rate == 1.0,
                "actual_value": test_pass_rate,
                "threshold": 1.0,
                "weight": 2.0
            })
        
        # Criterion 2: No unhandled errors
        error_evidence = [e for e in evidence if e.type == "error"]
        criteria_results.append({
            "criterion_id": "no_errors",
            "description": "No unhandled errors during execution",
            "required": True,
            "satisfied": len(error_evidence) == 0,
            "actual_value": len(error_evidence),
            "threshold": 0,
            "weight": 2.0
        })
        
        # Criterion 3: Task actually completed (not cancelled or left in progress)
        criteria_results.append({
            "criterion_id": "task_completed",
            "description": "Task reached completion state",
            "required": True,
            "satisfied": task_state.status == TaskStatus.COMPLETED,
            "actual_value": task_state.status.value,
            "threshold": TaskStatus.COMPLETED.value,
            "weight": 1.5
        })
        
        # Criterion 4: Patches were actually applied if needed
        patch_evidence = [e for e in evidence if e.type == "patches_applied"]
        if task_state.test_results and any(not t.passed for t in task_state.test_results):
            # If there were failing tests, we expect patches to have been applied
            criteria_results.append({
                "criterion_id": "patches_applied_when_needed",
                "description": "Patches applied when tests were failing",
                "required": True,
                "satisfied": len(patch_evidence) > 0,
                "actual_value": len(patch_evidence[0].value["count"]) if patch_evidence else 0,
                "threshold": 1,
                "weight": 1.5
            })
        else:
            # If tests were already passing, patches are optional
            criteria_results.append({
                "criterion_id": "patches_appropriate",
                "description": "Patch application appropriate to situation",
                "required": False,
                "satisfied": True,  # Neutral
                "actual_value": len(patch_evidence) if patch_evidence else 0,
                "threshold": 0,
                "weight": 0.5
            })
        
        return criteria_results
    
    def _compute_overall_success(self, criteria_results: List[Dict[str, Any]], 
                               task_state: TaskState) -> bool:
        """Compute overall success from criteria results."""
        if not criteria_results:
            return task_state.status == TaskStatus.COMPLETED and task_state.all_tests_passed
        
        # Check if any required criteria failed
        for criterion in criteria_results:
            if criterion.get("required", False) and not criterion.get("satisfied", False):
                return False
        
        # If we get here, all required criteria passed
        # Could add weighted scoring here if needed
        return True
    
    def _generate_judgment_summary(self, criteria_results: List[Dict[str, Any]], 
                                 task_state: TaskState, evidence: List[Evidence]) -> str:
        """Generate a human-readable summary of the judgment."""
        if task_state.status == TaskStatus.COMPLETED and task_state.all_tests_passed:
            return f"Task '{task_state.description}' completed successfully with all tests passing"
        elif task_state.status == TaskStatus.FAILED:
            return f"Task '{task_state.description}' failed: {task_state.error_message or 'Unknown error'}"
        elif task_state.status == TaskStatus.CANCELLED:
            return f"Task '{task_state.description}' was cancelled"
        else:
            # In progress or pending
            test_info = ""
            test_evidence = [e for e in evidence if e.type == "test_results"]
            if test_evidence:
                test_data = test_evidence[0].value
                test_info = f", {test_data['passed']}/{test_data['total']} tests passing"
            
            return f"Task '{task_state.description}' is {task_state.status.value}{test_info}"
    
    def _extract_lessons_learned(self, task_state: TaskState, 
                               evidence: List[Evidence]) -> List[str]:
        """Extract lessons learned from the task execution."""
        lessons = []
        
        # Learn from test results
        failed_tests = [t for t in task_state.test_results if not t.passed]
        if failed_tests:
            lessons.append(f"Encountered {len(failed_tests)} failing tests: {', '.join(t.test_name for t in failed_tests[:3])}{'...' if len(failed_tests) > 3 else ''}")
        
        # Learn from errors
        if task_state.error_message:
            lessons.append(f"Encountered error: {task_state.error_message[:100]}{'...' if len(task_state.error_message) > 100 else ''}")
        
        # Learn from patches
        if task_state.patches_applied:
            lessons.append(f"Applied {len(task_state.patches_applied)} patch(es) to fix issues")
        
        # Learn from duration
        if task_state.started_at and task_state.completed_at:
            import datetime
            duration = (task_state.completed_at - task_state.started_at).total_seconds()
            if duration > 60:
                lessons.append(f"Task took {duration:.0f} seconds - consider optimization for future similar tasks")
        
        return lessons
    
    def _generate_recommendations(self, criteria_results: List[Dict[str, Any]], 
                                task_state: TaskState, evidence: List[Evidence]) -> List[str]:
        """Generate recommendations for future similar tasks."""
        recommendations = []
        
        # Based on test failures
        failed_tests = [t for t in task_state.test_results if not t.passed]
        if failed_tests:
            recommendations.append("Investigate root causes of failing tests before attempting similar repairs")
            recommendations.append("Consider creating regression tests to prevent similar issues")
        
        # Based on errors
        if task_state.error_message:
            recommendations.append("Add error handling to prevent similar crashes in future")
            recommendations.append("Consider defensive programming practices for this code area")
        
        # Based on patches
        if len(task_state.patches_applied) > 3:
            recommendations.append("Large number of patches suggests deeper architectural issues may need addressing")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Task completed successfully - consider applying similar approach to related issues")
        
        return recommendations
    
    def get_judgment_history(self) -> List[Dict[str, Any]]:
        """Get history of all judgments made."""
        return self.judgments.copy()
    
    def clear_judgment_history(self):
        """Clear judgment history."""
        self.judgments.clear()