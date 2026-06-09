"""
Finalizer that creates user-facing results from agent verdicts.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .schemas import AgentVerdict, TaskState
from .orchestrator import Orchestrator
from .report_generator import ReportGenerator

logger = __import__('logging').getLogger(__name__)


@dataclass
class FinalOutput:
    """The final output presented to the user."""
    output_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    title: str = ""
    summary: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    related_artifacts: List[Dict[str, Any]] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    format: str = "markdown"  # markdown, json, html, text
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# {self.title}",
            "",
            self.summary,
            "",
            "## Details",
        ]
        
        for key, value in self.details.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                lines.append(f"- **{key}**: {value}")
            else:
                lines.append(f"- **{key}**: {type(value).__name__} (omitted for brevity)")
        
        lines.extend([
            "",
            "## Recommendations",
        ])
        
        if self.recommendations:
            for rec in self.recommendations:
                if isinstance(rec, dict):
                    lines.append(f"- {rec.get('description', str(rec))}")
                else:
                    lines.append(f"- {rec}")
        else:
            lines.append("- No specific recommendations")
        
        lines.extend([
            "",
            "## Next Steps",
        ])
        
        if self.next_steps:
            for step in self.next_steps:
                lines.append(f"- {step}")
        else:
            lines.append("- No specific next steps defined")
        
        lines.extend([
            "",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*",
            f"*Output ID: {self.output_id}*"
        ])
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Convert to JSON format."""
        import json
        return json.dumps({
            "output_id": self.output_id,
            "title": self.title,
            "summary": self.summary,
            "details": self.details,
            "recommendations": self.recommendations,
            "next_steps": self.next_steps,
            "related_artifacts": self.related_artifacts,
            "generated_at": self.generated_at.isoformat(),
            "format": self.format
        }, indent=2)


class Finalizer:
    """
    Finalizer that creates user-facing results from agent verdicts.
    
    The finalizer takes agent verdicts and task states and creates
    polished, user-friendly outputs that summarize what was accomplished,
    what was learned, and what should happen next.
    """
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.report_generator = ReportGenerator(orchestrator)
        self.final_outputs: List[FinalOutput] = []
        logger = __import__('logging').getLogger(__name__)
        self.logger = logger
    
    def finalize_task(self, task_state: TaskState, 
                     verdict: AgentVerdict) -> FinalOutput:
        """
        Create a final output from a task state and verdict.
        
        Args:
            task_state: The task state that was executed
            verdict: The verdict from judging the task
            
        Returns:
            FinalOutput suitable for presentation to the user
        """
        # Determine overall status and create appropriate title
        if verdict.success:
            title = f"✅ Task Completed Successfully: {task_state.description}"
        else:
            title = f"❌ Task Failed: {task_state.description}"
        
        # Create summary
        if verdict.success:
            summary = f"The task '{task_state.description}' has been completed successfully. "
            if task_state.test_results:
                passed = sum(1 for t in task_state.test_results if t.passed)
                total = len(task_state.test_results)
                summary += f"All {total} tests passed ({passed}/{total}). "
            if task_state.patches_applied:
                summary += f"Applied {len(task_state.patches_applied)} patch(es) to fix identified issues. "
        else:
            summary = f"The task '{task_state.description}' failed to complete successfully. "
            if task_state.error_message:
                summary += f"Error: {task_state.error_message} "
            if task_state.test_results:
                failed = sum(1 for t in task_state.test_results if not t.passed)
                total = len(task_state.test_results)
                summary += f"{failed}/{total} tests failed. "
        
        # Build details
        details = {
            "Task ID": task_state.task_id,
            "Description": task_state.description,
            "Status": task_state.status.value,
            "Priority": task_state.priority.value,
            "Duration (seconds)": self._calculate_duration(task_state),
            "Tests Run": len(task_state.test_results),
            "Tests Passed": sum(1 for t in task_state.test_results if t.passed),
            "Patches Applied": len(task_state.patches_applied),
            "Verdict Success": verdict.success
        }
        
        # Add timestamps if available
        if task_state.created_at:
            details["Created At"] = task_state.created_at.strftime('%Y-%m-%d %H:%M:%S')
        if task_state.started_at:
            details["Started At"] = task_state.started_at.strftime('%Y-%m-%d %H:%M:%S')
        if task_state.completed_at:
            details["Completed At"] = task_state.completed_at.strftime('%Y-%m-%d %H:%M:%S')
        
        # Build recommendations from verdict and task state
        recommendations = []
        
        # Add verdict recommendations
        for rec in verdict.recommendations:
            recommendations.append({
                "type": "verdict_recommendation",
                "description": rec,
                "priority": "medium"
            })
        
        # Add task-state based recommendations
        if not verdict.success:
            if task_state.error_message:
                recommendations.append({
                    "type": "error_resolution",
                    "description": f"Address the error: {task_state.error_message}",
                    "priority": "high"
                })
            
            failed_tests = [t for t in task_state.test_results if not t.passed]
            if failed_tests:
                recommendations.append({
                    "type": "test_improvement",
                    "description": f"Implement fixes for {len(failed_tests)} failing test(s)",
                    "priority": "high"
                })
        
        # Add general improvement recommendations
        if len(task_state.patches_applied) > 3:
            recommendations.append({
                "type": "code_quality",
                "description": "Consider refactoring - multiple patches suggest complex changes",
                "priority": "medium"
            })
        
        # Build next steps
        next_steps = []
        
        if verdict.success:
            next_steps.append("Monitor the solution in production or testing environment")
            next_steps.append("Consider adding regression tests to prevent similar issues")
            if task_state.patches_applied:
                next_steps.append("Document the changes made for future reference")
        else:
            next_steps.append("Review the error messages and test failures")
            next_steps.append("Consider alternative approaches to solve the problem")
            next_steps.append("Consult with team members or documentation for guidance")
        
        # Add artifacts if any
        related_artifacts = []
        if task_state.artifacts:
            for key, value in task_state.artifacts.items():
                related_artifacts.append({
                    "name": key,
                    "type": type(value).__name__,
                    "description": f"Artifact from task execution: {key}"
                })
        
        # Add report as an artifact
        related_artifacts.append({
            "name": "detailed_report",
            "type": "report",
            "description": "Full detailed report of the task execution"
        })
        
        # Create the final output
        final_output = FinalOutput(
            title=title,
            summary=summary.strip(),
            details=details,
            recommendations=recommendations,
            next_steps=next_steps,
            related_artifacts=related_artifacts,
            format="markdown"
        )
        
        # Store for history
        self.final_outputs.append(final_output)
        logger.info(f"Finalized task {task_state.task_id} with output {final_output.output_id}")
        
        return final_output
    
    def _calculate_duration(self, task_state: TaskState) -> Optional[float]:
        """Calculate task duration in seconds."""
        if task_state.started_at and task_state.completed_at:
            return (task_state.completed_at - task_state.started_at).total_seconds()
        elif task_state.started_at:
            # Still in progress
            return (datetime.now() - task_state.started_at).total_seconds()
        return None
    
    def get_final_outputs(self) -> List[FinalOutput]:
        """Get all final outputs created."""
        return self.final_outputs.copy()
    
    def get_latest_final_output(self) -> Optional[FinalOutput]:
        """Get the most recent final output."""
        if self.final_outputs:
            return self.final_outputs[-1]
        return None
    
    def clear_final_outputs(self):
        """Clear all stored final outputs."""
        self.final_outputs.clear()


def finalize_task(task_state: TaskState, verdict: AgentVerdict) -> FinalOutput:
    """Convenience function to finalize a task."""
    finalizer = Finalizer(Orchestrator())
    return finalizer.finalize_task(task_state, verdict)