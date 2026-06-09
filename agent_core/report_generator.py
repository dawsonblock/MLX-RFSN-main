"""
Report generation for agent tasks and workflows.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import textwrap

from .schemas import TaskState, TestResult, PatchPlan, AgentVerdict
from .orchestrator import Orchestrator

logger = __import__('logging').getLogger(__name__)


@dataclass
class ReportSection:
    """A section of a report."""
    title: str
    content: str
    level: int = 1  # Heading level (1=H1, 2=H2, etc.)
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        prefix = "#" * self.level
        return f"{prefix} {self.title}\n\n{self.content}\n"


@dataclass
class TaskReport:
    """A comprehensive report of a task execution."""
    task_id: str
    title: str
    generated_at: datetime = field(default_factory=datetime.now)
    sections: List[ReportSection] = field(default_factory=list)
    
    def add_section(self, title: str, content: str, level: int = 1):
        """Add a section to the report."""
        self.sections.append(ReportSection(title=title, content=content, level=level))
    
    def to_markdown(self) -> str:
        """Convert the entire report to markdown."""
        lines = [
            f"# {self.title}",
            f"*Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*",
            ""
        ]
        
        for section in self.sections:
            lines.append(section.to_markdown())
        
        return "\n".join(lines)
    
    def save_to_file(self, file_path: Path | str):
        """Save the report to a file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        logger.info(f"Report saved to {path}")


class ReportGenerator:
    """
    Generates reports from task executions and agent workflows.
    
    The report generator creates human-readable reports detailing:
    - What was attempted
    - What was found
    - What was done
    - What the results were
    - What was learned
    """
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.reports: Dict[str, TaskReport] = {}
    
    def generate_task_report(self, task_state: TaskState, 
                           verdict: Optional[AgentVerdict] = None) -> TaskReport:
        """
        Generate a comprehensive report for a task.
        
        Args:
            task_state: The task state to report on
            verdict: Optional verdict (will be generated if not provided)
            
        Returns:
            TaskReport detailing the task execution
        """
        if verdict is None:
            # We'd need a judge to generate this, but for now create a basic one
            from .judge import Judge
            judge = Judge(self.orchestrator)
            verdict = judge.judge_task_completion(task_state)
        
        report = TaskReport(
            task_id=task_state.task_id,
            title=f"Task Report: {task_state.description}"
        )
        
        # Overview section
        overview_lines = [
            f"**Task ID:** {task_state.task_id}",
            f"**Description:** {task_state.description}",
            f"**Status:** {task_state.status.value}",
            f"**Priority:** {task_state.priority.value}",
            f"**Created:** {task_state.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
        ]
        
        if task_state.started_at:
            overview_lines.append(f"**Started:** {task_state.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if task_state.completed_at:
            overview_lines.append(f"**Completed:** {task_state.completed_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if task_state.started_at and task_state.completed_at:
            import datetime
            duration = (task_state.completed_at - task_state.started_at).total_seconds()
            overview_lines.append(f"**Duration:** {duration:.2f} seconds")
        
        report.add_section("Overview", "\n".join(overview_lines), level=2)
        
        # Test results section
        if task_state.test_results:
            test_lines = []
            passed = sum(1 for t in task_state.test_results if t.passed)
            total = len(task_state.test_results)
            test_lines.append(f"**Test Results:** {passed}/{total} passed")
            test_lines.append("")
            
            if task_state.test_results:
                test_lines.append("| Test Name | Status | Duration (s) | Message |")
                test_lines.append("|-----------|--------|--------------|---------|")
                for test_result in task_state.test_results:
                    status_emoji = "✅" if test_result.passed else "❌"
                    message = (test_result.message or "").replace("|", "\\|")[:50]
                    test_lines.append(f"| {test_result.test_name} | {status_emoji} {test_result.status.value} | {test_result.duration_ms/1000.0:.3f} | {message} |")
            
            report.add_section("Test Results", "\n".join(test_lines), level=2)
        
        # Patches applied section
        if task_state.patches_applied:
            patch_lines = []
            patch_lines.append(f"**Patches Applied:** {len(task_state.patches_applied)}")
            patch_lines.append("")
            
            for i, patch in enumerate(task_state.patches_applied, 1):
                patch_lines.append(f"**Patch {i}:**")
                patch_lines.append(f"- Description: {patch.description}")
                patch_lines.append(f"- Rationale: {patch.rationale}")
                if patch.file_changes:
                    patch_lines.append(f"- Files Changed: {len(patch.file_changes)}")
                    for fc in patch.file_changes:
                        patch_lines.append(f"  - {fc.file_path}: {fc.change_type}")
                patch_lines.append("")
            
            report.add_section("Patches Applied", "\n".join(patch_lines), level=2)
        
        # Errors section
        if task_state.error_message:
            report.add_section("Error", f"```\n{task_state.error_message}\n```", level=2)
        
        # Verdict section
        verdict_lines = [
            f"**Overall Success:** {'✅ YES' if verdict.success else '❌ NO'}",
            f"**Summary:** {verdict.summary}",
        ]
        
        if verdict.lessons_learned:
            verdict_lines.append("")
            verdict_lines.append("**Lessons Learned:**")
            for lesson in verdict.lessons_learned:
                verdict_lines.append(f"- {lesson}")
        
        if verdict.recommendations:
            verdict_lines.append("")
            verdict_lines.append("**Recommendations:**")
            for rec in verdict.recommendations:
                verdict_lines.append(f"- {rec}")
        
        report.add_section("Verdict", "\n".join(verdict_lines), level=2)
        
        # Artifacts section
        if task_state.artifacts:
            artifact_lines = []
            artifact_lines.append("**Artifacts:**")
            for key, value in task_state.artifacts.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    artifact_lines.append(f"- {key}: {value}")
                else:
                    artifact_lines.append(f"- {key}: {type(value).__name__} (omitted for brevity)")
            
            report.add_section("Artifacts", "\n".join(artifact_lines), level=2)
        
        # Store report
        self.reports[task_state.task_id] = report
        logger.info(f"Generated report for task {task_state.task_id}")
        
        return report
    
    def generate_summary_report(self, task_ids: Optional[List[str]] = None) -> TaskReport:
        """
        Generate a summary report of multiple tasks.
        
        Args:
            task_ids: List of task IDs to include (None for all tracked tasks)
            
        Returns:
            Summary report
        """
        if task_ids is None:
            task_ids = list(self.reports.keys())
        
        tasks = [self.reports[tid] for tid in task_ids if tid in self.reports]
        
        if not tasks:
            return TaskReport(
                task_id="summary",
                title="No tasks to report"
            )
        
        report = TaskReport(
            task_id="summary",
            title="Task Summary Report"
        )
        
        # Summary statistics
        total_tasks = len(tasks)
        successful_tasks = sum(1 for t in tasks if t.sections and any("✅ YES" in s.content for s in t.sections if s.title == "Verdict"))
        failed_tasks = total_tasks - successful_tasks
        
        stats_lines = [
            f"**Total Tasks:** {total_tasks}",
            f"**Successful:** {successful_tasks}",
            f"**Failed:** {failed_tasks}",
            f"**Success Rate:** {successful_tasks/total_tasks:.1%}" if total_tasks > 0 else "**Success Rate:** 0%",
        ]
        
        report.add_section("Summary Statistics", "\n".join(stats_lines), level=2)
        
        # Recent tasks
        recent_lines = []
        recent_lines.append("**Recent Tasks:**")
        recent_lines.append("")
        recent_lines.append("| Task ID | Description | Status | Duration |")
        recent_lines.append("|---------|-------------|--------|----------|")
        
        # Sort by generation time, newest first
        sorted_tasks = sorted(tasks, key=lambda t: t.generated_at, reverse=True)
        for task_report in sorted_tasks[:10]:  # Show top 10
            # Extract info from the report
            task_id = task_report.task_id
            title_sections = [s for s in task_report.sections if s.title == "Overview"]
            description = "Unknown"
            if title_sections:
                # Parse description from overview
                for line in title_sections[0].content.split('\n'):
                    if line.startswith("**Description:**"):
                        description = line.replace("**Description:**", "").strip()
                        break
            
            status_sections = [s for s in task_report.sections if s.title == "Verdict"]
            status = "Unknown"
            if status_sections:
                for line in status_sections[0].content.split('\n'):
                    if line.startswith("**Overall Success:**"):
                        status = "Success" if "✅ YES" in line else "Failed"
                        break
            
            duration = "Unknown"
            overview_sections = [s for s in task_report.sections if s.title == "Overview"]
            if overview_sections:
                for line in overview_sections[0].content.split('\n'):
                    if line.startswith("**Duration:**"):
                        duration = line.replace("**Duration:**", "").strip()
                        break
            
            recent_lines.append(f"| {task_id} | {description} | {status} | {duration} |")
        
        report.add_section("Recent Tasks", "\n".join(recent_lines), level=2)
        
        return report
    
    def get_report(self, task_id: str) -> Optional[TaskReport]:
        """Get a report by task ID."""
        return self.reports.get(task_id)
    
    def save_report(self, task_id: str, file_path: Path | str):
        """Save a report to file."""
        report = self.get_report(task_id)
        if report:
            report.save_to_file(file_path)
        else:
            logger.error(f"No report found for task ID: {task_id}")
    
    def save_all_reports(self, directory: Path | str):
        """Save all reports to files in a directory."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)
        
        for task_id, report in self.reports.items():
            file_path = dir_path / f"task_{task_id}.md"
            report.save_to_file(file_path)
        
        logger.info(f"Saved {len(self.reports)} reports to {dir_path}")
    
    def clear_reports(self):
        """Clear all stored reports."""
        self.reports.clear()