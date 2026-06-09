"""
Solver that proposes patches and fixes for identified problems.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json

from .schemas import PatchPlan, FileChange, TestResult
from .orchestrator import Orchestrator
from .planner import Plan, Step

logger = __import__('logging').getLogger(__name__)


@dataclass
class Problem:
    """Represents a problem that needs to be solved."""
    problem_id: str = field(default_factory=lambda: str(__import__('uuid').uuid4()))
    description: str = ""
    severity: str = "medium"  # low, medium, high, critical
    location: Optional[str] = None  # file:line or similar
    test_failures: List[str] = field(default_factory=list)  # Related test names
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggested_solution: Optional[str] = None
    
    def __post_init__(self):
        if not self.description:
            raise ValueError("Problem description is required")


class Solver:
    """
    Solver that analyzes problems and proposes solutions.
    
    The solver takes test failures, error messages, and other evidence
    and generates concrete patch proposals to fix the issues.
    """
    
    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator
        self.solved_problems: List[Problem] = []
        self.unsolved_problems: List[Problem] = []
    
    def analyze_test_failures(self, test_results: List[TestResult]) -> List[Problem]:
        """
        Analyze test results to identify problems.
        
        Args:
            test_results: List of test results from test execution
            
        Returns:
            List of identified problems
        """
        problems = []
        
        for test_result in test_results:
            if not test_result.passed:
                problem = Problem(
                    description=f"Test failed: {test_result.test_name}",
                    severity="medium",
                    location=None,  # Would need to parse traceback for file:line
                    test_failures=[test_result.test_name],
                    evidence={
                        "test_result": test_result.__dict__
                    }
                )
                
                # Try to extract more specific info from traceback
                if test_result.traceback:
                    problem.evidence["traceback"] = test_result.traceback
                    # In a real implementation, we'd parse the traceback to get file:line
                
                problems.append(problem)
        
        return problems
    
    def solve_zero_scale_quantization(self, problems: List[Problem]) -> PatchPlan:
        """
        Generate a fix for zero-scale quantization bug.
        
        This is a specific solver for one of the critical RFSN issues.
        """
        plan = PatchPlan(
            description="Fix zero-scale quantization bug in KV manager",
            rationale="Prevent division by zero when all values in a group are zero"
        )
        
        # The fix is to change the scale calculation in _quantize method
        # From: scales = abs_max / qmax
        # To: raw_scale = abs_max / float(qmax); scales = maximum(raw_scale, 1e-8)
        
        plan.add_file_change(
            file_path="rfsn_v10/kv_manager.py",
            old_string="""        abs_max = mx.max(mx.abs(x), axis=-1)
        scales = abs_max / mx.maximum(mx.array(float(qmax), dtype=x.dtype), mx.array(1e-8, dtype=x.dtype))""",
            new_string="""        abs_max = mx.max(mx.abs(x), axis=-1)
        raw_scale = abs_max / float(qmax)
        scales = mx.maximum(raw_scale, mx.array(1e-8, dtype=x.dtype))""",
            change_type="modified"
        )
        
        plan.estimated_impact = "Fixes division by zero that produces NaN for all-zero groups"
        plan.risks = ["Change is minimal and targeted, low risk"]
        
        return plan
    
    def solve_fused_kernel_naming(self, problems: List[Problem]) -> PatchPlan:
        """
        Generate a fix for the falsely named 'fused' kernel.
        
        Renames the method to reflect it's sequential, not fused.
        """
        plan = PatchPlan(
            description="Rename fused kernel method to honest name",
            rationale="Method was named '_fused_packed_dequant_wht' but is sequential MLX operations"
        )
        
        # Rename method and update all calls
        plan.add_file_change(
            file_path="rfsn_v10/kv_manager.py",
            old_string="    def _fused_packed_dequant_wht(",
            new_string="    def _reconstruct_packed_dequant_wht(",
            change_type="modified"
        )
        
        # This would also need to update docstring and all calls - simplified for example
        plan.add_file_change(
            file_path="rfsn_v10/kv_manager.py",
            old_string="fused packed-dequant-WHT kernel",
            new_string="packed-dequant-WHT reconstruction path",
            change_type="modified"
        )
        
        plan.estimated_impact = "Corrects misleading claims about fused Metal kernel"
        plan.risks = ["Requires updating all call sites - medium risk if missed"]
        
        return plan
    
    def solve_problems(self, problems: List[Problem]) -> List[PatchPlan]:
        """
        Solve a list of problems by generating patch plans.
        
        Args:
            problems: List of problems to solve
            
        Returns:
            List of patch plans to fix the problems
        """
        patch_plans = []
        
        for problem in problems:
            # Dispatch to specific solvers based on problem characteristics
            if "zero-scale" in problem.description.lower() or "quantize" in problem.description.lower():
                patch_plan = self.solve_zero_scale_quantization([problem])
                patch_plans.append(patch_plan)
            elif "fused" in problem.description.lower() and "kernel" in problem.description.lower():
                patch_plan = self.solve_fused_kernel_naming([problem])
                patch_plans.append(patch_plan)
            else:
                # Generic solver - create a placeholder plan
                plan = PatchPlan(
                    description=f"Address problem: {problem.description}",
                    rationale=f"Problem identified: {problem.description}"
                )
                if problem.suggested_solution:
                    plan.rationale += f" Suggested approach: {problem.suggested_solution}"
                patch_plans.append(plan)
        
        return patch_plans