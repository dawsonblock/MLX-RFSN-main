"""
Basic integration test for agent_core components.
This demonstrates how the various components work together.
"""

from __future__ import annotations

from agent_core.orchestrator import Orchestrator
from agent_core.planner import Planner
from agent_core.solver import Solver
from agent_core.critic import Critic
from agent_core.judge import Judge
from agent_core.tool_runner import ToolRunner
from agent_core.patch_manager import PatchManager
from agent_core.report_generator import ReportGenerator
from agent_core.finalizer import Finalizer
from agent_core.schemas import TaskState, TaskStatus, Priority, TestResult, TestResultStatus

def test_agent_core_integration():
    """Test that all agent_core components can work together."""
    print("Testing Agent Core Integration...")
    
    # Initialize core components
    orchestrator = Orchestrator()
    planner = Planner(orchestrator)
    solver = Solver(orchestrator)
    critic = Critic(orchestrator)
    judge = Judge(orchestrator)
    tool_runner = ToolRunner(None)  # No orchestrator needed for basic tool running
    patch_manager = PatchManager(orchestrator, tool_runner)
    report_generator = ReportGenerator(orchestrator)
    finalizer = Finalizer(orchestrator)
    
    print("✓ All components initialized successfully")
    
    # Create a test task
    task = orchestrator.create_task(
        description="Test task: Verify agent core components work together",
        priority=Priority.HIGH
    )
    print(f"✓ Created task: {task.task_id}")
    
    # Start the task
    orchestrator.start_task(task.task_id)
    print(f"✓ Started task: {task.task_id}")
    
    # Create a plan for the task
    plan = planner.create_plan("Verify agent core component integration")
    print(f"✓ Created plan: {plan.plan_id}")
    
    # Add some steps to the plan
    plan.add_step(
        description="Verify orchestrator functionality",
        action=lambda: orchestrator.create_task("subtask", Priority.LOW)
    )
    plan.add_step(
        description="Verify planner functionality", 
        action=lambda: planner.create_plan("test plan")
    )
    
    # Simulate completing the task
    # In a real scenario, the solver would propose patches,
    # the critic would review them, the judge would evaluate results,
    # and the finalizer would create the final output
    
    # For this test, we'll simulate successful completion
    test_result = TestResult(
        test_name="integration_test",
        status=TestResultStatus.PASSED,
        duration_ms=100.0,
        message="All agent core components working together"
    )
    
    orchestrator.add_test_result(task.task_id, test_result)
    orchestrator.complete_task(task.task_id, success=True)
    print(f"✓ Completed task: {task.task_id}")
    
    # Generate final output
    from agent_core.schemas import AgentVerdict
    verdict = AgentVerdict.from_task_state(task)
    final_output = finalizer.finalize_task(task, verdict)
    
    print(f"✓ Generated final output: {final_output.output_id}")
    print(f"✓ Final output title: {final_output.title}")
    
    # Verify the output makes sense
    assert final_output.title.startswith("✅ Task Completed Successfully")
    assert "Verify agent core components work together" in final_output.summary
    assert verdict.success is True  # Note: FinalOutput doesn't have success field, but verdict does
    
    print("✓ All integration tests passed!")

if __name__ == "__main__":
    import sys
    try:
        test_agent_core_integration()
        print("\n🎉 All Agent Core integration tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)