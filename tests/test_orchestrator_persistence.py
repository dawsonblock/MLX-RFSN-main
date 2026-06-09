from __future__ import annotations

from agent_core.orchestrator import Orchestrator
from agent_core.schemas import TestResult, TestResultStatus


def test_orchestrator_save_and_load_state(tmp_path):
    state_path = tmp_path / "state" / "orchestrator.pkl"

    orch = Orchestrator(auto_save_path=state_path)
    task = orch.create_task("persist me")
    assert state_path.exists()

    assert orch.start_task(task.task_id)
    orch.add_test_result(
        task.task_id,
        TestResult(
            test_name="sample",
            status=TestResultStatus.PASSED,
            duration_ms=1.0,
        ),
    )

    reloaded = Orchestrator()
    reloaded.load_state(state_path)

    loaded_task = reloaded.get_task(task.task_id)
    assert loaded_task is not None
    assert loaded_task.description == "persist me"
    assert loaded_task.status.value == "in_progress"
    assert len(loaded_task.test_results) == 1
    assert reloaded.get_root_tasks()[0].task_id == task.task_id


def test_orchestrator_reset_updates_autosave(tmp_path):
    state_path = tmp_path / "orchestrator.pkl"
    orch = Orchestrator(auto_save_path=state_path)
    task = orch.create_task("to reset")
    assert orch.get_task(task.task_id) is not None

    orch.reset()

    reloaded = Orchestrator()
    reloaded.load_state(state_path)
    assert reloaded.get_task(task.task_id) is None
    assert reloaded.get_root_tasks() == []
