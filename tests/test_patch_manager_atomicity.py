from __future__ import annotations

from agent_core.orchestrator import Orchestrator
from agent_core.patch_manager import PatchManager
from agent_core.schemas import PatchPlan
from agent_core.tool_runner import ToolRunner


def test_patch_manager_rolls_back_on_mid_plan_failure(tmp_path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("alpha\n", encoding="utf-8")
    file_b.write_text("beta\n", encoding="utf-8")

    orchestrator = Orchestrator()
    runner = ToolRunner(orchestrator, workspace_root=tmp_path)
    manager = PatchManager(orchestrator, runner)

    plan = PatchPlan(description="atomic rollback")
    plan.add_file_change(
        file_path=file_a,
        old_content="alpha\n",
        new_content="alpha-updated\n",
        change_type="modified",
    )
    plan.add_file_change(
        file_path=file_b,
        old_content="wrong-old-content\n",
        new_content="beta-updated\n",
        change_type="modified",
    )

    result = manager.apply_patch_plan(plan)

    assert result.success is False
    assert result.error_message is not None
    assert file_a.read_text(encoding="utf-8") == "alpha\n"
    assert file_b.read_text(encoding="utf-8") == "beta\n"
