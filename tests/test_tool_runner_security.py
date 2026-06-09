from __future__ import annotations

from agent_core.orchestrator import Orchestrator
from agent_core.tool_runner import ToolRunner


def test_tool_runner_rejects_cwd_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    runner = ToolRunner(Orchestrator(), workspace_root=workspace)
    outside = tmp_path / "outside"
    outside.mkdir()

    result = runner.run_command("python3 -c \"print('ok')\"", cwd=outside)

    assert result.return_code == -4
    assert "inside workspace_root" in result.stderr


def test_tool_runner_prefix_permission_matching(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    runner = ToolRunner(Orchestrator(), workspace_root=workspace)

    allowed = runner.run_command("python3 -c \"print('ok')\"", cwd=workspace)
    denied = runner.run_command("jupyter_python --version", cwd=workspace)

    assert allowed.return_code == 0
    assert denied.return_code == -1
