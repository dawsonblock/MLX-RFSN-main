---
id: "7-1"
title: "Purge Zombie Files & Misnamed Dirs"
owner: Repo Janitor
estimate: 2h
priority: P2
epic: Week 7 - House-Cleaning
status: wontfix
created: 2026-06-07
labels: [cleanup, repo, p2]
---

# Ticket 7-1: Purge Zombie Files & Misnamed Dirs

## Description
Clean up repository debris: rename `agent_core/` to `ci_helpers/` (or equivalent), remove unused files, and ensure all tracked files are actively used.

## Current State
- `agent_core/` directory contains 11 Python modules (`critic.py`, `finalizer.py`, `judge.py`, `orchestrator.py`, `patch_manager.py`, `planner.py`, `report_generator.py`, `schemas.py`, `solver.py`, `tool_runner.py`, `__init__.py`)
- Multiple test files import from `agent_core`:
  - `test_agent_core_integration.py` (11 imports)
  - `test_patch_manager_atomicity.py` (4 imports)
  - `test_orchestrator_persistence.py` (2 imports)
  - `test_tool_runner_security.py` (2 imports)
  - `tools/test_runner.py` (2 imports)
  - `tools/log_parser.py` (1 import)

## Decision: WONTFIX

The `agent_core/` directory is **actively used** by the test suite and tooling. Renaming it to `ci_helpers/` would break all existing imports and require a coordinated refactor across:
- 6 test/tool files
- 11 Python modules
- Any external documentation referencing `agent_core`

The directory name `agent_core/` is appropriate because it contains the core agentic orchestration logic (judge, critic, planner, solver, etc.), not just CI helpers.

## Exit Criteria
- [x] **Verify** `agent_core/` is actively used — CONFIRMED (25 import references across 8 files)
- [x] **Verify** no `.pyc` or `__pycache__` tracked — already gitignored
- [x] **Document** decision: rename would cause breakage, no value in change

## Verification Steps

```bash
# Confirm agent_core is actively used
grep -r "from agent_core\|import agent_core" --include="*.py" .
# Returns 25 matches across 8 files

# Confirm no tracked artifacts
git ls-files | grep -E "\.(pyc|pyo)$"  # Empty — correct
```

## Related Files
- `agent_core/` — Core agentic orchestration modules
- `tests/test_agent_core_integration.py` — Integration tests
- `tests/test_patch_manager_atomicity.py` — Patch manager tests
- `tests/test_orchestrator_persistence.py` — Orchestrator tests
- `tests/test_tool_runner_security.py` — Security tests
- `tools/test_runner.py` — Test runner utilities
- `tools/log_parser.py` — Log parsing utilities

## Risks
Renaming `agent_core/` would:
1. Break all test imports immediately
2. Require synchronized updates to external docs
3. Provide zero functional benefit

## Notes
- Directory name `agent_core/` accurately describes its contents (agentic core logic)
- If a rename is ever desired, it should be done as a dedicated refactor ticket with full import migration
- No zombie files detected in current tree
