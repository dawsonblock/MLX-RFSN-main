---
name: server-feature-or-hardening
description: Workflow command scaffold for server-feature-or-hardening in MLX-RFSN-main.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /server-feature-or-hardening

Use this workflow when working on **server-feature-or-hardening** in `MLX-RFSN-main`.

## Goal

Implements new server features, configuration options, or hardening (timeouts, API keys, request limits), with corresponding documentation and tests.

## Common Files

- `rfsn_v10/server/app.py`
- `rfsn_v10/server/cli.py`
- `rfsn_v10/config.py`
- `docs/*.md`
- `tests/server/*.py`
- `tests/runtime/*.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit rfsn_v10/server/app.py and/or rfsn_v10/server/cli.py to add/modify features
- Update rfsn_v10/config.py for new config options
- Update or add documentation in docs/ (e.g., FEATURE_FLAGS.md, RUN_SERVER.md)
- Add or update tests in tests/server/ and/or tests/runtime/
- Update scripts/release_gate.py if release checks are affected

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.