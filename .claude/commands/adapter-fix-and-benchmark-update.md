---
name: adapter-fix-and-benchmark-update
description: Workflow command scaffold for adapter-fix-and-benchmark-update in MLX-RFSN-main.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /adapter-fix-and-benchmark-update

Use this workflow when working on **adapter-fix-and-benchmark-update** in `MLX-RFSN-main`.

## Goal

Fixes or enhances candidate/adapters for model evaluation, then updates benchmark scripts and results.

## Common Files

- `rfsn_v11/candidates/*.py`
- `benchmarks/kv_shootout.py`
- `artifacts/bench/shootout/results.*`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit or add files in rfsn_v11/candidates/ (e.g., *_adapter.py, base.py)
- Update benchmarks/kv_shootout.py to reflect adapter changes or new evaluation logic
- Regenerate benchmark result artifacts (artifacts/bench/shootout/results.*)
- Commit all changes together

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.