# Artifact History

This directory contains historical benchmark artifacts from previous runs.

## Purpose

Artifacts are moved here when:
- They are from an older benchmark run that is no longer the current reference
- They are from a deprecated candidate or configuration
- They are from a failed or incomplete validation run
- They are being archived for historical reference

## Directory Structure

- `debug/`: Debug artifacts from troubleshooting runs
- `kernel/`: Kernel-related artifacts (Metal kernel tests, etc.)
- `legacy/`: Artifacts from legacy/deprecated candidates
- `memory/`: Memory report artifacts
- `polar_baseline/`: Polar baseline artifacts
- `promotion/`: Promotion report artifacts
- `turbo_polar/`: TurboQuant+Polar artifacts

## Migration Status

Fix #12: Complete artifact-history migration

The artifact history migration is structured as follows:
1. Current artifacts live in `artifacts/bench/current/`
2. Historical artifacts are organized in `artifacts/bench/history/` by category
3. When a new benchmark run produces artifacts, the previous current artifacts are moved to history

## Artifact Lifecycle

1. **New run**: Artifacts are written to `artifacts/bench/current/`
2. **Validation**: If the run passes validation, artifacts become the current reference
3. **Archival**: When a new run succeeds, previous artifacts are moved to `artifacts/bench/history/`
4. **Cleanup**: Very old artifacts (older than N revisions) may be deleted to save space

## Fix #12 Completion

The artifact-history migration is considered complete when:
- All historical artifacts are properly categorized in history/ subdirectories
- No stale artifacts remain in current/ that should be in history/
- The migration process is documented and repeatable
- A cleanup policy is defined for very old artifacts

Current status: History directory structure exists with proper categorization.
