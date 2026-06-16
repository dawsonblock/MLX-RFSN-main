#!/usr/bin/env bash
# Release gate script for MLX-RFSN Fusion
# Phase 2: Thin wrapper around the canonical Python gate.
#
# This script ONLY:
#   1. Validates the Python version.
#   2. Ensures the execution environment is ready.
#   3. Invokes scripts/release_gate.py (the canonical implementation).
#   4. Returns the Python gate's exit code verbatim.
#
# All actual checks (compilation, imports, tests, integrity, wheel) live
# in release_gate.py so there is a single source of truth.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== MLX-RFSN Release Gate ==="

# 1. Validate Python version
PYTHON="${PYTHON:-python3}"
PY_MAJOR=$($PYTHON -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$($PYTHON -c 'import sys; print(sys.version_info.minor)')

if ! [[ "$PY_MAJOR" -eq 3 && ( "$PY_MINOR" -eq 11 || "$PY_MINOR" -eq 12 ) ]]; then
    echo "ERROR: Python $PY_MAJOR.$PY_MINOR is not supported. Use 3.11 or 3.12."
    exit 1
fi
echo "  Python $PY_MAJOR.$PY_MINOR OK"

# 2. Ensure repo root is on PYTHONPATH
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# 3. Forward to the canonical Python gate
#    Pass through any arguments given to this shell script.
echo "  Delegating to canonical gate: scripts/release_gate.py $*"
$PYTHON scripts/release_gate.py "$@"

# Return the Python gate's exit code exactly (set -e already handles this,
# but we make it explicit for clarity).
GATE_RC=$?
if [ "$GATE_RC" -eq 0 ]; then
    echo "=== Release Gate Passed ==="
else
    echo "=== Release Gate Failed (exit $GATE_RC) ==="
fi
exit "$GATE_RC"
