#!/usr/bin/env bash
# Release gate script for MLX-RFSN Fusion
# Reads release identity from release.toml
# Runs on any platform (Linux, macOS, Windows with bash)
set -euo pipefail

echo "=== MLX-RFSN Release Gate ==="

# Load release identity
if [ -f "release.toml" ]; then
    RELEASE_ID=$(grep "^release_id" release.toml | cut -d'"' -f2)
    DISPLAY_NAME=$(grep "^display_name" release.toml | cut -d'"' -f2)
    echo "Release: $DISPLAY_NAME ($RELEASE_ID)"
else
    echo "Warning: release.toml not found, using defaults"
    RELEASE_ID="unknown"
    DISPLAY_NAME="MLX-RFSN Fusion"
fi

# Fix #13: Sync version from release.toml to pyproject.toml and README.md
echo "[1/10] Syncing version from release.toml..."
python scripts/sync_version.py
echo "  Version sync completed."

# 2. Compile check
echo "[2/10] Compile check..."
python -m compileall -q rfsn_v10 rfsn_v11 tests benchmarks scripts memory

# 3. Test collection (must not fail — catches import/shadowing bugs)
echo "[3/10] Test collection..."
PYTHONPATH=. pytest --collect-only -q tests rfsn_v11/tests rfsn_v10/kernels/tests

# 4. CPU tests (no MLX required)
echo "[4/10] CPU tests..."
PYTHONPATH=. RFSN_BACKEND=numpy RFSN_TELEMETRY_HMAC_KEY=test-secret \
  pytest -q tests -m "not mlx and not slow and not benchmark and not experimental and not integration and not db"

# 5. rfsn_v11 unit tests (skip MLX-dependent tests on non-MLX platforms)
echo "[5/10] rfsn_v11 tests..."
PYTHONPATH=. pytest -q rfsn_v11/tests -m "not mlx"

# 6. Benchmark tests
echo "[6/10] Benchmark tests..."
PYTHONPATH=. pytest -q tests/benchmarks

# 7. Quick shootout smoke (strict execution only — promotion is separate)
# P0 Fix: Use strict-execution, not legacy --strict, and do not require a
# real model so the gate can run on CPU-only hosts.
echo "[7/10] Quick shootout smoke..."
PYTHONPATH=. python benchmarks/kv_shootout.py --quick --strict-execution
echo "  Quick shootout completed."

# 8. Release integrity check (validate BEFORE archiving)
echo "[8/10] Release integrity check..."
PYTHONPATH=. python scripts/check_release_integrity.py

# 9. Archive artifacts only after validation passes
echo "[9/10] Archiving artifacts to history..."
python scripts/archive_artifacts.py
echo "  Artifact archiving completed."

# 10. Build check (hermetic — do not mutate environment)
echo "[10/10] Build check..."
python -m build --wheel
# Verify wheel can be imported without installing
latest_wh=$(ls -t dist/*.whl 2>/dev/null | head -1)
if [ -n "$latest_wh" ]; then
    # Use zipimport to test without installing
    python -c "import zipimport, sys; sys.path.insert(0, '$latest_wh'); import rfsn_v10, rfsn_v11; print('wheel import ok')"
else
    echo "  No wheel found in dist/"
    exit 1
fi

# Fix #14: Reject 0.0.0 wheels in release gate
if [[ "$latest_wh" == *"0.0.0"* ]]; then
    echo "  ERROR: Wheel version is 0.0.0, rejecting for release"
    exit 1
else
    echo "  Wheel version check passed"
fi

echo "=== Release Gate Passed ==="
