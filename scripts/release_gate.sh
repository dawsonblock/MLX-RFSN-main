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

# 1. Compile check
echo "[1/8] Compile check..."
python -m compileall -q rfsn_v10 rfsn_v11 tests benchmarks scripts memory

# 2. Test collection (must not fail — catches import/shadowing bugs)
echo "[2/8] Test collection..."
PYTHONPATH=. pytest --collect-only -q tests rfsn_v11/tests

# 3. CPU tests (no MLX required)
echo "[3/8] CPU tests..."
PYTHONPATH=. RFSN_BACKEND=numpy RFSN_TELEMETRY_HMAC_KEY=test-secret \
  pytest -q tests -m "not mlx and not slow and not benchmark and not experimental and not integration and not db"

# 4. rfsn_v11 unit tests (skip MLX-dependent tests on non-MLX platforms)
echo "[4/8] rfsn_v11 tests..."
PYTHONPATH=. pytest -q rfsn_v11/tests -m "not mlx"

# 5. Benchmark tests
echo "[5/8] Benchmark tests..."
PYTHONPATH=. pytest -q tests/benchmarks

# 6. Quick shootout smoke (strict — quick mode handles no-MLX gracefully)
echo "[6/8] Quick shootout smoke..."
PYTHONPATH=. python benchmarks/kv_shootout.py --quick
echo "  Quick shootout completed."

# 7. Release integrity check
echo "[7/8] Release integrity check..."
PYTHONPATH=. python scripts/check_release_integrity.py

# 8. Build check (without contaminating current environment)
echo "[8/8] Build check..."
python -m pip install --upgrade build >/dev/null 2>&1
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

echo "=== Release Gate Passed ==="
