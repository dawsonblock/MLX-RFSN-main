#!/usr/bin/env bash
# Release gate script for MLX-RFSN Fusion Alpha 8.2
# Runs on any platform (Linux, macOS, Windows with bash)
set -euo pipefail

echo "=== MLX-RFSN Release Gate ==="

# 1. Compile check
echo "[1/7] Compile check..."
python -m compileall -q rfsn_v10 rfsn_v11 tests benchmarks scripts memory

# 2. Test collection (must not fail — catches import/shadowing bugs)
echo "[2/7] Test collection..."
PYTHONPATH=. pytest --collect-only -q tests rfsn_v11/tests

# 3. CPU tests (no MLX required)
echo "[3/7] CPU tests..."
PYTHONPATH=. RFSN_BACKEND=numpy RFSN_TELEMETRY_HMAC_KEY=test-secret \
  pytest -q tests -m "not mlx and not slow and not benchmark and not experimental and not integration and not db"

# 4. rfsn_v11 unit tests (skip MLX-dependent tests on non-MLX platforms)
echo "[4/7] rfsn_v11 tests..."
PYTHONPATH=. pytest -q rfsn_v11/tests -m "not mlx"

# 5. Benchmark tests
echo "[5/7] Benchmark tests..."
PYTHONPATH=. pytest -q tests/benchmarks

# 6. Quick shootout smoke (strict — quick mode handles no-MLX gracefully)
echo "[6/7] Quick shootout smoke..."
PYTHONPATH=. python benchmarks/kv_shootout.py --quick
echo "  Quick shootout completed."

# 7. Build check
echo "[7/7] Build check..."
python -m pip install --upgrade build >/dev/null 2>&1
python -m build
# Install only the most recent wheel (avoid conflicts from old builds)
latest_wh=$(ls -t dist/*.whl 2>/dev/null | head -1)
if [ -n "$latest_wh" ]; then
    python -m pip install --force-reinstall "$latest_wh" >/dev/null 2>&1
else
    echo "  No wheel found in dist/"
    exit 1
fi
python -c "import rfsn_v10, rfsn_v11; print('import ok')"

echo "=== Release Gate Passed ==="
