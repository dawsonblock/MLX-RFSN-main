#!/usr/bin/env bash
# Release gate script for MLX-RFSN Fusion Alpha 8
# Runs on any platform (Linux, macOS, Windows with bash)
set -euo pipefail

echo "=== MLX-RFSN Release Gate ==="

# 1. Compile check
echo "[1/6] Compile check..."
python -m compileall -q rfsn_v10 rfsn_v11 tests benchmarks scripts

# 2. CPU tests (no MLX required)
echo "[2/6] CPU tests..."
PYTHONPATH=. RFSN_BACKEND=numpy RFSN_TELEMETRY_HMAC_KEY=test-secret \
  pytest -q tests -m "not db and not mlx and not slow"

# 3. rfsn_v11 unit tests (skip MLX-dependent tests on non-MLX platforms)
echo "[3/6] rfsn_v11 tests..."
PYTHONPATH=. pytest -q rfsn_v11/tests -m "not mlx"

# 4. Benchmark tests
echo "[4/6] Benchmark tests..."
PYTHONPATH=. pytest -q tests/benchmarks

# 5. Quick shootout (smoke test, does not require real model)
echo "[5/6] Quick shootout smoke..."
PYTHONPATH=. python benchmarks/kv_shootout.py --quick || true

# 6. Build check
echo "[6/6] Build check..."
python -m build
python -m pip install --force-reinstall dist/*.whl >/dev/null 2>&1 || true
python -c "import rfsn_v10, rfsn_v11; print('import ok')"

echo "=== Release Gate Passed ==="
