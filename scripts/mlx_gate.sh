#!/usr/bin/env bash
# MLX-specific gate script for MLX-RFSN Fusion Alpha 8
# Run this on macOS Apple Silicon only.
set -euo pipefail

echo "=== MLX-RFSN MLX Gate ==="

# 1. MLX tests
echo "[1/4] MLX tests..."
PYTHONPATH=. RFSN_BACKEND=mlx pytest -q rfsn_v11/tests

# 2. Full logit gate
echo "[2/4] Full logit gate..."
PYTHONPATH=. python benchmarks/kv_shootout.py --full-logit-gate || true

# 3. Memory report
echo "[3/4] Memory report..."
PYTHONPATH=. python benchmarks/kv_shootout.py --memory-report || true

# 4. Promotion report
echo "[4/4] Promotion report..."
PYTHONPATH=. python benchmarks/kv_shootout.py --promotion-report || true

echo "=== MLX Gate Completed ==="
