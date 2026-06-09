#!/usr/bin/env bash
set -euo pipefail

echo "====================================="
echo "RFSN v10 Test Suite"
echo "====================================="

# Syntax check
echo "Running compileall..."
python3 -m compileall rfsn_v10 tests benchmarks

# Core tests
echo ""
echo "Running bitpack tests..."
python3 -m pytest tests/test_bitpack.py -v

echo ""
echo "Running KV manager tests..."
python3 -m pytest tests/test_kv_manager.py -v

echo ""
echo "Running metal kernel math tests..."
python3 -m pytest tests/test_metal_kernel_math.py -v

echo ""
echo "Running attention tests..."
python3 -m pytest tests/test_attention.py -v

echo ""
echo "Running runtime tests..."
python3 -m pytest tests/test_runtime.py -v

echo ""
echo "====================================="
echo "All tests completed."
echo "====================================="
