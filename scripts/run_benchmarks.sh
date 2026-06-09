#!/usr/bin/env bash
set -euo pipefail

echo "====================================="
echo "RFSN v10 Benchmarks"
echo "====================================="

echo ""
echo "--- Bitpack Benchmarks ---"
python3 benchmarks/benchmark_bitpack.py

echo ""
echo "--- KV Cache Benchmarks ---"
python3 benchmarks/benchmark_kv_cache.py

echo ""
echo "--- Attention Benchmarks ---"
python3 benchmarks/benchmark_attention.py

echo ""
echo "--- End-to-End Benchmarks ---"
python3 benchmarks/benchmark_end_to_end.py

echo ""
echo "====================================="
echo "All benchmarks completed."
echo "====================================="
