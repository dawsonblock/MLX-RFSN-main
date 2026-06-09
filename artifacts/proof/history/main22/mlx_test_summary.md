# MLX Test Summary

Release: Main 22
Date: 2026-06-02

Hardware:
- Mac model: arm64
- Apple chip: Apple M2 Pro
- RAM: 16 GB
- macOS: Darwin 25.2.0
- Python: 3.12.0
- MLX version: 0.31.2

Commands run:
- python -m compileall -q .
- python test_syntax.py
- python test_agent_core_integration.py
- python -m pytest -q -rs
- pytest tests/test_fused_kernel_mlx.py -q -s
- pytest tests/test_kernel_equivalence_mlx.py -q -s
- pytest tests/test_metal_kernel_math.py -q -s
- pytest tests/test_kv_manager.py -q -s
- pytest tests/test_bitpack.py -q -s
- pytest tests/test_invalid_symmetric_codes_mlx.py -q -s
- pytest tests/test_attention.py -q -s
- pytest tests/test_attention_reserved_blocks_mlx.py -q -s
- pytest tests/test_runtime.py -q -s
- pytest tests/test_memory_guard_runtime_mlx.py -q -s
- pytest tests/test_sparse_safety_gate.py -q -s

Results:
- passed: all
- failed: 0
- skipped: 0

Raw log: mlx_pytest_raw.log

Strict Metal fallback:
- allowed: no
- observed fallback: no

Fused kernel:
- route: metal_fused_dequant_wht_sign
- status: valid across all tested shapes/bits
- cosine vs reference: 1.000000
- max_abs_diff vs reference: 0.0

Notes:
- All tests passed on actual Apple Silicon hardware (M2 Pro).
- Metal kernel execution verified on macOS.
- Fused packed-dequant-WHT-sign kernel proven equivalent to sequential reference.
- Sparse decode remains disabled by default and is validated as experimental-safe.
