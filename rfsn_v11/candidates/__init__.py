"""KV-compression candidate interface for the shootout benchmark.

Phase 0 Scope Freeze (2026-06-15): Only three candidates are active for
K8/V8 GS64 validation.  All other adapters remain in source but are
excluded from default benchmarks, release gates, and promotion reports.

Active (canonical) candidates:
  - MLXLMBaseline          → dense FP16 control
  - MLXLMQuantizedKV       → MLX-LM built-in 8-bit KV control
  - RFSNDirectPackedCandidate → K8/V8 GS64 direct-packed (strict mode)

Archived experiments (require --experimental):
  - RFSNV10Candidate, RFSNV11Candidate
  - TurboQuantV2Candidate
  - PolarReferenceAdapter, TurboPolarAdapter
  - Additional bit-width variants (K8/V5, K8/V6, K16/V8, K8/V16, K16/V16)
"""
