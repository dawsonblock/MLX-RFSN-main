# RFSN Dense MLX Baseline Report

**Model:** `smoke/Qwen2.5-0.5B`  
**Generated:** 20260610T002513Z  
**Mode:** smoke (synthetic)  

## Per-Prompt Results

| prompt_id | context_len | output_tokens | prefill_tps | decode_tps | TTFT_ms | total_ms | peak_mb | kv_mb |
|---|---|---|---|---|---|---|---|---|
| short_chat_512 | 18 | 100 | 1109.6 | 57.6 | 34.1 | 1753.7 | 2546.1 | 1.7 |
| coding_512 | 27 | 100 | 1190.2 | 70.4 | 39.5 | 1442.2 | 1692.2 | 2.5 |

## Notes

- All runs at temperature=0, seed=42.
- Dense baseline has logit_cosine=1.0, top5_overlap=1.0 by definition.
- `kv_mb` is an estimate based on model architecture; actual device allocation may differ.
