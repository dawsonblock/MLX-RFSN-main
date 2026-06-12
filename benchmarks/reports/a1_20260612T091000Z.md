# RFSN Benchmark Report: `a1`

**Generated:** 20260612T091000Z  
**Models:** `smoke/Qwen2.5-0.5B`  

## Summary

| Verdict | Count |
|---|---|
| PROMOTE | 0 |
| KEEP_EXPERIMENTAL | 0 |
| REJECT | 0 |
| REGRESSION | 0 |
| SMOKE_PASS | 6 |
| SMOKE_FAIL | 0 |

## Candidate Results

### `A1_wht_grouped_k8v4_gs64` — prompt `short_chat_512`

**Verdict:** `SMOKE_PASS`  
**Model:** `smoke/Qwen2.5-0.5B`  

#### Quality

| metric | candidate | baseline |
|---|---|---|
| logit_cosine | 0.99983 | 1.00000 |
| top5_overlap | 0.993 | 1.000 |
| top10_overlap | 0.982 | 1.000 |
| attention_score_cosine | 0.99969 | 1.00000 |
| attention_top5_overlap | 0.994 | 1.000 |
| perplexity_delta | +0.0045 | +0.0000 |
| visible_output_drift | 0.007 | 0.000 |

#### Memory

| metric | value |
|---|---|
| peak_memory_mb | 1897.9 |
| kv_cache_memory_mb (dense est.) | 1.7 |
| compressed_kv_memory_mb | 0.7 |
| metadata_memory_mb | 0.5 |
| compression_factor | 2.27x |
| effective_bits/elem | 6.00 |

#### Runtime

| metric | value |
|---|---|
| prefill_tps | 1109.6 |
| decode_tps | 61.1 |
| first_token_latency_ms | 34.2 |
| total_latency_ms | 1898.8 |
| compression_time_ms | 3.53 |
| decompression_time_ms | 6.55 |

**Reason:** smoke run — harness validated (not promotion evidence)

### `A1_wht_grouped_k8v4_gs64` — prompt `coding_512`

**Verdict:** `SMOKE_PASS`  
**Model:** `smoke/Qwen2.5-0.5B`  

#### Quality

| metric | candidate | baseline |
|---|---|---|
| logit_cosine | 0.99835 | 1.00000 |
| top5_overlap | 0.974 | 1.000 |
| top10_overlap | 0.993 | 1.000 |
| attention_score_cosine | 0.99794 | 1.00000 |
| attention_top5_overlap | 0.984 | 1.000 |
| perplexity_delta | +0.0074 | +0.0000 |
| visible_output_drift | 0.019 | 0.000 |

#### Memory

| metric | value |
|---|---|
| peak_memory_mb | 1936.6 |
| kv_cache_memory_mb (dense est.) | 2.5 |
| compressed_kv_memory_mb | 1.1 |
| metadata_memory_mb | 0.5 |
| compression_factor | 2.27x |
| effective_bits/elem | 6.00 |

#### Runtime

| metric | value |
|---|---|
| prefill_tps | 941.8 |
| decode_tps | 82.4 |
| first_token_latency_ms | 47.6 |
| total_latency_ms | 1384.1 |
| compression_time_ms | 2.75 |
| decompression_time_ms | 7.00 |

**Reason:** smoke run — harness validated (not promotion evidence)

### `A1_wht_grouped_k8v4_gs64` — prompt `retrieval_2048`

**Verdict:** `SMOKE_PASS`  
**Model:** `smoke/Qwen2.5-0.5B`  

#### Quality

| metric | candidate | baseline |
|---|---|---|
| logit_cosine | 0.99784 | 1.00000 |
| top5_overlap | 0.974 | 1.000 |
| top10_overlap | 0.984 | 1.000 |
| attention_score_cosine | 0.99893 | 1.00000 |
| attention_top5_overlap | 0.993 | 1.000 |
| perplexity_delta | +0.0001 | +0.0000 |
| visible_output_drift | 0.016 | 0.000 |

#### Memory

| metric | value |
|---|---|
| peak_memory_mb | 2202.9 |
| kv_cache_memory_mb (dense est.) | 64.1 |
| compressed_kv_memory_mb | 28.2 |
| metadata_memory_mb | 0.5 |
| compression_factor | 2.27x |
| effective_bits/elem | 6.00 |

#### Runtime

| metric | value |
|---|---|
| prefill_tps | 1080.1 |
| decode_tps | 51.3 |
| first_token_latency_ms | 654.5 |
| total_latency_ms | 2707.9 |
| compression_time_ms | 2.88 |
| decompression_time_ms | 5.39 |

**Reason:** smoke run — harness validated (not promotion evidence)

### `A1_wht_grouped_k8v4_gs64` — prompt `summarization_2048`

**Verdict:** `SMOKE_PASS`  
**Model:** `smoke/Qwen2.5-0.5B`  

#### Quality

| metric | candidate | baseline |
|---|---|---|
| logit_cosine | 0.99709 | 1.00000 |
| top5_overlap | 0.976 | 1.000 |
| top10_overlap | 0.988 | 1.000 |
| attention_score_cosine | 0.99717 | 1.00000 |
| attention_top5_overlap | 0.979 | 1.000 |
| perplexity_delta | +0.0085 | +0.0000 |
| visible_output_drift | 0.005 | 0.000 |

#### Memory

| metric | value |
|---|---|
| peak_memory_mb | 1897.7 |
| kv_cache_memory_mb (dense est.) | 214.7 |
| compressed_kv_memory_mb | 94.5 |
| metadata_memory_mb | 0.5 |
| compression_factor | 2.27x |
| effective_bits/elem | 6.00 |

#### Runtime

| metric | value |
|---|---|
| prefill_tps | 1106.0 |
| decode_tps | 72.4 |
| first_token_latency_ms | 2153.0 |
| total_latency_ms | 3746.1 |
| compression_time_ms | 4.26 |
| decompression_time_ms | 3.00 |

**Reason:** smoke run — harness validated (not promotion evidence)

### `A1_wht_grouped_k8v4_gs64` — prompt `needle_8192`

**Verdict:** `SMOKE_PASS`  
**Model:** `smoke/Qwen2.5-0.5B`  

#### Quality

| metric | candidate | baseline |
|---|---|---|
| logit_cosine | 0.99845 | 1.00000 |
| top5_overlap | 0.990 | 1.000 |
| top10_overlap | 0.988 | 1.000 |
| attention_score_cosine | 0.99883 | 1.00000 |
| attention_top5_overlap | 0.973 | 1.000 |
| perplexity_delta | +0.0038 | +0.0000 |
| visible_output_drift | 0.006 | 0.000 |

#### Memory

| metric | value |
|---|---|
| peak_memory_mb | 1956.8 |
| kv_cache_memory_mb (dense est.) | 246.3 |
| compressed_kv_memory_mb | 108.4 |
| metadata_memory_mb | 0.5 |
| compression_factor | 2.27x |
| effective_bits/elem | 6.00 |

#### Runtime

| metric | value |
|---|---|
| prefill_tps | 809.1 |
| decode_tps | 49.3 |
| first_token_latency_ms | 3376.9 |
| total_latency_ms | 5687.7 |
| compression_time_ms | 4.88 |
| decompression_time_ms | 6.67 |

**Reason:** smoke run — harness validated (not promotion evidence)

### `A1_wht_grouped_k8v4_gs64` — prompt `multi_turn_4096`

**Verdict:** `SMOKE_PASS`  
**Model:** `smoke/Qwen2.5-0.5B`  

#### Quality

| metric | candidate | baseline |
|---|---|---|
| logit_cosine | 0.99832 | 1.00000 |
| top5_overlap | 0.979 | 1.000 |
| top10_overlap | 0.991 | 1.000 |
| attention_score_cosine | 0.99920 | 1.00000 |
| attention_top5_overlap | 0.983 | 1.000 |
| perplexity_delta | +0.0018 | +0.0000 |
| visible_output_drift | 0.017 | 0.000 |

#### Memory

| metric | value |
|---|---|
| peak_memory_mb | 1343.3 |
| kv_cache_memory_mb (dense est.) | 6.6 |
| compressed_kv_memory_mb | 2.9 |
| metadata_memory_mb | 0.5 |
| compression_factor | 2.27x |
| effective_bits/elem | 6.00 |

#### Runtime

| metric | value |
|---|---|
| prefill_tps | 1086.8 |
| decode_tps | 62.6 |
| first_token_latency_ms | 73.8 |
| total_latency_ms | 1863.7 |
| compression_time_ms | 1.17 |
| decompression_time_ms | 4.96 |

**Reason:** smoke run — harness validated (not promotion evidence)

## Dense Baseline Reference

| metric | value |
|---|---|
| model_id | `smoke/Qwen2.5-0.5B` |
| decode_tps | 57.6 |
| peak_memory_mb | 2546.1 |
| kv_cache_memory_mb | 1.7 |

---
*Generated by RFSN benchmark harness*
