# Performance Roadmap — MLX-RFSN Direct-Packed Attention

## P4: Rebuild the performance path

### P4.1 — Remove `QUERY_START` from template specialization (DONE)

**Status:** Implemented in `rfsn_v10/kernels/metal/packed_v4_attention.py`.

The shader now reads `query_start_arr[0]` at runtime instead of using the
`QUERY_START` template constant. This eliminates a per-position compiled
specialisation, reducing kernel-cache pressure and improving dispatch latency.

### P4.2 — Replace full-history concatenation with paged/block descriptors

**Current state:** `_prepare_concatenated_buffers()` concatenates all sealed
blocks on every call. Cumulative data movement is O(T^2 / block_size).

**Target architecture:**
- Persistent pre-allocated packed pools (geometric growth)
- Block descriptor table with independent block buffers
- Paged KV storage passed to shader without full-history concatenation

**Files to modify:**
- `rfsn_v10/kernels/metal/packed_v4_attention.py`
- `rfsn_v10/cache/incremental_layer_cache.py`

### P4.3 — Cache compiled kernels by stable geometry only

**Current state:** Template list includes `NUM_Q_TOKENS`, `QUERY_START`, etc.

**Target state:** Kernel cache key should depend only on:
- `NUM_Q_HEADS`, `HEAD_DIM`, `BITS`, `GROUP_SIZE`
- `Q_PER_KV`
- `CAUSAL`

Dynamic values (`NUM_Q_TOKENS`, `NUM_BLOCKS`, `TOTAL_T`) should become
runtime buffers or loop bounds, not template parameters.

### P4.4 — Parallelize QK and value accumulation across SIMD groups

**Current state:** One thread per (q_head, q_token) pair performs serial work
over all tokens and dimensions.

**Target state:**
- SIMD-group reductions for max/sum
- Tiled K/V loading into threadgroup memory
- Vectorized unpack/dequantization (simd-group width)
- Parallel output-dimension accumulation

### P4.5 — Tile packed K/V and scales into threadgroup memory

**Design sketch:**
```
threadgroup float tg_k[THREADGROUP_K_TILE][HEAD_DIM];
threadgroup float tg_v[THREADGROUP_V_TILE][HEAD_DIM];
```

Each threadgroup loads a tile of K/V, decodes on-the-fly, and shares across
SIMD groups before moving to the next tile.

### P4.6 — Benchmark prefill and decode separately

**Current state:** Wall-clock timing includes both prefill and decode.

**Target state:**
- `mx.eval()` synchronisation before and after each phase
- Separate `prefill_ms` and `decode_ms` in contract
- First-token latency explicitly measured

### P4.7 — Compare against dense MLX and mlx-lm built-in quantized KV

**Benchmark matrix:**
| Backend | KV format | Notes |
|---|---|---|
| Dense MLX baseline | Full FP16 | Control |
| mlx-lm quantized KV | 8-bit, gs=64 | Same compression as RFSN |
| RFSN packed V4 | K8/V8 WHT | Current path |
| RFSN packed V4 (optimized) | K8/V8 WHT | After P4 |

All runs must use identical model revision, tokenizer, prompt suite, and
temperature=0.0 for deterministic comparison.

---

## Estimated effort

| Item | Effort | Blocked by |
|---|---|---|
| P4.1 QUERY_START removal | Small | — (DONE) |
| P4.2 Paged descriptors | Large | correctness proof |
| P4.3 Stable kernel cache | Medium | P4.2 |
| P4.4 SIMD parallelism | Large | Metal shader expertise |
| P4.5 Threadgroup tiling | Large | P4.4 |
| P4.6 Separate benchmarks | Small | — |
| P4.7 Competitive baseline | Medium | P4.6 |

**Recommendation:** Do not begin P4.2–P4.5 until the correctness pipeline
(kernel tests, real-model tests, per-step logit comparison) has run
cleanly for at least one full model evaluation without fallback.
