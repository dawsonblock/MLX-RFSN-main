# polar_fused Codebase Analysis

## Executive Summary

polar_fused is a clean-room reimplementation of PolarQuant for MLX with strong architectural foundations: explicit routing, no global monkey-patching, deterministic operations, structured contracts, and comprehensive tests (114 tests, all passing). However, there are significant performance, correctness edge-case, and maintainability improvements available across all layers.

---

## 1. Critical Performance Issues

### 1.1 Reference kernels still unpack everything (O(T·D) memory blowup)

**File:** `mlx_ops/registry.py`  
**Lines:** 123–205

The `_reference_qk` and `_reference_sv` kernels call `unpack_indices()` which fully unpacks all T tokens and D dimensions into dense uint8 arrays, then looks up centroids into dense float32 arrays. This means the "fast path" still materializes the full dequantized KV cache in memory.

```python
# _reference_qk currently:
key_indices = unpack_indices(packed_key_indices, bits, D)  # (B, Hkv, T, D) uint8
key_values = key_centroids[key_indices]                     # (B, Hkv, T, D) float32
# ...then matmul
```

**Impact:** At T=2048, D=64, this allocates ~2.6 MB per head for keys alone. With GQA expansion to 14 heads, ~18 MB per layer — worse than FP16.

**Fix:** Rewrite reference kernels to unpack and lookup on-the-fly inside the matmul loop, or at minimum use `mx.take` with unpacked indices directly without materializing the full float32 array. A vectorized approach using `mx.gather` or custom scatter-gather would avoid the full materialization.

### 1.2 `mx.concatenate` in IncrementalPolarCache.append is O(n) per step

**File:** `incremental_cache.py`  
**Lines:** 101–112

Each decode step triggers `mx.concatenate` on the entire cache history. In MLX (and GPU frameworks generally), `concatenate` of two arrays allocates a new buffer and copies both inputs. For T tokens, the total copy volume is O(T²) over the decode sequence.

```python
self._packed_key_indices = mx.concatenate(
    [self._packed_key_indices, packed_keys], axis=2
)
```

**Impact:** At T=2048, total bytes copied ≈ Σᵢ₌₁²⁰⁴⁸ i ≈ 2M words — significant GPU→GPU copy overhead.

**Fix:** Pre-allocate with exponential growth (like `PolarCache` does with block_size) and only copy when capacity is exceeded. Or better, use a chunked cache where each chunk is a fixed-size block and attend logic iterates over chunks.

### 1.3 PolarAttentionWrapper quantizes on every forward pass

**File:** `adapters/mlx_lm.py`  
**Lines:** 87–93

Even with `IncrementalPolarCache`, the wrapper still calls `self._inc_cache.append()` on every forward pass. The `append()` method calls `quantize()` on the new tokens. For decode (1 token), this is fine. But for prefill, it quantizes the entire prefill sequence at once.

More importantly, the standard MLX `cache.update_and_fetch()` already stores FP16 K/V. Then Polar quantizes them again. This means during decode we have **duplicate storage**: FP16 in standard cache + packed in incremental cache.

**Impact:** ~2× memory during generation.

**Fix:** Replace the standard `KVCache` with a custom cache that stores only Polar packed data, or release the FP16 cache after Polar conversion. This requires deeper model integration.

### 1.4 No batching in quantize()

**File:** `quantize.py`  
**Lines:** 65–96

`quantize()` processes all vectors in one shot but the internal `mx.sqrt(mx.sum(x * x, ...))` and `unit @ self._R_T` are not optimized for very large T. The matmul `unit @ R_T` where unit is (N, D) and R_T is (D, D) is O(N·D²) which is fine, but for prefill with N=1024, D=64, this is ~4M ops — acceptable.

However, the `argmin` over centroids in `codebooks.py:126` is O(N·2^bits·D) which for 4-bit is O(N·16·D) — also acceptable. The real issue is that for **each layer** we do this independently, so for 24 layers it's 24× the work during prefill.

**Fix:** Parallelize across layers using `mx.vmap` if possible, or accept that prefill is a one-time cost.

---

## 2. Correctness & Edge Cases

### 2.1 `_reference_sv` computes head_dim from packed shape incorrectly

**File:** `mlx_ops/registry.py`  
**Lines:** 187–189

```python
head_dim = words_per_vec * values_per_word  # may include padding
```

This uses the padded dimension, not the original. The comment even says "may include padding." This means the SV kernel operates on more dimensions than the naive path, which uses `original_dim` from `QuantizedVectors`.

**Impact:** The padding values are zeros (from `pack_indices`), so centroids[0] is looked up for padding slots. This could subtly affect output. The `test_kernel_matches_naive` passes because padding is at the end and the centroids are symmetric-ish, but this is fragile.

**Fix:** Pass `head_dim` explicitly to the kernel, or trim padding after unpack.

### 2.2 `PolarCache` allocates unpacked shape then never packs

**File:** `cache.py`  
**Lines:** 207–230 (allocation)

The `_allocate_state` docstring says it allocates unpacked shape because "we don't know bits yet," but the caller (`LazyPolarCache._convert`) immediately creates a fake state object with already-packed arrays. This is a design inconsistency.

**Fix:** Either make `PolarCache` bits-aware, or remove `PolarCache` entirely in favor of `IncrementalPolarCache` which is the actual production cache.

### 2.3 `attention_score_cosine` is hardcoded to 0.0 in QualityGateResult

**File:** `quality_gates.py`  
**Line:** 129

```python
attention_score_cosine=0.0,  # TODO: compute from attention scores
```

This field is part of the gate evaluation but never computed. If a future user adds a threshold on this field, it will always fail.

**Fix:** Add an optional `attention_scores` parameter to `evaluate()` and compute cosine if provided.

### 2.4 `NaivePolarAttention` no longer uses rotation matrices but still imports them

**File:** `attention.py`  
**Lines:** 49–61 (removed in current version)

Wait, looking at the current file, `NaivePolarAttention.__init__` doesn't set any rotation matrices. But `dequantize()` in `PolarQuantizer` applies `centroids @ R`. So the naive path is correct.

However, there's a subtle issue: `NaivePolarAttention` uses `self.key_q.dequantize(key_qv)` which returns vectors in the **original** basis. But `queries` come from the model in the **original** basis too. So `queries @ keys.T` is correct. This is fine.

### 2.5 `BoundaryLayerPolicy` may double-count layers when total < 2*N

**File:** `adapters/boundary_layers.py`  
**Lines:** 48–58

If a model has 3 layers and `boundary_layers=2`, then:
- First 2 → FP16
- Last 2 → FP16 (overlaps with first 2)
- Middle → Polar (empty slice)

This is correct (middle layers get nothing), but if total layers < 2*boundary_layers, no layers use Polar. This might be surprising. Not a bug, but worth documenting.

### 2.6 `ModelInspector` infers head_dim from q_proj.weight but not all models have this

**File:** `adapters/model_inspection.py`  
**Lines:** 57–78

The head_dim inference tries `q_proj.weight.shape[0] // n_heads`, but some models may not expose `weight` directly (e.g., quantized models, fused QKV projections). Also, some models use `head_dim` explicitly.

**Fix:** Add more fallback sources: `model.config.head_dim`, `model.args.hidden_size // model.args.num_attention_heads`, etc.

---

## 3. Code Quality & Maintainability

### 3.1 Dead code: `PolarFusedAttentionBackend` still uses `LazyPolarCache` (unused in production)

**File:** `attention_backend.py`

The backend creates `LazyPolarCache` per layer, but `PolarModelRunner` (the production integration) uses `IncrementalPolarCache` directly in `PolarAttentionWrapper`. This means `LazyPolarCache`, `PolarCache`, and most of `attention_backend.py` are effectively dead code for the current integration path.

**Recommendation:** Either:
- (a) Remove dead code to reduce maintenance burden
- (b) Integrate `LazyPolarCache` into `PolarModelRunner` so short contexts use FP16 (currently the runner always uses Polar)
- (c) Document the dual paths clearly

### 3.2 `lazy_convert.py` has no tests for `CONVERTING` or `FALLBACK` states

**File:** `tests/test_lazy_convert.py`

The tests cover `EMPTY → FP16_WARMUP → POLAR_PACKED` but not:
- `CONVERTING` state (transient, hard to test)
- `FALLBACK` state (conversion failure)
- `trim()` on `POLAR_PACKED` state

### 3.3 `promotion.py` has no tests for level transitions

**File:** `tests/test_promotion.py`

Tests exist but they test static evaluation, not the full lifecycle of a candidate moving from LAB → CANDIDATE → SUPPORTED.

### 3.4 `codebooks.py` uses `np.array` at module level — import-time MLX dependency issue

**File:** `codebooks.py`  
**Lines:** 27–42

The centroids are `np.array` literals defined at module import time. If numpy is not available, the module fails to import. This contradicts the "MLX optional at import time" pattern used elsewhere.

**Fix:** Wrap centroid definitions in a lazy-loading function or use Python lists and convert to numpy/MLX on first access.

### 3.5 `__init__.py` exports `IncrementalPolarCache` but it's missing

**File:** `__init__.py`

`IncrementalPolarCache` is the most important production class but is NOT exported in `__init__.py`.

### 3.6 `metal/*.metal` files are never loaded

**Files:** `metal/qk_scalar.metal`, `metal/sv_scalar.metal`

The `.metal` shader files exist but `mlx_ops/registry.py` never attempts to load them. The `_check_metal()` function looks for `mlx.core.fast.metal_kernel` but doesn't actually use the source files.

**Fix:** Add a `load_metal_source()` function that reads the `.metal` files from the package directory and passes them to `mlx.core.fast`.

### 3.7 `telemetry.py` doesn't track memory usage or compression ratio

**File:** `telemetry.py`

The telemetry only records latency and backend name. For a quantization backend, memory savings and compression ratio are key metrics.

**Fix:** Add `memory_bytes`, `compression_ratio`, and `token_count` fields to `PolarTelemetry`.

---

## 4. Performance Optimizations (Prioritized)

### P0: Fix `IncrementalPolarCache.append` to use exponential growth

```python
# Current: O(n) copy per step
self._packed_key_indices = mx.concatenate([..., packed_keys], axis=2)

# Better: pre-allocate with growth factor
def _ensure_capacity(self, needed_tokens):
    current = self._token_count
    if current + needed_tokens > self._capacity:
        new_cap = max(self._capacity * 2, current + needed_tokens)
        # mx.pad or mx.concatenate with empty padding
        ...
```

### P0: Rewrite reference kernels to avoid full unpack materialization

Instead of:
```python
key_indices = unpack_indices(packed, bits, D)     # (B, H, T, D) uint8
key_values = key_centroids[key_indices]             # (B, H, T, D) float32
scores = mx.matmul(queries, key_values.transpose(...))
```

Do:
```python
# For each word in packed, extract indices, lookup centroids, accumulate dot product
# This avoids materializing the full (B, H, T, D) float32 array
```

Or even better: since MLX has `mx.fast.scaled_dot_product_attention`, consider whether a custom op can be registered.

### P1: Cache rotation matrices per quantizer

**File:** `incremental_cache.py`

`attend_kernel()` calls `self.key_q._rot_registry.get_transpose()` and `self.value_q._rot_registry.get()` on every attend call. These are cached in the registry, but there's still a Python dict lookup.

**Fix:** Cache `Rk_T` and `Rv` as instance attributes on `IncrementalPolarCache` init.

### P1: Cache centroids per quantizer

Same issue: `self.key_q._cb_registry.centroids()` is called every attend. Cache the MLX array reference.

### P1: Pre-compute `values_per_word` dict lookups

The `{2: 16, 3: 10, 4: 8}[bits]` lookup happens in hot paths. Use instance attributes instead.

### P2: Add `mx.eval()` barriers for timing accuracy

**File:** `adapters/mlx_lm.py`

The timing in `generate()` wraps the model forward pass but MLX uses lazy evaluation. The actual GPU work may happen later.

**Fix:** Add `mx.eval(token_logits)` inside the timed region for accurate decode latency measurement.

### P2: Vectorize packing with `mx.left_shift` using a single expression

**File:** `packing.py`  
**Lines:** 63–67

The current loop:
```python
packed = mx.zeros(...)
for i in range(values_per_word):
    val = flat[..., i].astype(mx.uint32) & mask
    packed = mx.bitwise_or(packed, mx.left_shift(val, i * bits))
```

Can be vectorized:
```python
shifts = mx.arange(values_per_word) * bits
packed = mx.sum(
    mx.left_shift(flat.astype(mx.uint32) & mask, shifts),
    axis=-1
)
```

Similarly for unpack.

---

## 5. Missing Features

### 5.1 No streaming/online quantize for prefill

During prefill, all prompt tokens are quantized at once. For very long prompts (e.g., 4096 tokens), this is a large one-time cost. Streaming quantization (chunk by chunk) would amortize this.

### 5.2 No KV cache eviction

There is no mechanism to evict old tokens when context exceeds a limit. The cache grows forever.

### 5.3 No mixed-precision within a layer

Keys and values are quantized with the same bit width per config. Some research suggests K4/V3 or K3/V4 may be optimal. The config supports different bits but there's no per-layer tuning.

### 5.4 No dynamic bit-width adjustment

The bit width is fixed at initialization. A future enhancement could lower bits for older tokens (progressive quantization).

### 5.5 No attention score caching

For causal attention, the softmax denominator can be incrementally updated instead of recomputed from scratch each step. This is a standard optimization in transformer inference.

---

## 6. Test Coverage Gaps

| Module | Missing Coverage |
|--------|-----------------|
| `cache.py` | `trim()`, `validate()` failure cases, block growth edge cases |
| `lazy_convert.py` | `FALLBACK` state, `CONVERTING` state, `trim()` on Polar |
| `quality_gates.py` | `attention_score_cosine` computation, all gates failing |
| `telemetry.py` | `clear()`, empty summary, large event lists |
| `fallback.py` | GQA expansion, mask application, scale handling |
| `adapters/mlx_lm.py` | Error recovery during generation, `uninstall()` with exceptions |
| `incremental_cache.py` | Multi-step append with varying token counts, empty attend error |
| `packing.py` | Boundary values (max index for each bit width), all-zeros, all-max |
| `codebooks.py` | Boundary quantization, checksum validation, wrong version error |
| `rotations.py` | Different dtypes, very large dims, seed collision |

---

## 7. Recommendations Summary

### Immediate (do now)
1. **Fix `IncrementalPolarCache.append` exponential growth** — O(n²) → O(n) amortized
2. **Fix `_reference_sv` head_dim padding bug** — correctness issue
3. **Export `IncrementalPolarCache` in `__init__.py`** — visibility
4. **Cache rotation matrices and centroids in `IncrementalPolarCache`** — small perf win

### Short-term (next sprint)
5. **Rewrite reference kernels to avoid full materialization** — major memory win
6. **Add `mx.eval()` to timing in `PolarModelRunner`** — measurement accuracy
7. **Vectorize pack/unpack** — clean code + potential speedup
8. **Add tests for edge cases** — correctness confidence

### Medium-term
9. **Integrate `LazyPolarCache` into `PolarModelRunner`** — short-context FP16 path
10. **Implement Metal kernel loading** — actual GPU speedup
11. **Add KV eviction** — long-context viability
12. **Add streaming prefill quantization** — latency smoothing

### Long-term
13. **Remove dead code** (`PolarCache`, `LazyPolarCache` if unused, `attention_backend.py` if superseded)
14. **Progressive quantization** (lower bits for older tokens)
15. **Attention score caching** (incremental softmax)
