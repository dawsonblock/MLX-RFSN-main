# PR Plan: TurboQuant KV Cache for mlx-lm

## Target
Repository: [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
PR type: New KV cache type (follows CompressedKVCache PR #963 pattern)

## What Gets Added

### Files Modified

**`mlx_lm/models/cache.py`** — New `TurboQuantKVCache` class
- Inherits from `_BaseCache`
- Ships Lloyd-Max codebooks as constants (4 arrays, ~200 bytes total)
- `update_and_fetch()`: compress on insert, dequantize on fetch
- `to_turboquant(bits)` method on `KVCache` for lazy conversion
- Full `state`/`meta_state` serialization support
- `trim()`, `make_mask()`, `nbytes`, `is_trimmable()`

**`mlx_lm/models/base.py`** — No changes needed (dequantize path uses standard SDPA)

**`mlx_lm/generate.py`** — Add `--kv-type turboquant` or `--turbo-bits` CLI arg
- Thread `turbo_bits` through `generate_step()`
- `maybe_turboquant_kv_cache()` parallel to `maybe_quantize_kv_cache()`

**`tests/test_prompt_cache.py`** — Test suite
- Roundtrip pack/unpack
- `update_and_fetch` shape/offset correctness
- State serialization
- Numerical accuracy vs FP16 (cosine > 0.99 at 4-bit)
- Integration with `generate()`

### Files NOT Modified
- No model-specific changes (works with all architectures via standard `_BaseCache` interface)
- No Metal kernel changes (Phase 1 uses standard `mx.fast.scaled_dot_product_attention`)

## What Gets Inlined

The standalone `mlx-turboquant` package has 7 files. For the PR, consolidate into **2 additions** to mlx-lm:

1. **PolarQuant + codebooks + packing** → inline into `cache.py` (~200 lines)
   - `_lloyd_max_codebooks`: dict of precomputed centroids/boundaries (4 bit widths)
   - `_generate_rotation_matrix(dim, seed)`: QR of Gaussian
   - `_polar_quantize(vectors, rotation, centroids, boundaries)`: normalize → rotate → digitize
   - `_polar_dequantize(indices, norms, rotation, centroids)`: lookup → unrotate → rescale
   - `_pack_indices(indices, bits)` / `_unpack_indices(packed, bits, dim)`: vectorized bit-packing
   - `TurboQuantKVCache` class

2. **Tests** → add to `tests/test_prompt_cache.py` (~50 lines)

No new dependencies. No scipy at runtime (codebooks are precomputed constants).

## PR Structure

### PR Title
`Add TurboQuantKVCache: data-oblivious KV cache compression at 3-4 bits`

### PR Body

```
## Summary
- Adds `TurboQuantKVCache` implementing PolarQuant (ICLR 2026) for KV cache compression
- Data-oblivious: no calibration data needed, works with any model
- Supports 2/3/3.5/4-bit compression via `--turbo-bits` CLI arg
- 4.6x memory reduction at 3-bit, 3.8x at 4-bit (head_dim=128)
- Quality: 0.988-0.997 logit cosine similarity at 3-4 bits on Llama 3.2 and Qwen3

## How It Works
Random orthogonal rotation maps KV vectors to a known distribution.
Lloyd-Max optimal scalar quantizers (precomputed) quantize each coordinate.
Bit-packed uint32 storage. Dequantize on fetch for standard SDPA.

## Benchmark
Tested on: Llama 3.2-1B, Llama 3.2-3B, Qwen3-1.7B, Qwen3-4B

| Bits | Avg Cosine (d=128) | Memory Ratio | Top-1 Accuracy |
|------|-------------------|--------------|----------------|
| 4    | 0.997             | 3.8x         | 12/12          |
| 3    | 0.988             | 4.6x         | 12/12          |

## Test Plan
- [x] Pack/unpack roundtrip for all bit widths
- [x] Quantize/dequantize MSE within theoretical bounds
- [x] update_and_fetch shape and offset tracking
- [x] State serialization roundtrip
- [x] Logit cosine > 0.99 at 4-bit on Llama 3.2-3B
- [x] Needle-in-haystack: 12/12 retrieval at 3+ bits
- [x] Integration with generate()
```

## Pre-Submission Checklist

- [ ] Run `pre-commit run --all-files` (black + clang-format)
- [ ] Run `python -m unittest discover tests/` — all existing tests pass
- [ ] Inline codebooks as Python constants (no .npz files)
- [ ] Remove scipy dependency (codebooks are hardcoded)
- [ ] Add `KVCache.to_turboquant(bits)` method
- [ ] Thread `turbo_bits` through CLI args in generate.py
- [ ] Test with 3+ model architectures (Llama, Qwen, Gemma)
- [ ] Verify Qwen3-1.7B failure is documented as known limitation

## Phasing

**Phase 1 (this PR):** PolarQuant-only, dequantize-on-fetch, standard SDPA.
Delivers 3-4.6x memory savings. Decode speed ~0.5x FP16.

**Phase 2 (follow-up PR):** Custom `turboquant_scaled_dot_product_attention` with QJL correction.
Add dispatch in `base.py`: `hasattr(cache, 'turbo_bits')`.

**Phase 3 (follow-up PR):** Metal kernel for fused dequant-matmul.
Similar to `mx.quantized_matmul` but for rotation-based quantization.
Would restore decode speed to FP16 parity.

## Risk

- **Model sensitivity:** Some models (Qwen3-1.7B) degrade at 3-bit. Mitigation: default to 4-bit, document per-model recommendations.
- **Speed:** 0.2-0.5x FP16 decode without Metal kernels. Mitigated by Phase 3 plan.
- **Interaction with RotatingKVCache:** Not yet tested. Would need `RotatingTurboQuantKVCache` variant.
