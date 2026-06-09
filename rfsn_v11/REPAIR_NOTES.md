# RFSN v11 — Repair Notes

This file documents every bug fixed during the implementation of `rfsn_v11` relative
to the upstream source repositories it was built from.

---

## Bug 1: `cfg.__dict__` in streaming server path

**File:** `rfsn_v10/server/app.py:226`
**Symptom:** `generator.generate(prompt, **cfg.__dict__)` passes Pydantic v2 internal
dunder fields (e.g. `__fields_set__`, `__dict__`, `model_config`) to the generator's
`generate()` method, causing `TypeError: unexpected keyword argument` crashes in
production or silently passing garbage kwargs.

**Root cause:** Pydantic v2 `BaseModel.__dict__` is not a clean field dict — it includes
Pydantic internals. The correct call is `.model_dump()`.

**Fix applied in:** `rfsn_v11/server/app.py`
- Streaming path: `cfg.__dict__` → `cfg.model_dump(exclude={"stream"})`
- `GenerationConfig` itself is now a Pydantic `BaseModel` (not dataclass) so `.model_dump()`
  is always available and always returns only declared fields.

---

## Bug 2: Blocking non-streaming path on asyncio event loop

**File:** `rfsn_v10/server/app.py:196`
**Symptom:** `generator.chat(prompt, ...)` is a synchronous blocking call invoked directly
inside an `async def` handler. This blocks the asyncio event loop for the full generation
duration, preventing all other requests from being served.

**Fix applied in:** `rfsn_v11/server/app.py`
```python
# BEFORE (blocks event loop):
result = generator.chat(prompt, ...)

# AFTER (offloads to thread pool):
result = await asyncio.to_thread(generator.chat, prompt, ...)
```

---

## Bug 3: Blocking streaming path on asyncio event loop

**File:** `rfsn_v10/server/app.py:226` (inside `_sse_stream` async generator)
**Symptom:** `for idx, token in enumerate(generator.generate(prompt, ...))` runs a
synchronous Python generator on the event loop thread, blocking it between every token.
Other concurrent requests cannot execute while tokens are being generated.

**Fix applied in:** `rfsn_v11/server/app.py` (`_sse_stream`)
- Generator runs in a daemon thread that pushes tokens to a `queue.Queue`.
- The async consumer uses `await loop.run_in_executor(None, queue.get)` to receive
  tokens without blocking the event loop.

---

## Bug 4: `pack_indices` Python loop in packing.py

**File:** `mlx-turboquant-main/mlx_turboquant/packing.py:39-41`
**Symptom (despite file header claiming "Fully vectorized"):**
```python
packed = shifted[..., 0]
for i in range(1, vals_per_int):
    packed = packed | shifted[..., i]
```
For 2-bit packing, `vals_per_int = 16`, so this emits 15 sequential MLX graph nodes per
call. At large batch sizes this creates significant Python-level overhead and prevents
the compiler from fusing the reduction.

**Fix applied in:** `rfsn_v11/quant/packing.py`
```python
# Vectorized OR-reduce via sum (values never overlap, so sum == bitwise OR)
packed = mx.sum(shifted, axis=-1).astype(mx.uint32)
```
The `packed_indices → unpack_indices` roundtrip is exact (verified by tests for
bits=2, 3, 4 and dim=64, 128, 256).

---

## Bug 5: Wrong `PolarQuant` source for value quantization

**Plan note:** The draft plan mentioned `rfsn_v10/quantization/polar_quant.py`. This is
the **wrong file** — it implements a completely different hierarchical atan2/radius
decomposition with 395 lines, `PolarQuantizer` class, `iterative_hierarchical_polar_forward`.

**Correct source:** `mlx-turboquant-main/mlx_turboquant/polar_quant.py` (107 lines,
`PolarQuant` class, rotation + Lloyd-Max codebook lookup). This is what the
`rfsn_v11/quant/value_quant.py` implementation is based on.

---

## Bug 6: `codebooks.npz` not in `package-data`

**File:** `mlx-turboquant-main/mlx_turboquant/data/codebooks.npz`
**Symptom:** If not declared in `[tool.setuptools.package-data]`, the `.npz` file is not
included in the installed wheel. `load_codebook()` raises `FileNotFoundError` at runtime
with no clear indication of the root cause.

**Fix applied in:** `rfsn_v11/pyproject.toml`
```toml
[tool.setuptools.package-data]
"rfsn_v11" = ["quant/data/codebooks.npz"]
```

---

## Bug 7: `tg_acc[32 * 128]` hardcoded in fused attention kernel

**File:** `turboquant-mlx-main/turboquant/fused_v2_attn.py:149`
**Symptom:** `threadgroup float tg_acc[32 * 128]` allocates exactly 32 × 128 floats in
threadgroup memory. For models with `head_dim > 128` (e.g. Llama-3 70B uses D=128, but
Qwen-2.5 32B uses D=128, Mistral uses D=128, Yi uses D=128, Gemma-2 27B uses D=256),
this **silently corrupts threadgroup memory** — no bounds check in Metal threadgroup
arrays.

**Fix applied in:** `rfsn_v11/kernels/fused_sparse_attn.py`
- Two kernel variants: `_fused_attn_d128` (tg_acc[32 × 128]) and `_fused_attn_d256`
  (tg_acc[32 × 256]).
- Python dispatch selects variant from actual `D` at call time.
- Hard `assert D in (128, 256)` before any kernel invocation.

---

## Bug 8: MLX Metal kernel `grid` is total threads, not threadgroup count

**Background:** The draft plan identified a "grid dispatch bug" in `rfsn_v10/kernels.py:112`
claiming `grid=(n, 1, 1)` over-dispatches by 64x. This claim was **incorrect**.

**Reality:** In MLX's `mx.fast.metal_kernel` API, `grid` specifies **total threads**
(analogous to `[encoder dispatchThreads:threadgroupsPerGrid:]` in Apple's raw Metal
API, **not** `dispatchThreadgroups:`). So:
```
grid=(n, 1, 1), threadgroup=(64, 1, 1)
→ n total threads, n/64 threadgroups  ✅ CORRECT
```
The rfsn_v10 code was correct as-is. The fix applied here is simply preserving the
original correct behavior and adding the note to prevent future misidentification.

---

## Non-changes (items confirmed correct, no fix needed)

- `wht64_metal` sign and normalization — uses `1/sqrt(64)` = `1/8`, correctly implemented.
- `_apply_signs_on_the_fly` 128-LRU cache — correct, preserved unchanged.
- `_dequantize_unsigned` symmetric bias — `q - qmax` correctly centers codes.
- `compute_block_hash` SHA256 chain — correct, preserved from vmlx source.
- `_merge_reserved_and_scored_blocks` sink+recent logic — correct, preserved from rfsn_v10.

---

## Experimental gate extensions (v11 additions)

Three new gates added to `ExperimentalConfig` and `require_experimental()`:
- `qjl_prod`: TurboQuant QJL as primary bit-quantizer (centroid resolution failure proven)
- `sub4bit_small_head`: `v_bits < 4` on models with `head_dim < 64`
- `isoquant`: IsoQuant/hybrid-polar-cartesian paths from v10 (not validated in v11)

All three default to `False` and raise `RuntimeError` until explicitly enabled.
