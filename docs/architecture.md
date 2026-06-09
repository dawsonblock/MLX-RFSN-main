# MLX-RFSN Architecture

## Overview

MLX-RFSN is a KV-cache compression + sparse-attention runtime for Apple Silicon.

The repository is structured so that:
1. `rfsn_v10` is the stable baseline
2. `rfsn_v11` is the fusion prototype
3. `external/` holds reference repos (not runtime deps)
4. The winner is chosen by `benchmarks/kv_shootout.py`

## Component roles

```
MLX-RFSN/
  rfsn_v10/           Stable alpha baseline
    quantization/       k8_v5_gs32 and k8_v5_gs64 validated presets
    runtime/            Decode-time sparse attention, KV manager
    server/             FastAPI /v1/chat/completions server
    kernels/            Fused Metal GPU kernels

  rfsn_v11/           Fusion prototype (experimental)
    quant/              WHT key quant, PolarQuant value quant, packing
    attention/          Block-sparse dispatch, adaptive controller
    candidates/         Candidate interface + adapters for shootout
    tests/              MLX-gated tests (pytest.importorskip guards)

  benchmarks/         Benchmark suite
    kv_shootout.py      Candidate comparison — decides the winner
    synthetic_kv_roundtrip.py
    benchmark_kv_cache.py
    ... (others)

  external/           Reference repos — NOT runtime packages
    turboquant-mlx/     TurboQuant V2 rotation + attention ideas
    mlx-turboquant/     PolarQuant / Lloyd-Max value quant ideas
    vmlx/               Serving reference (do not merge)

  docs/               Documentation
  artifacts/          Benchmark outputs
  agent_core/         Telemetry + agent integration
  tools/              CLI utilities
```

## Data flow

```
Prompt
  → tokenizer.encode()
  → model.forward() [MLX]
      → KV cache (compressed by chosen candidate)
      → attention (sparse or dense)
  → logits → decode token
  → repeat
```

## KV cache compression path

```
Keys:
  FP16 → [WHT rotation] → [8-bit scalar quantization] → stored codes + scales

Values:
  FP16 → [unit normalize] → [QR rotation] → [Lloyd-Max codebook] → stored indices + norms

Decode attention:
  Decompress keys → scaled dot-product → softmax → weighted sum of values
```

## Decision rule

The shootout (`benchmarks/kv_shootout.py`) compares:

| Candidate                  | Role                        |
|----------------------------|-----------------------------|
| `mlx_lm_baseline`          | Control (no compression)    |
| `mlx_lm_quantized_kv`      | MLX-LM built-in option      |
| `rfsn_v10_k8_v5_gs32`      | Stable baseline             |
| `rfsn_v10_k8_v5_gs64`      | Stable baseline alt         |
| `rfsn_v11_fusion`          | Fusion prototype            |
| `turboquant_v2`            | TurboQuant rotation path    |
| `polar_reference`          | PolarQuant reference        |

If `rfsn_v10 k8_v5_gs32` wins → keep as stable default.
If `rfsn_v11 fusion` wins → promote, freeze v10 as legacy.
If `turboquant_v2` wins → make TurboQuantV2 the compression candidate.
If `mlx_lm_baseline` wins → stop building custom compression as default.

No emotion. The numbers decide.

## Serving

`external/vmlx` is a serving reference. For local serving use:
- MLX-LM directly
- Open WebUI + MLX-LM
- LiteLLM proxy

Do not merge vMLX into this repo.

## External memory (future)

TurboVec (`external/turbovec/` — not yet present) is for RAG/document memory.
It is not connected to the KV cache. Add only when needed.
