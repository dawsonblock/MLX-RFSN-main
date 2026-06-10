# MLX-RFSN Product Boundary

## What MLX-RFSN Is

MLX-RFSN is a KV-cache compression research and benchmarking system for Apple Silicon.

It does three jobs:

1. **Compare KV-cache compression candidates.**
   It runs multiple compression methods (RFSN v10, RFSN v11, TurboQuant V2, Polar reference) against the same models and prompts and produces comparable metrics.

2. **Prove memory/speed/quality tradeoffs on Apple Silicon.**
   It measures real memory usage, decode throughput, and logit-level fidelity on MLX-equipped Macs.

3. **Export the winning candidate as an integration layer for MLX-LM/vMLX.**
   When a candidate passes all quality gates with real cache injection, it can be promoted and wired into upstream serving code.

## What MLX-RFSN Is Not

- **Not a general AI server.** It does not expose chat completions, function calling, or multi-user serving. Use MLX-LM, vMLX, or an Open WebUI + LiteLLM stack for that.
- **Not a full vector database.** It does not replace Qdrant, Chroma, or similar systems. It may benchmark external memory layers, but it is not one.
- **Not a production deployment platform.** It is an alpha-stage research tool. Do not deploy it as-is to serve users.

## Stable Components

| Component | Role | Status |
|-----------|------|--------|
| `rfsn_v10` | Baseline KV quantization (k8_v5_gs32, k8_v5_gs64) | Stable |
| `rfsn_v11` | Fusion candidate (asymmetric K/V, WHT + PolarQuant) | Experimental, offline-only |
| `kv_shootout` | Decision benchmark and promotion gate | Active |
| `external/turboquant-mlx` | Reference implementation (TurboQuant V2) | Reference |
| `external/mlx-turboquant` | Reference implementation (PolarQuant) | Reference |
| `external/vmlx` | Serving reference | External |

## Exit Condition

Nobody reading this repository should think it is:
- Production AGI
- A finished model runtime
- A full server stack

If any documentation implies otherwise, that documentation is a bug.
