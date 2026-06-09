# external/

Reference repositories. These are **not runtime dependencies** and are
not packaged into the rfsn_v10 or rfsn_v11 wheels.

| Directory         | Source                          | Role                                      |
|-------------------|---------------------------------|-------------------------------------------|
| `turboquant-mlx/` | turboquant-mlx-main             | TurboQuant V2 attention/rotation ideas    |
| `mlx-turboquant/` | mlx-turboquant-main             | PolarQuant / Lloyd-Max value quant ideas  |
| `vmlx/`           | vmlx-main                       | Serving reference (do not merge)          |
| `turbovec/`       | (add later)                     | RAG/vector memory — not KV cache          |
| `vllm-kivi/`      | (notes only)                    | KIVI NVIDIA reference — not Apple path    |

## Usage

Browse these directories for algorithmic ideas.
Adapters that use these ideas live in `rfsn_v11/candidates/`.
Do not import from these paths at runtime.
