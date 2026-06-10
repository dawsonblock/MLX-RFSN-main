# Platform Support

## Feature Matrix

| Feature | Linux CPU | Mac Apple Silicon | Windows NVIDIA |
|---------|-----------|-------------------|----------------|
| rfsn_v10 CPU tests | Yes | Yes | Yes |
| rfsn_v11 MLX tests | Skip | Yes | No |
| MLX-LM generation | No | Yes | No |
| vMLX | No | Yes | No |
| TurboVec memory | Yes | likely | Yes |
| vLLM / KIVI | No | No | later (NVIDIA only) |

## Install Modes by Platform

```bash
# Any platform — core only, no MLX, no memory
pip install -e ".[basic]"

# macOS Apple Silicon — MLX + KV compression benchmarks
pip install -e ".[fusion]"

# Any platform — external memory (Qdrant, sentence-transformers)
pip install -e ".[memory]"

# Full development stack
pip install -e ".[fusion,memory,dev]"
```

## CI Behavior

- Linux CI runs `basic` + `dev` only.
- macOS CI runs `fusion` + `dev` on Apple Silicon runners.
- Windows CI runs `basic` + `dev` only.
