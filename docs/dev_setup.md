# Development Setup

## Supported Python

| Version | Status |
|---------|--------|
| 3.11    | Supported |
| 3.12    | Supported (recommended) |
| 3.13+   | **Not supported** — MLX does not build on 3.13 |
| < 3.11  | Not supported |

## Quick Install (macOS / Apple Silicon)

```bash
# 1. Install pyenv if not present
brew install pyenv

# 2. Install Python 3.12
pyenv install 3.12.8
pyenv local 3.12.8

# 3. Create and activate venv
python -m venv .venv
source .venv/bin/activate

# 4. Check Python version
python scripts/check_python_version.py

# 5. Install with all extras
pip install -U pip setuptools wheel
pip install -e ".[dev,mlx,production]"
```

## Running Tests

```bash
# Fast unit tests (CPU-safe, no MLX required)
PYTHONPATH=. RFSN_BACKEND=numpy pytest -q -m "unit or security"

# All CPU-safe tests
PYTHONPATH=. RFSN_BACKEND=numpy pytest -q -m "not mlx and not slow and not benchmark"

# MLX tests (Apple Silicon only)
PYTHONPATH=. RFSN_BACKEND=mlx pytest -q -m "mlx"

# Release gate (CPU-only)
python scripts/release_gate.py --cpu-only
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RFSN_BACKEND` | `mlx` | `mlx` or `numpy` |
| `RFSN_MODEL_ID` | — | HuggingFace model ID or local path |
| `RFSN_ENABLE_QUANTIZED_KV` | `false` | Enable v10 KV compression |
| `RFSN_ENABLE_SPARSE_DECODE` | `false` | Enable sparse decode (experimental) |
| `RFSN_EXPERIMENTAL_QJL` | `false` | Enable QJL (experimental) |
| `RFSN_EXPERIMENTAL_POLAR` | `false` | Enable PolarQuant (experimental) |
| `RFSN_TELEMETRY_ENABLED` | `false` | Enable telemetry |

All experimental features are disabled by default. They must be explicitly opted in.
