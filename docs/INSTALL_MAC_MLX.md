# Install on macOS (Apple Silicon)

## Requirements

- macOS 13+ (Ventura or later)
- Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12 (not 3.13+)

## Step-by-step

### 1. Install pyenv

```bash
brew install pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
source ~/.zshrc
```

### 2. Install Python 3.12

```bash
pyenv install 3.12.8
pyenv local 3.12.8
python --version  # should print 3.12.8
```

### 3. Clone and install

```bash
git clone https://github.com/dawsonblock/MLX-RFSN-main
cd MLX-RFSN-main
python -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -e ".[dev,mlx,production]"
```

### 4. Verify

```bash
python scripts/check_python_version.py
python scripts/release_gate.py --cpu-only
```

### 5. Start the server

```bash
rfsn-config-check
rfsn-server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
# Then open: http://127.0.0.1:8000/dashboard
```

## LAN access (iPhone/iPad)

```bash
RFSN_HOST=0.0.0.0 RFSN_REQUIRE_API_KEY=true RFSN_API_KEY=your-secret-here \
  rfsn-server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

Then open `http://<your-mac-ip>:8000/dashboard` from your phone.

## Troubleshooting

**"Unsupported Python"**: Run `pyenv local 3.12.8` and reactivate venv.

**MLX import fails**: Make sure you are on Apple Silicon and macOS 13+.

**Model not found**: Use a full HuggingFace ID, e.g. `mlx-community/Qwen2.5-0.5B-Instruct-4bit`.

**Server blocks on first request**: Model loads lazily on first request. This is expected.
