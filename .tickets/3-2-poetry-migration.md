---
id: "3-2"
title: "Replace Shell Scripts with Poetry Tasks"
owner: DevOps
estimate: 3h
priority: P1
epic: Week 3 - Reproducibility
status: not-started
created: 2026-06-07
labels: [build, poetry, devops, p1]
---

# Ticket 3-2: Replace Shell Scripts with Poetry Tasks

## Description
Migrate build system from setuptools/pip to Poetry. Replace shell scripts with Poetry tasks. Remove Homebrew dependencies from CI.

## Current State
- Using `setuptools` with `pip install -e ".[mlx,dev]"`
- `pyproject.toml` has `[build-system]` = setuptools
- Scripts in `scripts/` directory

## Target State
- Using `poetry install`
- Poetry scripts for common tasks
- No shell scripts (or minimal wrappers)
- No Homebrew in CI

## Exit Criteria
- [ ] `pyproject.toml` uses `[tool.poetry]` format
- [ ] `poetry.lock` committed
- [ ] Poetry tasks for: test, lint, benchmark, docs
- [ ] No shell scripts in `scripts/` (or marked deprecated)
- [ ] Fresh clone on clean VM: `poetry install && pytest` passes
- [ ] No Homebrew dependencies in CI

## Migration Plan

### Step 1: Add Poetry Configuration
```toml
[tool.poetry]
name = "rfsn-v10"
version = "0.10.0"
description = "RFSN v10 — Quantized KV-cache + sparse-attention runtime"
readme = "README.md"
packages = [{include = "rfsn_v10"}, {include = "agent_core"}]

[tool.poetry.dependencies]
python = "==3.11.*"
numpy = "==1.26.4"
mlx = {version = "==0.21.1", optional = true}
torch = {version = "==2.3.1", optional = true}
pydantic = ">=2.5"
pyyaml = ">=6.0.1"
requests = ">=2.31"
matplotlib = ">=3.8"

[tool.poetry.group.dev.dependencies]
pytest = "==8.1.*"
pytest-timeout = "==2.3.*"
pytest-asyncio = "==0.23.*"
pytest-cov = "==5.0.*"
ruff = "==0.4.*"
mypy = "==1.8.*"

[tool.poetry.scripts]
rfsn-test = "scripts.poetry_tasks:test"
rfsn-lint = "scripts.poetry_tasks:lint"
rfsn-benchmark = "scripts.poetry_tasks:benchmark"
rfsn-serve = "rfsn_v10.cli:main"
```

### Step 2: Create Poetry Tasks Module
```python
# scripts/poetry_tasks.py
def test():
    import subprocess
    subprocess.run(["pytest", "-q", "--cov=rfsn_v10"], check=True)

def lint():
    import subprocess
    subprocess.run(["ruff", "check", "."], check=True)
    subprocess.run(["mypy", "rfsn_v10"], check=True)

def benchmark():
    import subprocess
    subprocess.run(["python", "benchmarks/run_deterministic.py"], check=True)
```

### Step 3: Update CI
```yaml
# Before
- run: pip install -e ".[mlx,dev]"

# After
- run: pip install poetry
- run: poetry install --with dev --extras "mlx"
- run: poetry run pytest
```

## Verification Steps

```bash
# 1. Fresh clone in clean environment
docker run -it python:3.11 bash
git clone <repo>
cd rfsn_v10_corrected_full

# 2. Install poetry
pip install poetry

# 3. Install project
poetry install --with dev --extras "mlx"

# 4. Run tests
poetry run pytest -q

# 5. Use poetry scripts
poetry run rfsn-test
poetry run rfsn-lint
```

## Decision Required
**Question**: Should we migrate to Poetry or update the repair plan to reflect pip-based workflow?

**Trade-offs**:
- Poetry: Better lock files, cleaner scripts, modern Python packaging
- Pip: Simpler, already works, less migration risk

**Recommendation**: Given the project is already functional with pip, consider:
1. Keep pip for now
2. Add `requirements.txt` with exact pins
3. Update repair plan to reflect actual state

Or, if Poetry is preferred:
1. Create separate ticket for Poetry migration post-v10
2. Keep current pip-based system stable

## Related Files
- `pyproject.toml` — Update build-system
- `scripts/poetry_tasks.py` — Create
- `.github/workflows/*.yml` — Update all jobs
- `poetry.lock` — Generate and commit
