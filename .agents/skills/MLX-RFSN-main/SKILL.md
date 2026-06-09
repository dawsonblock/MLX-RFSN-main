```markdown
# MLX-RFSN-main Development Patterns

> Auto-generated skill from repository analysis

## Overview

This skill covers the core development practices of the **MLX-RFSN-main** repository, a Python project using the Flask framework for server-side logic. The codebase focuses on candidate adapter evaluation, benchmarking, server features, and robust packaging/release processes. You'll learn the project's coding conventions, key workflows, and how to use suggested commands to streamline contributions.

---

## Coding Conventions

**File Naming**
- Use `snake_case` for Python files.
  - Example: `base_adapter.py`, `config.py`

**Import Style**
- Prefer **relative imports** within modules.
  - Example:
    ```python
    from .base import AdapterBase
    from ..utils import load_config
    ```

**Export Style**
- Use **named exports** (explicitly define what is exported).
  - Example:
    ```python
    __all__ = ["AdapterBase", "SpecialAdapter"]
    ```

**Commit Patterns**
- Freeform commit messages, sometimes with prefixes.
- Average commit message length: ~83 characters.

---

## Workflows

### Adapter Fix and Benchmark Update
**Trigger:** When a new candidate adapter is added or an existing one is fixed/enhanced, and benchmarks must be rerun to validate.
**Command:** `/update-adapter-benchmarks`

1. Edit or add adapter files in `rfsn_v11/candidates/` (e.g., `*_adapter.py`, `base.py`).
2. Update `benchmarks/kv_shootout.py` to reflect adapter changes or new evaluation logic.
3. Regenerate benchmark result artifacts in `artifacts/bench/shootout/` (e.g., `results.*`).
4. Commit all changes together.

**Example:**
```python
# rfsn_v11/candidates/my_adapter.py
from .base import AdapterBase

class MyAdapter(AdapterBase):
    ...
```
```bash
python benchmarks/kv_shootout.py --regenerate
git add rfsn_v11/candidates/my_adapter.py benchmarks/kv_shootout.py artifacts/bench/shootout/results.*
git commit -m "Add MyAdapter and update benchmarks"
```

---

### Server Feature or Hardening
**Trigger:** When the server needs a new feature, security improvement, or bugfix, especially around configuration, endpoints, or runtime behavior.
**Command:** `/update-server-feature`

1. Edit `rfsn_v10/server/app.py` and/or `rfsn_v10/server/cli.py` to add/modify server features.
2. Update `rfsn_v10/config.py` for new configuration options.
3. Update or add documentation in `docs/` (e.g., `FEATURE_FLAGS.md`, `RUN_SERVER.md`).
4. Add or update tests in `tests/server/` and/or `tests/runtime/`.
5. Update `scripts/release_gate.py` if release checks are affected.
6. Commit all related changes together.

**Example:**
```python
# rfsn_v10/server/app.py
from flask import Flask

app = Flask(__name__)

@app.route("/health")
def health_check():
    return "OK", 200
```
```bash
git add rfsn_v10/server/app.py docs/RUN_SERVER.md tests/server/test_health.py
git commit -m "Add health check endpoint"
```

---

### Benchmark Promotion Policy Update
**Trigger:** When the criteria for promoting candidates changes, or new prompt suites/categories are introduced.
**Command:** `/update-benchmark-policy`

1. Edit `benchmarks/kv_shootout.py` to add new gates, verdict logic, or prompt suites.
2. Update `docs/CANDIDATE_PROMOTION.md` and related docs to describe new policies.
3. Optionally update tests for new gates or categories.
4. Commit all changes together.

**Example:**
```python
# benchmarks/kv_shootout.py
def new_promotion_gate(candidate):
    return candidate.score > 0.95
```
```bash
git add benchmarks/kv_shootout.py docs/CANDIDATE_PROMOTION.md tests/test_benchmark_gates.py
git commit -m "Add new promotion gate for candidates"
```

---

### Packaging and Release Hardening
**Trigger:** When preparing for a new release, updating Python version, or fixing packaging/deployment issues.
**Command:** `/release-prep`

1. Update `.python-version` and/or `pyproject.toml` for new Python/package versions.
2. Edit `scripts/make_release_zip.py`, `scripts/clean_artifacts.py`, `scripts/release_gate.py` as needed.
3. Update documentation (`docs/README_REALITY.md`, `RUN_SERVER.md`, etc.) for new release notes or instructions.
4. Remove obsolete artifacts or logs.
5. Commit all changes together.

**Example:**
```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.10"
```
```bash
python scripts/clean_artifacts.py
git add .python-version pyproject.toml docs/README_REALITY.md
git commit -m "Update to Python 3.10 and clean release artifacts"
```

---

## Testing Patterns

- **Framework:** Unknown (no standard detected), but tests are in Python.
- **Test File Pattern:** Python test files are in `tests/` directories, named as `test_*.py`.
- **Example Test File:**
    ```python
    # tests/server/test_health.py
    def test_health_check(client):
        response = client.get("/health")
        assert response.status_code == 200
    ```
- **Note:** There is a mention of `*.test.ts` (TypeScript), but main tests appear to be Python-based.

---

## Commands

| Command                    | Purpose                                                      |
|----------------------------|--------------------------------------------------------------|
| /update-adapter-benchmarks | Run adapter fixes/additions and update benchmarks            |
| /update-server-feature     | Add or modify server features, config, docs, and tests       |
| /update-benchmark-policy   | Update benchmark scripts and docs for new promotion policies |
| /release-prep              | Prepare packaging, versioning, and release documentation     |
```
