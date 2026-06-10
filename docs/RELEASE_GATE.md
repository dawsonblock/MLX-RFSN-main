# Release Gate

This document defines the minimum bar for tagging a stable release of RFSN v10.

---

## Checklist

Run this checklist before tagging any `v10.x.y` release.

### 1. Unit tests

```bash
python -m pytest tests/ -x -q
```

All tests must pass. Zero failures, zero errors.

### 2. Import smoke test

```bash
python -c "import rfsn_v10; print(rfsn_v10.__version__)"
```

### 3. Config validation

```bash
RFSN_MODEL_ID=test rfsn-config-check
```

No configuration errors.

### 4. LAN guard

```bash
python -c "
from rfsn_v10.config import ServerConfig
from pydantic import ValidationError
try:
    ServerConfig(host='0.0.0.0', require_api_key=False)
    raise AssertionError('LAN guard did not fire')
except ValidationError as e:
    assert 'LAN mode' in str(e), f'Wrong error: {e}'
    print('LAN guard OK')
"
```

### 5. KV flag compat

```bash
python -m pytest tests/test_kv_flag_compat.py tests/test_lan_guard.py -q
```

### 6. Server health endpoint

Start the server in a background process and confirm `/health` returns 200:

```bash
RFSN_MODEL_ID=mlx-community/Qwen2.5-0.5B-Instruct-4bit rfsn-server &
sleep 5
curl -sf http://127.0.0.1:8000/health | python -m json.tool
kill %1
```

### 7. Metrics endpoint

```bash
curl -sf http://127.0.0.1:8000/metrics | python -m json.tool
```

Response must include `requests_total`, `last_latency_ms`, `model_loaded`.

### 8. Benchmark gate (full run)

```bash
rfsn-bench
```

The winning candidate must have:
- `promotion_eligible=true`
- `logit_gate_passed=true`
- `memory_gate_passed=true`
- real cache path used (`real_cache_used=true`)
- `gate_status="PASS"`

If the verdict is `PROMOTE`, update `docs/CANDIDATE_PROMOTION.md` status lines.
If no candidate qualifies, the promotion report must say so honestly.

### 9. Proof regression

```bash
python scripts/check_proof_regression.py
```

No regressions vs stored baseline.

### 10. Release ZIP

```bash
python scripts/make_release_zip.py
```

ZIP must be created without errors and must not include `.git/`, `artifacts/`,
`__pycache__`, or `.env` files.

---

## Version bump

1. Update `rfsn_v10/_version.py`
2. Update `RELEASE.md` with a changelog entry
3. Tag: `git tag v10.x.y && git push --tags`

---

## Go / No-Go criteria

| Criterion | Required |
|-----------|----------|
| All unit tests pass | Yes |
| LAN guard fires correctly | Yes |
| KV compat tests pass | Yes |
| Server starts and /health returns 200 | Yes |
| /metrics endpoint returns valid JSON | Yes |
| Benchmark quality gate passes | Yes |
| Proof regression check passes | Yes |
| Release ZIP builds cleanly | Yes |
| No uncommitted changes | Yes |
