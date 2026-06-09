---
id: "7-2"
title: "Single Pydantic-Validated Config.yaml with Strict Mode"
owner: Backend
estimate: 3h
priority: P1
epic: Week 7 - House-Cleaning
status: complete
created: 2026-06-07
labels: [config, pydantic, validation, p1]
---

# Ticket 7-2: Single Pydantic-Validated Config.yaml with Strict Mode

## Description
Create a single, strictly-validated configuration file. Unknown keys must raise `ValidationError`. Centralize all settings, eliminate environment variable sprawl.

## Current State
- `configs/default_runtime.yaml` exists
- `rfsn_v10/config.py` handles config loading
- Unknown keys may be silently ignored

## Target State
- `configs/config.yaml` (rename from default_runtime.yaml)
- Pydantic model with `extra='forbid'`
- All settings in one place
- Fast failure on invalid config

## Exit Criteria
- [ ] `configs/config.yaml` with strict Pydantic validation
- [ ] Unknown keys raise `pydantic.ValidationError`
- [ ] All settings centralized (backend, telemetry, limits)
- [ ] CI test: invalid key fails fast
- [ ] Documentation of all config options

## Technical Details

### Pydantic Model
```python
# rfsn_v10/config.py
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

class BackendConfig(BaseModel):
    name: Literal["metal", "numpy", "cuda"] = "metal"
    fallback: bool = True

class TelemetryConfig(BaseModel):
    enabled: bool = True
    endpoint: str = "http://localhost:8123"
    flush_interval_ms: int = 1000
    max_queue_size: int = 10000

class LimitsConfig(BaseModel):
    max_context: int = 32768
    max_batch: int = 16
    memory_threshold: float = 0.9

class RFSNConfig(BaseModel, extra='forbid'):  # Strict mode
    backend: BackendConfig = BackendConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    limits: LimitsConfig = LimitsConfig()

    @classmethod
    def load(cls, path: str = "configs/config.yaml") -> "RFSNConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
```

### Config File
```yaml
# configs/config.yaml
backend:
  name: metal
  fallback: true

telemetry:
  enabled: true
  endpoint: https://clickhouse:8443
  flush_interval_ms: 1000
  max_queue_size: 10000

limits:
  max_context: 32768
  max_batch: 16
  memory_threshold: 0.9
```

## Strict Mode Test
```python
# tests/test_config.py
def test_unknown_key_fails():
    import tempfile
    import yaml
    from rfsn_v10.config import RFSNConfig
    from pydantic import ValidationError

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml') as f:
        yaml.dump({"unknown_key": "value"}, f)
        f.flush()
        with pytest.raises(ValidationError):
            RFSNConfig.load(f.name)
```

## Verification Steps

```bash
# 1. Test valid config
pytest tests/test_config.py -v

# 2. Test invalid config
echo "invalid_key: test" > /tmp/bad_config.yaml
python -c "
from rfsn_v10.config import RFSNConfig
try:
    RFSNConfig.load('/tmp/bad_config.yaml')
    print('FAILED: Should have raised ValidationError')
except Exception as e:
    print(f'OK: Raised {type(e).__name__}')
"
```

## Migration Strategy
1. Create new `RFSNConfig` class with strict validation
2. Add `configs/config.yaml` (copy from `default_runtime.yaml`)
3. Update all code to use `RFSNConfig.load()`
4. Deprecate `default_runtime.yaml`
5. Remove old config loading code

## Related Files
- `rfsn_v10/config.py` — Update
- `configs/config.yaml` — Create (or rename)
- `configs/default_runtime.yaml` — Deprecate
- `tests/test_config.py` — Add strict mode test

## Notes
- Strict mode prevents typos in config (e.g., `telemtry` vs `telemetry`)
- Consider JSON Schema export for IDE autocomplete
- Document all options in `docs/configuration.md`
