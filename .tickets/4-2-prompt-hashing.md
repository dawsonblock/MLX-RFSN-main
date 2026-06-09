---
id: "4-2"
title: "SHA-256 Hash All User Prompts Before Insert"
owner: Backend
estimate: 2h
priority: P0
epic: Week 4 - Secure Telemetry
status: complete
created: 2026-06-07
labels: [security, privacy, telemetry, p0]
---

# Ticket 4-2: SHA-256 Hash All User Prompts Before Insert

## Description
Hash user prompts with SHA-256 before database insert. Never store raw user text in telemetry database. This is a privacy requirement — we can analyze usage patterns without retaining the actual prompts.

## Exit Criteria
- [ ] DB column `prompt_hash` contains 64-char hex
- [ ] No `prompt_text` or `prompt_raw` column exists (or is empty)
- [ ] Prompts are hashed with SHA-256 before insert
- [ ] Verification: DB dump shows only hashes, no plaintext

## Technical Details

### Database Schema
Before:
```sql
CREATE TABLE telemetry (
    timestamp DateTime,
    prompt_text String,  -- ❌ Privacy violation
    ...
)
```

After:
```sql
CREATE TABLE telemetry (
    timestamp DateTime,
    prompt_hash FixedString(64),  -- ✅ SHA-256 hex
    prompt_length UInt32,         -- ✅ Metadata only
    ...
)
```

### Implementation

**Hash Function** (constant-time, no salt needed for this use case):
```python
import hashlib

def hash_prompt(prompt: str) -> str:
    """SHA-256 hash of prompt, returns 64-char hex."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()
```

**Telemetry Insert**:
```python
# Before
data = {
    "prompt_text": user_prompt,  # ❌
    ...
}

# After
data = {
    "prompt_hash": hash_prompt(user_prompt),  # ✅
    "prompt_length": len(user_prompt),
    ...
}
```

## Verification Steps

```bash
# 1. Insert test prompt
python -c "
from rfsn_v10.telemetry import insert
data = {'prompt': 'secret password 12345', 'model': 'test'}
insert(data)
"

# 2. Query ClickHouse directly
docker exec rfsn-clickhouse clickhouse-client -q "SELECT prompt_hash, prompt_length FROM telemetry WHERE model='test'"

# 3. Verify NO plaintext
docker exec rfsn-clickhouse clickhouse-client -q "SELECT * FROM telemetry WHERE prompt_hash LIKE '%secret%' FORMAT Pretty"
# Should return 0 rows
```

## Migration Strategy

If existing data has `prompt_text`:

1. Create new table with hashed schema
2. Migrate: `INSERT INTO telemetry_new SELECT ..., sha256(prompt_text) ...`
3. Drop old column (or table)
4. Update application code

## Related Files
- `rfsn_v10/telemetry/` — All telemetry insert code
- `rfsn_v10/telemetry/schema.py` — DB schema definition
- `alembic/` — Migration (Ticket 4-4)

## Compliance Note
This addresses GDPR/CCPA requirements for data minimization. We can still:
- Count prompt frequency
- Analyze length distributions
- Correlate with performance metrics

We cannot:
- Read user prompts
- Train on user data
- Share prompts with third parties
