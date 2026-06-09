---
id: "4-4"
title: "Alembic Migrations for Telemetry Schema"
owner: DevOps
estimate: 2h
priority: P1
epic: Week 4 - Secure Telemetry
status: not-started
created: 2026-06-07
labels: [database, migrations, devops, p1]
---

# Ticket 4-4: Alembic Migrations for Telemetry Schema

## Description
Implement database schema versioning with Alembic. Enable reproducible schema changes and CI-based migration checks.

## Exit Criteria
- [ ] `alembic/` directory with `env.py`, `versions/`
- [ ] Initial migration creates telemetry tables
- [ ] `alembic upgrade head` runs on empty DB without manual SQL
- [ ] Version file committed to repo
- [ ] CI runs `alembic check` to detect unmigrated changes

## Technical Details

### Setup Commands
```bash
# 1. Install alembic
pip install alembic

# 2. Initialize
alembic init alembic

# 3. Configure alembic.ini
# - Set sqlalchemy.url from env var
# - Set script_location = alembic

# 4. Create initial migration
alembic revision -m "initial telemetry schema"
# Edit migration to create tables

# 5. Apply
alembic upgrade head
```

### Migration Example
```python
# alembic/versions/001_initial_telemetry.py
revision = '001'
down_revision = None

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.create_table(
        'telemetry',
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('prompt_hash', sa.String(64), nullable=False),
        sa.Column('prompt_length', sa.Integer(), nullable=True),
        sa.Column('model', sa.String(50), nullable=True),
        sa.Column('backend', sa.String(20), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True),
    )
    op.create_index('idx_timestamp', 'telemetry', ['timestamp'])
    op.create_index('idx_model', 'telemetry', ['model'])

def downgrade():
    op.drop_table('telemetry')
```

## Verification Steps

```bash
# 1. Fresh database
docker run -d --name test-clickhouse -p 8123:8123 clickhouse/clickhouse-server

# 2. Run migrations
alembic upgrade head

# 3. Verify tables exist
docker exec test-clickhouse clickhouse-client -q "SHOW TABLES"
# Should list 'telemetry'

# 4. Check migration status
alembic current
# Should show current revision

# 5. Downgrade test
alembic downgrade -1
alembic upgrade head
```

## CI Integration
```yaml
# .github/workflows/ci.yml
- name: Check migrations
  run: |
    alembic check  # Fails if model changes without migration
    alembic upgrade head
    alembic downgrade base
    alembic upgrade head
```

## Related Files
- `alembic/` — New directory (to create)
- `rfsn_v10/telemetry/models.py` — SQLAlchemy models (to create)
- `requirements.txt` — Add alembic

## Notes
- ClickHouse doesn't have traditional migrations; consider using ClickHouse's `ALTER` commands
- Alternative: Use `clickhouse-migrations` tool
