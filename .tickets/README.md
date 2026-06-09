# RFSN v10 Tickets

This directory contains executable tickets for the RFSN v10 "Get-It-Done" repair plan.

## Structure

Each ticket is a Markdown file with YAML frontmatter:

```markdown
---
id: "4-1"
title: "Wrap ClickHouse Client in TLS + RFSN-Auth Header"
owner: Backend
estimate: 3h
priority: P0
epic: Week 4 - Secure Telemetry
status: active
created: 2026-06-07
labels: [security, telemetry, tls, p0]
---

# Ticket Title
...
```

## Status Values

- `complete` — Ticket finished, exit criteria met
- `active` — Currently being worked
- `verify` — Needs verification/audit
- `not-started` — Not yet begun
- `blocked` — Blocked by dependency

## Priority Levels

- `P0` — Critical, blocks release (security, correctness)
- `P1` — Important, required for v10-beta
- `P2` — Nice to have, can defer

## Ticket Files

### Active / Critical
| ID | Title | Priority | Estimate |
|----|-------|----------|----------|
| 4-1 | [ClickHouse TLS](./4-1-clickhouse-tls.md) | P0 | 3h |
| 4-2 | [Prompt SHA-256 Hashing](./4-2-prompt-hashing.md) | P0 | 2h |
| 4-3 | [Retry Queue + SIGTERM Flush](./4-3-retry-queue.md) | P0 | 4h |
| 4-4 | [Alembic Migrations](./4-4-alembic-migrations.md) | P1 | 2h |
| 3-2 | [Poetry Migration](./3-2-poetry-migration.md) | P1 | 3h |
| 7-2 | [Strict Config Validation](./7-2-strict-config-validation.md) | P1 | 3h |

### Completed (Reference)
Tickets 0-1 through 6-2 are complete — see [TICKETS.md](../TICKETS.md) for full list.

## Importing to GitHub Issues

Use the import script:

```bash
# 1. Authenticate with gh CLI
gh auth login

# 2. Import all tickets
python .tickets/import_to_github.py

# 3. Or import specific tickets
python .tickets/import_to_github.py --tickets 4-1,4-2,4-3

# 4. Dry run (preview only)
python .tickets/import_to_github.py --dry-run
```

## Creating New Tickets

1. Copy template: `cp .tickets/TEMPLATE.md .tickets/X-Y-description.md`
2. Fill in frontmatter and content
3. Update [TICKETS.md](../TICKETS.md) summary table

## Ticket Template

See [TEMPLATE.md](./TEMPLATE.md) for new ticket template.
