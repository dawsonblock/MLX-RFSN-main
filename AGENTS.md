# Project Notes

## Branch
`mlx-rfsn-fusion-alpha-8-3`

## Test Commands
- Collect all tests: `pytest --collect-only -q`
- Core cache tests: `pytest rfsn_v10/cache/tests/ -q`
- Generation tests: `pytest tests/test_generation.py -q`
- Alembic migrations (skips gracefully if deps missing): `pytest tests/test_alembic_migrations.py -q`
- Identity tests: `pytest rfsn_v10/cache/tests/test_identity.py -q`
- Full cache + generation + migrations: `pytest tests/test_generation.py tests/test_alembic_migrations.py rfsn_v10/cache/tests/ -q`

## Key Architecture
- Dense/chunked prefill → encode each K/V block once → discard complete dense history → direct packed QK → online softmax → direct packed SV → bounded staging or dense tail only.
- `requantized_token_count == 0` invariant for every generation.
- `PackedBlock` is immutable after creation (V3 format).

## Recent Commits (Repair Plan)
1. `repair/ci-tests-db-isolation` — Fix CI, test collection, session tests, generator tests, DB isolation.
2. `repair/trim-geometry-positions` — Disable unsafe trim, freeze cache geometry, validate position ownership.
3. `repair/packedblock-v3` — Replace PackedBlock V2 with validated V3 format.
4. `repair/vector-aligned-k8v5` — Replace global packing with vector-aligned K8/V5 plus active WHT/sign.
5. `repair/numpy-oracle-attention` — Expand NumPy oracle to cover packing and direct packed attention.

## Important Invariants
- `CartesianCodec` defaults: `use_wht=True`, `sign_seed=42`, `group_size` must be a multiple of 64.
- `QuantizedLayerCache.trim()` raises `NotImplementedError` in this release.
- `validate_block_positions()` enforces monotonic, non-overlapping block positions.
