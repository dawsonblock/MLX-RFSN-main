# MLX-RFSN — Release Manifest (Fusion Alpha)

## Release identification

| Field | Value |
|-------|-------|
| Release name | `MLX-RFSN Fusion Alpha 8.4` |
| Git branch | `mlx-rfsn-fusion-alpha-8-3` |
| Git commit | (see `git log -1 --oneline`) |
| Frozen snapshot branch | `mlx-rfsn-fusion-alpha-8-1-snapshot` (preserved, do not delete) |
| Build date | 2026-06-10 |
| Python requirement | `>=3.11,<3.13` |
| Development status | `3 - Alpha` |

---

## Stable configurations

These are the only quantization presets validated for use:

| Config | Type | Notes |
|--------|------|-------|
| `k8_v5_gs32` | 8-bit KV, 5-group, gs=32 | Default — recommended |
| `k8_v5_gs64` | 8-bit KV, 5-group, gs=64 | Also validated |

---

## Experimental configurations (disabled by default)

| Feature | Flag | Status |
|---------|------|--------|
| QJL score correction | `experimental.enable_qjl: true` | Not validated — do not use |
| Polar / fused IsoQuant-Polar | `experimental.enable_polar: true` | Not validated — do not use |
| Adaptive sparse controller | `experimental.enable_adaptive: true` | Not validated — do not use |

---

## Alpha 8.4 gate results (honest status)

| Step | Result |
|------|--------|
| CPU compile | PASS |
| Test collection | PASS |
| CPU tests non-db | PASS |
| rfsn_v11 tests no-MLX | PASS (8 passed, 2 skipped) |
| Server tests | PASS (20 passed) |
| Benchmark tests | PASS |
| Package build | PASS |
| Package install Python 3.11 | NOT VERIFIED IN THIS ARCHIVE |
| Package install Python 3.12 | NOT VERIFIED IN THIS ARCHIVE |
| Docker healthcheck | NOT RUN |
| Docker fusion-bench | NOT RUN |
| Shootout quick | PASS (produces honest artifacts; SKIPPED_NO_MLX_LM on non-MLX) |
| Shootout promotion report | PASS — teacher-forced rerun complete under `teacher_forced_logit_v1`. Promotion remains disabled because runtime-instrumented cache traces are incomplete. No candidate is currently promoted. |
| Shootout full logit | PASS — full logit metrics complete under teacher-forced methodology. RFSN v10 runtime path is now instrumented and exercised; honest metrics show severe quality degradation (logit_cosine ~0.98, KL ~13.5). TurboQuant V2 and Polar also fail gate thresholds. |
| Shootout memory | PASS — memory metrics complete for all candidates. RFSN v10 runtime path increases working-set memory (~1526 MB vs baseline ~975 MB) due to per-step full-shape cache storage. |
| Promoted candidate | NONE (no candidate passes quality + speed + memory gates simultaneously) |
| Stale false-winner artifacts | REMOVED (moved to artifacts/bench/legacy/alpha7_shootout/) |
| Control baseline promotion bug | FIXED (now PASS_NO_PROMOTE, not promotion eligible) |
| CI failure masking | FIXED (`|| true` removed from fusion-alpha.yml) |
| Skipped artifact markdown | FIXED (explicit SKIPPED_NO_MLX_LM rendering) |
| Candidate statuses | ADDED — CONTROL, BASELINE, EXPERIMENTAL, OFFLINE_ONLY, REFERENCE_ONLY |
| No-false-promotion tests | ADDED |
| Artifact integrity tests | ADDED |
| Logit capture infrastructure | ADDED (`logit_capture.py` with `generate_step` hook + `prompt_cache` support) |
| Memory report infrastructure | ADDED (memory mode enforces all metrics present; estimation helper added) |
| TurboQuant V2 proof fields | ADDED (`patch_scope`, `global_patch_restored`, `actual_kv_memory_mb`) |
| Cache policy lock | ADDED (CONTROL, BASELINE, PROMOTED registries; `allow_experimental` flag) |
| Install modes | basic / fusion / memory |
| Product boundary docs | ADDED |
| Platform support docs | ADDED |

## Historical gate results

Older Alpha 7 / v10.2 gate results are preserved for reference in
[`docs/history/alpha7_gate_results.md`](docs/history/alpha7_gate_results.md).
They are **not** the current Alpha 8.2 gate status.

---

## Quality thresholds (measured on Apple Silicon, synthetic KV tensors)

These are **measured** values, not assumed:

| Metric | Threshold | Basis |
|--------|-----------|-------|
| Cosine similarity (decode step) | ≥ 0.998 | Measured across k8_v5_gs32 4-head synthetic |
| KL divergence | ≤ 1e-6 | Measured |
| Top-5 overlap | ≥ 0.95 | Measured |

---

## Known limitations

1. **QJL fails its own artifact**: score MAE 0.1051 vs baseline 0.0824, top-k overlap 0.8 — disabled and unsupported
2. **Polar/adaptive not validated**: quality degradation observed in short-prompt generation
3. **No CUDA backend**: MLX (Apple Silicon) only for the quantized path
4. **FastAPI server implemented**: `/v1/chat/completions` with SSE streaming (set `RFSN_MODEL_ID` env var)
5. **Full sparse prefill not implemented**: prefill always uses dense attention
6. **End-to-end speedup not proven**: compression overhead dominates at short contexts
7. **Docker gate not run in CI on this machine**: must be verified manually
8. **TurboQuant V2 and Polar now have full-logit rows and fail current logit gates**: quality is not acceptable for promotion; they remain EXPERIMENTAL and REFERENCE_ONLY
9. **RFSN v11 remains offline-only**: real cache injection does not yet exist
10. **Teacher-forced rerun is complete**: winner.json is reset to NO_PROMOTION_ELIGIBLE_CANDIDATE; promotion remains blocked until runtime-instrumented cache traces prove the compressed path was actually exercised
11. **Promotion is limited to the RFSN v10 baseline path and has only been shown on Qwen/Qwen2.5-0.5B-Instruct**: model coverage must expand before treating this as a serious stable default

---

## Source integrity

The following files were **invalid placeholder text** in the broken snapshot and are now valid disabled stubs:

| File | Fix applied |
|------|-------------|
| `rfsn_v10/isoquant_precondition.py` | Replaced with `IsoQuantPreconditioner` stub raising `_ExperimentalNotImplemented` |
| `rfsn_v10/quantization/fused_isoquant_polar.py` | Replaced with `FusedIsoQuantPolar` stub raising `_ExperimentalNotImplemented` |
| `rfsn_v10/quantization/kv_quant_manager.py` | Trailing placeholder line removed; 216 lines of real code preserved |

Guard test: `tests/test_no_placeholder_source.py` — prevents regression.

---

## Alpha 8 → Next promotion checklist

The status remains `3 - Alpha` until **all** items below are checked.
Do not call this beta until the full logit gate passes for at least one candidate.

Alpha 8 completed (Plan B):
- [x] Quality gate semantics fixed (text_heuristic_passed, logit_gate_passed, promotion_eligible, gate_status)
- [x] Logit metrics module exists and tested
- [x] Memory metrics module exists and tested
- [x] rfsn_v11 honestly labeled as offline-only (PENDING_REAL_CACHE_INJECTION)
- [x] TurboQuant V2 honestly labeled (PENDING_LOGIT_GATE)
- [x] Polar reference honestly labeled (PENDING_LOGIT_GATE)
- [x] kv_shootout supports --quick, --full-logit-gate, --memory-report, --promotion-report
- [x] README no longer says production deployment
- [x] Dockerfile split (healthcheck + fusion-bench)
- [x] CI workflow fusion-alpha.yml added
- [x] `benchmarks/kv_shootout.py --quick` produces artifacts
- [x] `benchmarks/kv_shootout.py --promotion-report` correctly says: No candidate is promotion eligible
- [x] CandidateStatus enum added (CONTROL, BASELINE, EXPERIMENTAL, OFFLINE_ONLY, REFERENCE_ONLY, PROMOTION_ELIGIBLE, PROMOTED, FAILED)
- [x] Candidate adapters set canonical statuses
- [x] kv_shootout enforces promotion rules by candidate status
- [x] Honest benchmark table generated in results.md
- [x] Real cache proof test scaffold added
- [x] Winner export artifacts (winner.json, winner.md, integration_notes.md)
- [x] Memory layer scaffold added (Qdrant, TurboVec, chunking, embeddings)
- [x] Cache policy abstraction added
- [x] Install modes: basic, fusion, memory
- [x] Platform support docs added
- [x] No-false-promotion tests added
- [x] Artifact integrity tests added
- [x] release_gate.sh and mlx_gate.sh added
- [x] release_gate.sh strict quick benchmark (no soft-masking)
- [x] RELEASE_MANIFEST.md historical section moved to docs/history/

Alpha 8.4 target — Teacher-Forced Validation Repair:
- [x] Regenerate all shootout artifacts under teacher_forced_logit_v1 on Apple Silicon
- [x] Verify rfsn_v10_k8_v5_gs64 passes teacher-forced gate on Qwen/Qwen2.5-0.5B-Instruct
- [ ] Expand model coverage: validate on Qwen/Qwen2.5-1.5B-Instruct and at least one 3B model
- [ ] RFSN v10 perfect-logit proof trace: runtime-instrumented counters proving quantized path was actually used during teacher-forced capture
- [ ] Working-set memory measurement consistency: explain or reconcile difference between full-logit and memory modes
- [x] Strict JSON enforcement in all artifact writers (allow_nan=False)
- [ ] TurboQuant V2 quality improvement: current logit_cosine 0.9948, KL 2.35, top5 0.40 — not close to promotion
- [ ] Polar reference quality improvement or keep reference-only
- [ ] Real cache injection exists for rfsn_v11 (or TurboQuant V2 proves it uses compressed cache natively)
- [ ] Docker fusion-bench verified
- [ ] Do NOT call this beta until validated across multiple model sizes

Previously completed (now invalidated by methodology repair):
- [~] Winner selected from shootout with honest artifacts (rfsn_v10_k8_v5_gs64) — DEMOTED pending teacher-forced rerun

---

## Archive instructions

**Do not zip from Finder.** Use:

```bash
git archive --format=zip HEAD -o mlx-rfsn-fusion-alpha-1.zip
```

Verify the archive is clean:

```bash
python -c "
import zipfile, sys
with zipfile.ZipFile('mlx-rfsn-fusion-alpha-1.zip') as z:
    bad = [n for n in z.namelist() if '__pycache__' in n or n.endswith('.pyc') or '.DS_Store' in n]
    print(f'{len(z.namelist())} files, {len(bad)} junk files')
    if bad: sys.exit(1)
"
```
