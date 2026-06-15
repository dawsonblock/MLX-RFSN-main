# Promotion Gate Report

## No candidate is promotion eligible

**Status:** NO_NATIVE_EVIDENCE_YET

**Official promoted candidate: NONE**

No native Apple Silicon evidence bundle exists. Promotion requires:
- Non-empty `token_sequence_hash`
- Runtime-instrumented cache traces (`packed_bytes_written > 0`)
- `full_history_materialization_calls == 0`
- `measurement_kind == MEASURED` (actual tensor accounting, not estimates)

**Methodology:** `teacher_forced_logit_v1`  
**Promotion allowed:** False  
**Schema version:** 2.0  
