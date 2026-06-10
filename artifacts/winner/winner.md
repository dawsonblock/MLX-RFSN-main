# Winner Report

## No winner

**Status:** NO_PROMOTION_ELIGIBLE_CANDIDATE

**Reason:** Teacher-forced logit gate has been introduced; candidates must be revalidated under the corrected methodology before promotion.

**Current best baseline:** rfsn_v10_k8_v5_gs64

**Important caveats:**
- This is an alpha-level build, not a beta release.
- The teacher-forced logit comparison is the corrected methodology. All prior promotion artifacts based on earlier methods are considered stale.
- Promotion has only been shown on one model (0.5B). Before treating this as a serious stable default, validate on at least Qwen/Qwen2.5-1.5B-Instruct and one 3B model.
- Working-set memory is measurement-mode dependent; actual KV cache bytes are the stable compression proof.
- TurboQuant V2 and Polar remain experimental / reference-only; do not tune thresholds to pass them.
- RFSN v11 remains offline-only until real cache injection exists.
