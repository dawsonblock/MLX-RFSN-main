# Packaged App Integrity Gate

Date: 2026-05-12

## Root Cause

The Python/Electron release gate could invalidate `/Applications/vMLX.app`
while trying to verify it. The local `twine` executable on `PATH` used a
shebang pointing at the app-bundled Python:

```text
#!/Applications/vMLX.app/Contents/Resources/bundled-python/python/bin/python3
```

Running `twine check dist` through that stale console script wrote
`__pycache__` files into the signed app bundle. `codesign --verify --deep
--strict` then failed correctly because the sealed resource tree had changed.

The live family audit harness also launched packaged Python without `-B` and
from the source checkout cwd, which created two false-proof risks:

- signed app mutation from bytecode writes
- repo source shadowing the packaged `vmlx_engine`

## Fix

- `panel/scripts/release-gate-python-app.py`
  - Selects repo/current-Python `twine` first when the module is available.
  - Runs fallback console-script `twine` with external `PYTHONPYCACHEPREFIX`.
  - Runs packaged Python probes with external `PYTHONPYCACHEPREFIX`.
  - Fails before codesign if any packaged `__pycache__` appears.

- `tests/cross_matrix/run_production_family_audit.py`
  - Live model servers now launch with `-B -s -P`.
  - Live server cwd is `/tmp/vmlx_family_audit`, not the repo root.
  - Child env clears `PYTHONPATH` and disables bytecode writes.
  - Source JANG injection is explicit through `VMLINUX_AUDIT_USE_SOURCE_JANG=1`.

## Verification

- Release gate:
  - `docs/internal/release-gates/20260512_145338/SUMMARY.md`
  - PASS: dist metadata, panel tests, typecheck, bundled hashes, packaged
    imports, packaged pycache clean, strict codesign.
  - WARN: `spctl` only, expected for local ad-hoc install.

- Installed live ZAYA-VL JANGTQ4:
  - `docs/internal/release-gates/20260512_post_mpp_full_matrix/live_zaya_vl_jangtq4_installed_after_harness_fix.json`
  - PASS: 14/14 live checks.
  - Health: `kernel_type=turboquant_codebook_mpp_nax`, MPP/NAX active.
  - Cache repeat: `paged+zaya_cca`, `cached_tokens=32`.

- Installed live Qwen3.6 JANGTQ4:
  - `docs/internal/release-gates/20260512_post_mpp_full_matrix/live_qwen36_moe_tq4_installed_after_harness_fix.json`
  - PASS: 13/13 live checks.
  - Health: `kernel_type=turboquant_codebook_mpp_nax`, MPP/NAX active.
  - Cache repeat: `paged+ssm`, `cached_tokens=18`.

After both installed live rows, packaged app pycache count remained `0` and
strict codesign still passed.
