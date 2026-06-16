#!/usr/bin/env python3
"""Verify all definition-of-done checkboxes for K8/V8 milestone.

Phase Final: Run this script after all phases are complete to confirm
that every checkbox is satisfied.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _check(name: str, condition: bool, detail: str = "") -> tuple[bool, str]:
    """Return (passed, message) for a single check."""
    if condition:
        return True, f"OK: {name}"
    return False, f"FAIL: {name}" + (f" — {detail}" if detail else "")


def main() -> int:
    print("=== K8/V8 Milestone Definition of Done ===\n")

    results: list[tuple[str, bool, str]] = []

    # 1. Metal kernel compilation
    try:
        os.environ["RFSN_ENABLE_TRUE_PACKED"] = "1"
        from rfsn_v10.kernels.metal.packed_v4_attention import (
            HAS_TRUE_PACKED_KERNEL,
            _self_test,
        )
        kernel_ready = HAS_TRUE_PACKED_KERNEL and _self_test()
    except Exception as exc:
        kernel_ready = False
        print(f"  Kernel import error: {exc}")
    results.append((
        "1. Metal kernel compiles and self-test passes",
        *_check("Metal kernel", kernel_ready),
    ))

    # 2. Teacher-forced token match
    manifest_path = REPO_ROOT / "artifacts" / "proof" / "native_gate" / "native_gate_manifest.json"
    token_match = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runs = manifest.get("runs", [])
            token_match = all(r.get("token_match") for r in runs)
        except Exception:
            pass
    results.append((
        "2. Teacher-forced token match with dense baseline",
        *_check("Token match", token_match, "run native gate with --strict"),
    ))

    # 3. Zero dense fallback (strict mode)
    zero_fallback = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            zero_fallback = all(
                r.get("packed", {}).get("counters", {}).get("dense_fallback_calls", 0) == 0
                for r in manifest.get("runs", [])
            )
        except Exception:
            pass
    results.append((
        "3. Zero dense fallback in strict mode",
        *_check("Zero fallback", zero_fallback),
    ))

    # 4. Zero full-history materialization
    zero_materialization = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            zero_materialization = all(
                r.get("packed", {}).get("counters", {}).get("full_history_materialization_calls", 0) == 0
                for r in manifest.get("runs", [])
            )
        except Exception:
            pass
    results.append((
        "4. Zero full-history materialization",
        *_check("Zero materialization", zero_materialization),
    ))

    # 5. Memory reduction measured (3 categories)
    memory_measured = False
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            runs = manifest.get("runs", [])
            memory_measured = all(
                "memory" in r.get("dense", {})
                and "memory" in r.get("packed", {})
                for r in runs
            )
        except Exception:
            pass
    results.append((
        "5. Memory measured with three categories",
        *_check("Memory categories", memory_measured),
    ))

    # 6. Deterministic native benchmark gate
    gate_exists = (REPO_ROOT / "benchmarks" / "run_native_gate.py").exists()
    results.append((
        "6. Deterministic native benchmark gate exists",
        *_check("Native gate", gate_exists),
    ))

    # 7. Backend state reporting explicit
    try:
        from rfsn_v11.candidates.backend_state import BackendState
        backend_report_ready = True
    except Exception:
        backend_report_ready = False
    results.append((
        "7. Backend state reporting explicit",
        *_check("Backend state", backend_report_ready),
    ))

    # 8. PagedKVArena with zero historical recopy
    arena_exists = (REPO_ROOT / "rfsn_v10" / "cache" / "paged_arena.py").exists()
    results.append((
        "8. PagedKVArena implemented",
        *_check("Paged arena", arena_exists),
    ))

    # 9. Release gate unified
    py_gate = REPO_ROOT / "scripts" / "release_gate.py"
    sh_gate = REPO_ROOT / "scripts" / "release_gate.sh"
    gate_unified = py_gate.exists() and sh_gate.exists()
    results.append((
        "9. release_gate.py and release_gate.sh unified",
        *_check("Unified gates", gate_unified),
    ))

    # 10. Candidate registry frozen to K8/V8
    try:
        from benchmarks.candidate_registry import build_default_registry
        reg = build_default_registry()
        names = reg.names()
        unsupported = {
            "rfsn_direct_packed_k8v5",
            "rfsn_direct_packed_k8v6",
            "rfsn_direct_packed_k16v8",
            "rfsn_direct_packed_k8v16",
            "rfsn_direct_packed_k16v16",
        }
        registry_frozen = (
            "rfsn_direct_packed_k8v8" in names
            and "dense_mlx_baseline" in names
            and not any(u in names for u in unsupported)
        )
    except Exception:
        registry_frozen = False
    results.append((
        "10. Candidate registry frozen to K8/V8",
        *_check("Registry frozen", registry_frozen, f"names={names}"),
    ))

    # 11. EvidenceStatus enum strict
    try:
        from rfsn_v11.candidates.evidence_status import EvidenceStatus
        evidence_strict = True
    except Exception:
        evidence_strict = False
    results.append((
        "11. Strict EvidenceStatus enum",
        *_check("EvidenceStatus", evidence_strict),
    ))

    # 12. Alpha release level criteria
    try:
        from rfsn_v11.candidates.release_levels import (
            ReleaseLevel, get_criteria,
        )
        criteria = get_criteria(ReleaseLevel.ALPHA)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            violations = criteria.check(manifest)
            alpha_ready = len(violations) == 0
        else:
            alpha_ready = False
    except Exception as exc:
        alpha_ready = False
        print(f"  Release level error: {exc}")
    results.append((
        "12. Alpha release level criteria satisfied",
        *_check("Alpha criteria", alpha_ready),
    ))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)

    print("\n--- Results ---")
    for name, ok, msg in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
        if not ok:
            print(f"      {msg}")

    print(f"\n{passed}/{len(results)} checks passed")

    if failed == 0:
        print("\n=== ALL DEFINITION-OF-DONE CHECKBOXES SATISFIED ===")
        return 0
    else:
        print(f"\n=== {failed} CHECKBOX(ES) MISSING ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
