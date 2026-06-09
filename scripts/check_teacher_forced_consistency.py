#!/usr/bin/env python3
"""Reconcile bulk real-generation throughput with per-step teacher-forced

trace.

Loads both artifacts and flags discrepancies.  A mismatch means the bulk
benchmark methodology should be treated as suspect / under investigation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict | list:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    base = Path("artifacts/proof/experimental")
    bulk_path = base / "real_generation_throughput.json"
    trace_path = base / "teacher_forced_step_trace.json"

    if not bulk_path.exists():
        print(f"SKIP: {bulk_path} not found")
        return 0
    if not trace_path.exists():
        print(f"SKIP: {trace_path} not found")
        return 0

    bulk = _load_json(bulk_path)
    trace_doc = _load_json(trace_path)

    if not isinstance(bulk, dict):
        print(f"FAIL: {bulk_path} is not a dict")
        return 1
    trace = (
        trace_doc.get("traces") if isinstance(trace_doc, dict) else trace_doc
    )
    if not isinstance(trace, list):
        print(f"FAIL: {trace_path} missing 'traces' list")
        return 1

    # Aggregate step-trace by (config, prompt_tokens)
    agg: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in trace:
        key = (row.get("config"), row.get("prompt_tokens"))
        if None in key:
            continue
        bucket = agg.setdefault(key, {"cosine": [], "top5": [], "kl": []})
        bucket["cosine"].append(row.get("logit_cosine_vs_fp16", 0.0))
        bucket["top5"].append(row.get("top5_overlap_vs_fp16", 0.0))
        bucket["kl"].append(row.get("kl_vs_fp16", 0.0))

    discrepancies = []
    for (config, prompt_tokens), metrics in agg.items():
        if not metrics["cosine"]:
            continue
        avg_cosine = sum(metrics["cosine"]) / len(metrics["cosine"])
        avg_top5 = sum(metrics["top5"]) / len(metrics["top5"])
        avg_kl = sum(metrics["kl"]) / len(metrics["kl"])

        # Find matching bulk entry
        bulk_entry = None
        for entry in bulk.get("teacher_forced_logits", []):
            if (
                entry.get("config") == config
                and entry.get("prompt_tokens") == prompt_tokens
            ):
                bulk_entry = entry
                break

        if bulk_entry is None:
            discrepancies.append(
                f"{config}@{prompt_tokens}: missing in bulk benchmark"
            )
            continue

        bulk_cosine = bulk_entry.get("logit_cosine_vs_fp16")
        bulk_top5 = bulk_entry.get("top5_overlap_vs_fp16")
        bulk_kl = bulk_entry.get("kl_vs_fp16")

        # Allow small tolerance for floating-point differences
        tol = 0.001
        if bulk_cosine is not None and abs(avg_cosine - bulk_cosine) > tol:
            discrepancies.append(
                f"{config}@{prompt_tokens}: cosine mismatch "
                f"trace_avg={avg_cosine:.6f} bulk={bulk_cosine:.6f}"
            )
        if bulk_top5 is not None and abs(avg_top5 - bulk_top5) > tol:
            discrepancies.append(
                f"{config}@{prompt_tokens}: top5 mismatch "
                f"trace_avg={avg_top5:.6f} bulk={bulk_top5:.6f}"
            )
        if bulk_kl is not None and abs(avg_kl - bulk_kl) > tol:
            discrepancies.append(
                f"{config}@{prompt_tokens}: KL mismatch "
                f"trace_avg={avg_kl:.6f} bulk={bulk_kl:.6f}"
            )

    if discrepancies:
        print(
            "DISCREPANCY: bulk real_generation_throughput.json disagrees "
            "with teacher_forced_step_trace.json"
        )
        for d in discrepancies:
            print(f"  - {d}")
        print(
            "\nRECOMMENDATION: Mark real_generation_throughput.json as "
            "stale / under investigation."
        )
        return 2

    print("PASS: bulk benchmark matches step-trace aggregation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
