#!/usr/bin/env python3
"""Verify proof_summary.md matches JSON artifacts exactly."""

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def check(proof_dir: Path) -> list[str]:
    errors: list[str] = []

    lc_path = proof_dir / "long_context_validation.json"
    rm_path = proof_dir / "real_model_validation.json"
    tp_path = proof_dir / "generation_throughput.json"
    ps_path = proof_dir / "proof_summary.md"
    su_path = proof_dir / "summary.json"

    # 1. proof_summary.md must exist and mention Main 28
    if not ps_path.exists():
        errors.append("proof_summary.md missing")
        return errors
    proof_text = ps_path.read_text(encoding="utf-8")
    if "Main 28" not in proof_text:
        errors.append("proof_summary.md does not identify Main 28")

    # 2. long_context: every config claimed as passing must pass all contexts
    if lc_path.exists():
        lc = _load_json(lc_path)
        for ctx_entry in lc.get("contexts", []):
            for cfg in ctx_entry.get("configs", []):
                name = cfg.get("name", "")
                status = cfg.get("status", "")
                # A config passes long-context only if pass in every context
                if status not in ("pass", "fail"):
                    errors.append(
                        f"long_context {ctx_entry.get('tokens')} "
                        f"{name}: unexpected status {status!r}"
                    )

    # 3. real_model: rejected configs must not appear in recommended section
    if rm_path.exists() and ps_path.exists():
        rm = _load_json(rm_path)
        rejected = set()
        for cfg in rm.get("configs", []):
            if cfg.get("status") != "pass":
                rejected.add(cfg.get("name", ""))

        # Find "Recommended Configs" section
        in_recommended = False
        recommended_lines = []
        for line in proof_text.splitlines():
            if "Recommended Configs" in line or "Recommended" in line:
                in_recommended = True
            elif in_recommended and line.startswith("## "):
                in_recommended = False
            elif in_recommended:
                recommended_lines.append(line)

        recommended_text = "\n".join(recommended_lines).lower()
        for r in rejected:
            if r and r.lower() in recommended_text:
                errors.append(
                    f"rejected config {r!r} appears in Recommended section"
                )

    # 4. throughput: total_end_to_end_ms must not be claimed as speedup
    # if worse than baseline
    if tp_path.exists() and ps_path.exists():
        tp = _load_json(tp_path)
        baseline_total = None
        for cfg in tp.get("configs", []):
            if cfg.get("name") == "baseline_fp16":
                baseline_total = cfg.get("total_end_to_end_ms_mean")
                break

        if baseline_total is not None:
            for cfg in tp.get("configs", []):
                if cfg.get("name") == "baseline_fp16":
                    continue
                total = cfg.get("total_end_to_end_ms_mean")
                if total and total >= baseline_total:
                    low = proof_text.lower()
                    negations = [
                        "not ", "no ", "never ", "unproven",
                        "not proven",
                    ]
                    # Check each sentence containing "speedup" or "faster"
                    for sentence in low.split("."):
                        if "speedup" in sentence or "faster" in sentence:
                            if not any(n in sentence for n in negations):
                                errors.append(
                                    f"throughput: {cfg.get('name')} total "
                                    f"({total:.1f}ms) >= baseline "
                                    f"({baseline_total:.1f}ms), "
                                    f"but proof_summary claims speedup"
                                )

    # 5. summary.json recommended_default must match proof_summary
    if su_path.exists() and ps_path.exists():
        su = _load_json(su_path)
        rec = su.get("recommended_default", "")
        if rec and rec.lower() not in proof_text.lower():
            errors.append(
                f"summary.json recommended_default {rec!r} "
                f"not found in proof_summary.md"
            )

    # 6. sparse disabled
    if rm_path.exists():
        rm = _load_json(rm_path)
        if rm.get("sparse_enabled") is True:
            errors.append("sparse_enabled is True in real_model_validation.json")
        if "sparse" in proof_text.lower():
            low = proof_text.lower()
            if "disabled" not in low and "experimental" not in low:
                errors.append(
                    "proof_summary.md must say sparse is "
                    "disabled/experimental"
                )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proof-dir", default="artifacts/proof/main28")
    args = parser.parse_args()

    errors = check(Path(args.proof_dir))
    if errors:
        for e in errors:
            print(f"CONSISTENCY ERROR: {e}", file=sys.stderr)
        return 1
    print("proof-summary consistency OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
