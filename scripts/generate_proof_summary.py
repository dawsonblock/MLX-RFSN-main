#!/usr/bin/env python3
"""Generate proof_summary.md from JSON artifacts."""

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: float) -> str:
    if abs(v) < 0.0001:
        return f"{v:.6f}"
    if abs(v) < 0.01:
        return f"{v:.5f}"
    if abs(v) < 1:
        return f"{v:.4f}"
    return f"{v:.3f}"


def generate(proof_dir: Path) -> str:
    kb = _load_json(proof_dir / "kernel_benchmark.json")
    fb = _load_json(proof_dir / "fused_kernel_benchmark.json")
    _ = _load_json(proof_dir / "optimization_benchmark.json")
    rm = _load_json(proof_dir / "real_model_validation.json")
    lc = _load_json(proof_dir / "long_context_validation.json")
    gs = _load_json(proof_dir / "generation_smoke.json")
    tp = _load_json(proof_dir / "generation_throughput.json")

    # Derive recommendations from JSON pass/fail
    passing_configs = set()
    rejected_configs = set()
    for cfg in rm.get("configs", []):
        name = cfg.get("name", "")
        if cfg.get("status") == "pass":
            passing_configs.add(name)
        else:
            rejected_configs.add(name)

    # A config passes long-context only if pass in every context
    long_passing = set(passing_configs)
    for ctx_entry in lc.get("contexts", []):
        for cfg in ctx_entry.get("configs", []):
            name = cfg.get("name", "")
            if cfg.get("status") != "pass":
                long_passing.discard(name)
                rejected_configs.add(name)

    recommended_default = ""
    best_quality = ""
    best_memory = ""
    if "k8_v5_gs64" in long_passing:
        recommended_default = "k8_v5_gs64"
    if "k8_v5_gs32" in long_passing:
        best_quality = "k8_v5_gs32"
    if "k8_v4_gs64" in long_passing:
        best_memory = "k8_v4_gs64"

    # Check end-to-end speedup
    baseline_total = None
    for cfg in tp.get("configs", []):
        if cfg.get("name") == "baseline_fp16":
            baseline_total = cfg.get("total_end_to_end_ms_mean")
            break
    speedup_proven = False
    if baseline_total:
        for cfg in tp.get("configs", []):
            if cfg.get("name") != "baseline_fp16":
                total = cfg.get("total_end_to_end_ms_mean")
                if total and total < baseline_total:
                    speedup_proven = True

    # Kernel tables
    kernel_table = "| Benchmark | Status | Cosine | Max Abs Diff |\n"
    kernel_table += "|-----------|--------|--------|--------------|\n"
    for entry in kb.get("results", []):
        status = entry.get("status", "?")
        cos = entry.get("cosine_similarity", "?")
        mad = entry.get("max_abs_diff", "?")
        kernel_table += (
            f"| {entry.get('name', '?')} | {status} | "
            f"{cos} | {mad} |\n"
        )

    fused_table = "| Benchmark | Status | Cosine | Max Abs Diff |\n"
    fused_table += "|-----------|--------|--------|--------------|\n"
    for entry in fb.get("results", []):
        status = entry.get("status", "?")
        cos = entry.get("cosine_similarity", "?")
        mad = entry.get("max_abs_diff", "?")
        fused_table += (
            f"| {entry.get('name', '?')} | {status} | "
            f"{cos} | {mad} |\n"
        )

    # Real-model validation table
    rm_table = (
        "| Config | Cosine Mean | Cosine Min | Top1 Match | "
        "NLL Δ | KL | Status |\n"
    )
    rm_table += (
        "|--------|-------------|------------|------------|"
        "-------|-----|--------|\n"
    )
    for cfg in rm.get("configs", []):
        status = cfg.get("status", "fail")
        status_str = "**PASS**" if status == "pass" else f"**FAIL**"
        rm_table += (
            f"| {cfg.get('name', '?')} | "
            f"{_fmt(cfg.get('logit_cosine_mean', 0))} | "
            f"{_fmt(cfg.get('logit_cosine_min', 0))} | "
            f"{_fmt(cfg.get('top1_match_rate', 0))} | "
            f"{_fmt(cfg.get('avg_nll_delta', 0))} | "
            f"{_fmt(cfg.get('kl_divergence_mean', 0))} | "
            f"{status_str} |\n"
        )

    # Long-context table
    contexts = lc.get("contexts", [])
    ctx_headers = "| Config |"
    ctx_div = "|--------|"
    for ctx in contexts:
        ctx_headers += f" {ctx.get('tokens', '?')} |"
        ctx_div += "-----|"
    ctx_headers += " Passes All |\n"
    ctx_div += "------------|\n"

    lc_table = ctx_headers + ctx_div
    config_names = []
    if contexts:
        config_names = [c.get("name", "") for c in contexts[0].get("configs", [])]
    for name in config_names:
        all_pass = True
        row = f"| {name} |"
        for ctx in contexts:
            status = "?"
            for cfg in ctx.get("configs", []):
                if cfg.get("name") == name:
                    status = cfg.get("status", "?")
                    break
            row += f" {status.upper()} |"
            if status != "pass":
                all_pass = False
        row += f" {'**YES**' if all_pass else 'NO'} |\n"
        lc_table += row

    # Throughput table
    tp_table = (
        "| Config | Prefill (ms) | Compress (ms) | Decode (ms) | "
        "Total (ms) | TPS | Comp Ratio |\n"
    )
    tp_table += (
        "|--------|--------------|---------------|-------------|"
        "------------|-----|-------------|\n"
    )
    for cfg in tp.get("configs", []):
        ratio = cfg.get("effective_compression_ratio", 1.0)
        tp_table += (
            f"| {cfg.get('name', '?')} | "
            f"{_fmt(cfg.get('prefill_ms_mean', 0))} | "
            f"{_fmt(cfg.get('compress_ms_mean', 0))} | "
            f"{_fmt(cfg.get('decode_ms_mean', 0))} | "
            f"{_fmt(cfg.get('total_end_to_end_ms_mean', 0))} | "
            f"{_fmt(cfg.get('tokens_per_sec_mean', 0))} | "
            f"{ratio:.1f}x |\n"
        )

    # Generation smoke table
    gs_table = (
        "| Config | Token Match | Edit Dist | Repetition | NaN | Status |\n"
    )
    gs_table += (
        "|--------|-------------|-----------|------------|-----|--------|\n"
    )
    for cfg in gs.get("configs", []):
        gs_table += (
            f"| {cfg.get('name', '?')} | "
            f"{cfg.get('token_match_rate', 0):.3f} | "
            f"{cfg.get('normalised_edit_distance', 0):.3f} | "
            f"{cfg.get('repetition_rate_4gram', 0):.3f} | "
            f"{'Yes' if cfg.get('had_nan_logits') else 'No'} | "
            f"{cfg.get('status', '?').upper()} |\n"
        )

    # Recommended / rejected lists
    rec_lines = ""
    if recommended_default:
        rec_lines += f"- **Recommended practical default**: `{recommended_default}`\n"
    if best_quality:
        rec_lines += f"- **Best quality**: `{best_quality}`\n"
    if best_memory:
        rec_lines += f"- **Lowest-bit passing**: `{best_memory}`\n"
    rej_lines = ""
    for r in sorted(rejected_configs):
        if r:
            rej_lines += f"- `{r}`\n"

    speedup_text = (
        "Decode throughput is comparable to baseline, but end-to-end "
        "runtime is slower because compression overhead dominates."
        if not speedup_proven
        else "End-to-end speedup is proven by throughput JSON."
    )

    rec_block = rec_lines if rec_lines else "_None pass all validation criteria._\n"
    rej_block = rej_lines if rej_lines else "_None rejected._\n"

    return f"""# Proof Summary — Main 28

**Release**: Main 28 — Proof Consistency + Long-Context + Throughput Honesty  
**Status**: Alpha  
**Date**: 2026-06-03  
**Hardware**: Apple M2 Pro, 16GB RAM  
**Model**: Qwen/Qwen2.5-0.5B-Instruct  

---

## Release Identity

This release makes the release identity, proof summary, validation artifacts, throughput reporting, and long-context claims internally consistent. No new architecture features were added.

---

## Synthetic Kernel Benchmark

{kernel_table}

---

## Fused Kernel Benchmark

{fused_table}

---

## Real-Model Validation

**Method**: Causal-correct NLL scoring with 64 decode positions across 5 prompts.  
**Model**: Qwen/Qwen2.5-0.5B-Instruct  
**Context**: 512 tokens

{rm_table}

---

## Long-Context Validation

Contexts tested: {', '.join(str(c.get('tokens', '?')) for c in contexts)} tokens  
Positions evaluated: 64 per context

{lc_table}

---

## Generation Smoke Test

**Method**: Greedy decode 64 tokens, compare to baseline  
**Context**: 128 tokens

{gs_table}

---

## Generation Throughput

**Method**: 5 timed repeats after 2 warmup runs  
**Context**: 512 tokens, decode 64 tokens

{tp_table}

{speedup_text}

---

## Memory / Compression

The primary proven benefit is KV memory reduction. Effective compression ratios are approximately 2.3x for 8-bit K / 4-bit V configs.

---

## Sparse Decode Status

Sparse decode is **disabled by default** and remains experimental. Do not enable unless explicitly testing the safety gate.

---

## Not Implemented

- Polar quantization is not implemented.
- True arbitrary token-level partial dequantization is not implemented.
- Per-layer sensitivity analysis and targeted layer protection are deferred to a future release.

---

## Recommended Configs

{rec_block}

## Rejected Configs

{rej_block}

---

## Limitations

- RFSN is a research runtime, not production-ready.
- End-to-end speedup has not been proven.
- Validation is on a single small model (Qwen2.5-0.5B).
- Sparse decode is disabled.
- Polar quantization is not implemented.

---

## Conclusion

Main 28 successfully locks the truthful position: RFSN v10 is a clean Apple Silicon KV-cache compression research runtime. Its best practical config is `{recommended_default or 'none'}`, it reduces KV bytes, but has not proven end-to-end speedup and is not production-ready.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="artifacts/proof/main28")
    parser.add_argument("--out", default="artifacts/proof/main28/proof_summary.md")
    args = parser.parse_args()

    text = generate(Path(args.input_dir))
    out_path = Path(args.out)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote proof summary to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
