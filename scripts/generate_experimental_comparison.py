#!/usr/bin/env python3
"""
Generate stable-vs-experimental comparison report.

Reads per-variant artifacts explicitly and produces:
  - artifacts/proof/experimental/comparison_summary.json
  - artifacts/proof/experimental/comparison_summary.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# --- Mappings --------------------------------------------------------------

EXPERIMENTAL_ARTIFACTS = {
    "experimental_hybrid": {
        "real": "real_model_validation.json",
        "long": "long_context_validation.json",
        "config_name": "experimental_hybrid",
    },
    "turbo_polar": {
        "real": "turbo_real_model.json",
        "long": "turbo_long_context.json",
        "config_name": "experimental_hybrid",
    },
    "adaptive": {
        "real": "adaptive_real_model.json",
        "long": "adaptive_long_context.json",
        "config_name": "experimental_hybrid",
    },
    "turbo_k8r8v6": {
        "real": "turbo_k8r8v6_real.json",
        "long": "turbo_k8r8v6_long.json",
        "config_name": "experimental_hybrid",
    },
}

STABLE_CONFIGS = {
    "stable_k8_v5_gs64": "k8_v5_gs64",
    "stable_k8_v5_gs32": "k8_v5_gs32",
    "stable_k8_v4_gs64": "k8_v4_gs64",
}

REQUIRED_CONTEXTS = [512, 1024, 2048]


# --- Helpers ---------------------------------------------------------------

def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _find_config(
    data: dict[str, Any] | None, name: str
) -> dict[str, Any] | None:
    if data is None:
        return None
    for cfg in data.get("configs", []):
        if cfg.get("name") == name:
            return cfg
    return None


def _find_first_config(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    configs = data.get("configs", [])
    if len(configs) >= 2:
        return configs[1]  # skip baseline_fp16
    if configs:
        return configs[0]
    return None


def extract_context_passes(long_data: dict, config_name: str) -> dict:
    """Return {ctx_key: pass/fail/unknown} from long-context artifact."""
    out = {str(ctx): "unknown" for ctx in REQUIRED_CONTEXTS}
    for ctx_row in long_data.get("contexts", []):
        ctx = (
            ctx_row.get("context")
            or ctx_row.get("tokens")
            or ctx_row.get("context_tokens")
        )
        if ctx is None:
            continue
        ctx_key = str(int(ctx))
        for cfg in ctx_row.get("configs", []):
            name = (
                cfg.get("name")
                or cfg.get("config")
                or cfg.get("config_name")
            )
            if name == config_name:
                status = cfg.get("status", "unknown")
                out[ctx_key] = "pass" if status == "pass" else "fail"
    return out


def extract_single_config_context_passes(long_data: dict) -> dict:
    """Fallback for experimental files with a single non-baseline config."""
    out = {str(ctx): "unknown" for ctx in REQUIRED_CONTEXTS}
    for ctx_row in long_data.get("contexts", []):
        ctx = (
            ctx_row.get("context")
            or ctx_row.get("tokens")
            or ctx_row.get("context_tokens")
        )
        if ctx is None:
            continue
        status = ctx_row.get("status")
        if status is None and "configs" in ctx_row \
                and len(ctx_row["configs"]) == 1:
            status = ctx_row["configs"][0].get("status")
        if status:
            out[str(int(ctx))] = "pass" if status == "pass" else "fail"
    return out


def classify_context_status(context_results: dict) -> tuple[str, dict]:
    """
    Returns:
        ("all_pass" | "failed" | "unknown", detail)
    """
    detail = {}
    for ctx in REQUIRED_CONTEXTS:
        value = context_results.get(str(ctx), "unknown")
        detail[f"pass_{ctx}"] = value
        if value == "fail":
            return "failed", detail
    for ctx in REQUIRED_CONTEXTS:
        if detail[f"pass_{ctx}"] == "unknown":
            return "unknown", detail
    return "all_pass", detail


def _memory_from_accounting(mem_data: dict, name: str) -> dict:
    if mem_data is None:
        return {}
    rows = mem_data.get("rows") or mem_data.get("configs", [])
    for row in rows:
        if row.get("config") == name:
            return row
    return {}


def _memory_from_throughput(throughput_data: dict, name: str) -> dict:
    if throughput_data is None:
        return {}
    for cfg in throughput_data.get("configs", []):
        if cfg.get("name") == name:
            fp16 = cfg.get("fp16_kv_bytes")
            comp = cfg.get("compressed_kv_bytes")
            ratio = cfg.get("effective_compression_ratio")
            return {
                "fp16_kv_bytes": fp16,
                "total_compressed_bytes": comp,
                "actual_compression_ratio": ratio,
                "memory_basis": "generation_throughput",
            }
    return {}


def load_stable_config(
    stable_dir: Path, alias: str, stable_name: str
) -> dict:
    real = _load_json(stable_dir / "real_model_validation.json")
    long = _load_json(stable_dir / "long_context_validation.json")
    throughput = _load_json(stable_dir / "generation_throughput.json")

    real_cfg = _find_config(real, stable_name) or {}
    context_passes = extract_context_passes(long or {}, stable_name)

    # Prefer same-basis memory from experimental script run
    same_basis_mem = _load_json(
        Path("artifacts/proof/experimental_stable_basis")
        / "memory_accounting.json"
    )
    mem = _memory_from_accounting(same_basis_mem, stable_name)
    if not mem:
        mem = _memory_from_throughput(throughput, stable_name)

    ctx_status, ctx_detail = classify_context_status(context_passes)

    row = {
        "config": alias,
        "source_config": stable_name,
        "cosine_mean": real_cfg.get("logit_cosine_mean"),
        "cosine_min": real_cfg.get("logit_cosine_min"),
        "top5_overlap": real_cfg.get("top5_overlap_mean"),
        "kl": real_cfg.get("kl_divergence_mean"),
        "nll_delta": real_cfg.get("avg_nll_delta"),
        **ctx_detail,
        "compression_ratio": mem.get("actual_compression_ratio"),
        "total_compressed_bytes": mem.get("total_compressed_bytes"),
        "memory_basis": mem.get("memory_basis"),
    }

    real_status = real_cfg.get("status", "unknown")
    memory_ok = mem.get("actual_compression_ratio") is not None

    if alias == "baseline_fp16":
        row["recommended_status"] = "reference"
    elif real_status not in ("pass", "reference"):
        row["recommended_status"] = "rejected_real_model"
    elif ctx_status == "failed":
        row["recommended_status"] = "rejected_context"
    elif ctx_status == "unknown":
        row["recommended_status"] = "needs_context_data"
    elif not memory_ok:
        row["recommended_status"] = "needs_memory_data"
    elif row["compression_ratio"] is not None \
            and row["compression_ratio"] <= 1.0:
        row["recommended_status"] = "rejected_no_compression"
    else:
        row["recommended_status"] = "candidate"

    return row


def load_experimental_variant(
    exp_dir: Path, name: str, spec: dict, memory_lookup: dict
) -> dict:
    real_path = exp_dir / spec["real"]
    long_path = exp_dir / spec["long"]
    if not real_path.exists() or not long_path.exists():
        return {
            "config": name,
            "recommended_status": "missing_data",
            "pass_512": "unknown",
            "pass_1024": "unknown",
            "pass_2048": "unknown",
        }

    real = _load_json(real_path) or {}
    long = _load_json(long_path) or {}
    cfg_name = spec.get("config_name", name)
    real_cfg = _find_config(real, cfg_name) or _find_first_config(real) or {}
    context_passes = extract_context_passes(long, cfg_name)
    if all(v == "unknown" for v in context_passes.values()):
        context_passes = extract_single_config_context_passes(long)

    ctx_status, ctx_detail = classify_context_status(context_passes)
    mem = memory_lookup.get(name, {})

    row = {
        "config": name,
        "cosine_mean": real_cfg.get("logit_cosine_mean"),
        "cosine_min": real_cfg.get("logit_cosine_min"),
        "top5_overlap": real_cfg.get("top5_overlap_mean"),
        "kl": real_cfg.get("kl_divergence_mean"),
        "nll_delta": real_cfg.get("avg_nll_delta"),
        **ctx_detail,
        "compression_ratio": mem.get("actual_compression_ratio"),
        "total_compressed_bytes": mem.get("total_compressed_bytes"),
        "memory_basis": mem.get("memory_basis"),
    }

    real_status = real_cfg.get("status", "unknown")
    memory_ok = mem.get("actual_compression_ratio") is not None

    if name == "baseline_fp16":
        row["recommended_status"] = "reference"
    elif real_status not in ("pass", "reference"):
        row["recommended_status"] = "rejected_real_model"
    elif ctx_status == "failed":
        row["recommended_status"] = "rejected_context"
    elif ctx_status == "unknown":
        row["recommended_status"] = "needs_context_data"
    elif not memory_ok:
        row["recommended_status"] = "needs_memory_data"
    elif row["compression_ratio"] is not None \
            and row["compression_ratio"] <= 1.0:
        row["recommended_status"] = "rejected_no_compression"
    else:
        row["recommended_status"] = "candidate"

    return row


def _make_md_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Config",
        "CosMean",
        "CosMin",
        "Top5",
        "KL",
        "NLLΔ",
        "512",
        "1024",
        "2048",
        "CompRatio",
        "Bytes",
        "Status",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        vals = [
            r["config"],
            _fmt(r.get("cosine_mean")),
            _fmt(r.get("cosine_min")),
            _fmt(r.get("top5_overlap")),
            _fmt(r.get("kl")),
            _fmt(r.get("nll_delta")),
            r.get("pass_512", "-"),
            r.get("pass_1024", "-"),
            r.get("pass_2048", "-"),
            _fmt(r.get("compression_ratio")),
            str(r.get("total_compressed_bytes", "-"))
            if r.get("total_compressed_bytes") is not None
            else "-",
            r.get("recommended_status", "-"),
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stable-vs-experimental comparison report"
    )
    parser.add_argument(
        "--experimental-dir",
        default="artifacts/proof/experimental",
        help="Directory with experimental validation artifacts",
    )
    parser.add_argument(
        "--stable-dir",
        default="artifacts/proof/main28",
        help="Directory with stable validation artifacts (e.g. main28)",
    )
    parser.add_argument(
        "--configs",
        default=(
            "baseline_fp16,stable_k8_v5_gs64,stable_k8_v5_gs32,"
            "experimental_hybrid,turbo_polar,adaptive,turbo_k8r8v6"
        ),
        help="Comma-separated config names to compare",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/proof/experimental",
        help="Output directory for comparison artifacts",
    )
    args = parser.parse_args()

    exp_dir = Path(args.experimental_dir)
    stable_dir = Path(args.stable_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build memory lookup from experimental memory_accounting.json
    mem_data = _load_json(exp_dir / "memory_accounting.json")
    memory_lookup: dict[str, dict] = {}
    if mem_data is not None:
        rows = mem_data.get("rows") or mem_data.get("configs", [])
        for row in rows:
            cfg_name = row.get("config")
            if cfg_name:
                memory_lookup[cfg_name] = row

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    rows: list[dict[str, Any]] = []
    for name in config_names:
        if name == "baseline_fp16":
            rows.append({
                "config": "baseline_fp16",
                "cosine_mean": None,
                "cosine_min": None,
                "top5_overlap": None,
                "kl": None,
                "nll_delta": None,
                "pass_512": "pass",
                "pass_1024": "pass",
                "pass_2048": "pass",
                "compression_ratio": 1.0,
                "total_compressed_bytes": None,
                "recommended_status": "reference",
            })
        elif name in STABLE_CONFIGS:
            rows.append(
                load_stable_config(stable_dir, name, STABLE_CONFIGS[name])
            )
        elif name in EXPERIMENTAL_ARTIFACTS:
            rows.append(
                load_experimental_variant(
                    exp_dir, name, EXPERIMENTAL_ARTIFACTS[name], memory_lookup
                )
            )
        else:
            rows.append({
                "config": name,
                "recommended_status": "missing_data",
                "pass_512": "unknown",
                "pass_1024": "unknown",
                "pass_2048": "unknown",
            })

    payload = {
        "release": "experimental",
        "configs_compared": config_names,
        "stable_source": str(stable_dir),
        "experimental_source": str(exp_dir),
        "qjl_status": {
            "enabled_by_default": False,
            "passes_attention_score_benchmark": False,
            "recommendation": "disabled",
        },
        "rows": rows,
        "notes": [
            "recommended_status values: reference, candidate, "
            "rejected_real_model, rejected_quality, rejected_context, "
            "rejected_no_compression, needs_memory_data, "
            "needs_context_data, missing_data",
            "No config is recommended after failing any context.",
            "Unknown context fields produce needs_context_data, "
            "never candidate.",
        ],
        "memory_notes": [
            "experimental_hybrid, turbo_polar, and adaptive use "
            "identical bit widths (8/8/7/8/5); memory differs "
            "only if cartesian_bits or group_size changes. "
            "adaptive differs only by adaptive_angle_range=True "
            "(quality tuning, not storage).",
            "turbo_k8r8v6 uses cartesian_bits=6, producing "
            "larger codes (2,600,928 bytes).",
            "All configs now use mean_per_prompt_real_model_cache "
            "basis via validate_experimental_quant.py.",
            "Bit-packing is real for 2-8 bit code buffers. "
            "Code widths above 8 use raw uint32 fallback.",
            "No Metal kernels exist for the experimental quantizer path.",
            "No experimental throughput speedup is proven.",
        ],
    }
    json_path = out_dir / "comparison_summary.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {json_path}")

    md_lines = [
        "# Stable vs Experimental Quantization Comparison",
        "",
        f"**Stable source:** `{stable_dir}`  ",
        f"**Experimental source:** `{exp_dir}`  ",
        "",
        "## Config Comparison",
        "",
        _make_md_table(rows),
        "",
        "## Legend",
        "",
        "- **reference**: baseline_fp16 (no compression)",
        "- **candidate**: passes quality + all contexts + compression > 1.0",
        "- **rejected_real_model**: failed real-model quality thresholds",
        "- **rejected_quality**: legacy label for failed quality thresholds",
        "- **rejected_context**: failed one or more context lengths",
        "- **rejected_no_compression**: compression ratio <= 1.0",
        "- **needs_memory_data**: missing memory accounting data",
        "- **needs_context_data**: missing or incomplete long-context data",
        "- **missing_data**: config not found in validation artifacts",
        "",
        "*Generated by generate_experimental_comparison.py*",
    ]
    md_path = out_dir / "comparison_summary.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
