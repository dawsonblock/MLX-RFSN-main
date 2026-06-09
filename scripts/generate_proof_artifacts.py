#!/usr/bin/env python3
"""Generate proof artifacts for benchmark scenarios.

Outputs:
- kv_cache_runs.json
- e2e_scenarios.json
- proof_summary.md
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow direct execution from repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from benchmarks.benchmark_kv_cache import benchmark_kv  # noqa: E402
    from benchmarks.benchmark_end_to_end import benchmark_e2e  # noqa: E402
    from tools.proof_regression import load_thresholds_file  # noqa: E402
except Exception as exc:  # pragma: no cover - exercised on non-MLX envs
    raise SystemExit(
        "MLX is required to generate proof artifacts. "
        "Run this on Apple Silicon with: pip install -e '.[dev,mlx]'"
    ) from exc


KV_SHAPES = [
    (1, 8, 1024, 64),
    (1, 8, 2048, 64),
    (1, 32, 4096, 128),
]

KV_CONFIGS = [
    (8, 3, True),
    (8, 3, False),
    (8, 8, False),
]

E2E_SCENARIOS = [
    # scenario, shape_q, shape_kv, k_bits, v_bits, use_incoherent,
    # top_k_ratio, use_compressed_on_miss, enable_sparse_decode,
    # reserved_sink_blocks, reserved_recent_blocks
    ("dense_baseline", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 1.00, True, False, 1, 2),
    ("sparse_disabled_by_default", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, True, False, 1, 2),
    ("sparse_topk_075_sink1_recent2", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.75, True, True, 1, 2),
    ("sparse_topk_050_sink1_recent2", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, True, True, 1, 2),
    ("sparse_topk_050_sink1_recent4", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, True, True, 1, 4),
    ("sparse_topk_025_sink1_recent4", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.25, True, True, 1, 4),
    # Legacy scenarios retained for baseline comparability in strict regression checks.
    ("cache_miss_full_precision_path", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, False, True, 1, 2),
    ("cache_miss_use_compressed_on_miss_path", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, True, True, 1, 2),
    ("cache_hit_compressed_path", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.50, True, True, 1, 2),
    ("sparse_decode_path", (1, 8, 1, 64), (1, 8, 2048, 64), 8, 3, True, 0.60, True, True, 1, 2),
    ("dense_decode_path", (1, 8, 8, 64), (1, 8, 2048, 64), 8, 3, True, 0.25, True, True, 1, 2),
]


def _metadata() -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }


def run_kv_benchmarks(iterations: int) -> dict:
    runs = []
    for shape in KV_SHAPES:
        for k_bits, v_bits, use_incoherent in KV_CONFIGS:
            result = benchmark_kv(shape, k_bits, v_bits, use_incoherent, iterations=iterations)
            result["scenario"] = f"shape={shape}|k={k_bits}|v={v_bits}|incoherent={use_incoherent}"
            runs.append(result)
    return {"metadata": _metadata(), "iterations": iterations, "runs": runs}


def run_e2e_benchmarks(iterations: int) -> dict:
    runs = []
    for (
        scenario,
        shape_q,
        shape_kv,
        k_bits,
        v_bits,
        use_incoherent,
        top_k_ratio,
        use_compressed_on_miss,
        enable_sparse_decode,
        reserved_sink_blocks,
        reserved_recent_blocks,
    ) in E2E_SCENARIOS:
        result = benchmark_e2e(
            shape_q,
            shape_kv,
            k_bits,
            v_bits,
            use_incoherent,
            top_k_ratio,
            reserved_sink_blocks=reserved_sink_blocks,
            reserved_recent_blocks=reserved_recent_blocks,
            enable_sparse_decode=enable_sparse_decode,
            use_compressed_on_miss=use_compressed_on_miss,
            iterations=iterations,
        )
        # Keep canonical aliases expected by external proof parsers.
        result["execution_mode"] = result.get("cache_hit_execution_mode")
        result["latency_ms"] = result.get("cache_hit_total_latency_ms")
        result["scenario"] = scenario
        runs.append(result)
    return {"metadata": _metadata(), "iterations": iterations, "runs": runs}


def write_summary(output_dir: Path, kv_payload: dict, e2e_payload: dict, profile: str) -> None:
    kv_runs = kv_payload["runs"]
    e2e_runs = e2e_payload["runs"]

    best_kv = min(kv_runs, key=lambda r: r["retrieve_latency_ms"]) if kv_runs else None
    sparse_runs = [r for r in e2e_runs if str(r.get("scenario", "")).startswith("sparse_")]
    best_sparse = max(
        sparse_runs,
        key=lambda r: float(r.get("sparse_audit_cosine") or 0.0),
        default=None,
    )
    worst_sparse = min(
        sparse_runs,
        key=lambda r: float(r.get("sparse_audit_cosine") or 0.0),
        default=None,
    )
    best_dense = next((r for r in e2e_runs if r["scenario"] == "dense_baseline"), None)

    thresholds = load_thresholds_file(REPO_ROOT / "scripts/proof_regression_thresholds.json")
    absolute_cfg = thresholds.get("absolute_quality_min", {})

    def _min_metric(runs: list[dict], metric: str) -> float | None:
        values = []
        for run in runs:
            value = run.get(metric)
            if value is None:
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                continue
        return min(values) if values else None

    def _status(min_value: float | None, threshold: float) -> str:
        if min_value is None:
            return "fail"
        if min_value >= threshold:
            return "pass"
        return "warn"

    sparse_threshold = float(absolute_cfg.get("absolute_sparse_audit_cosine_min", 0.90))
    quant_threshold = float(absolute_cfg.get("absolute_quant_audit_cosine_min", 0.95))
    value_threshold = float(absolute_cfg.get("absolute_value_cosine_min", 0.90))

    sparse_min = _min_metric(e2e_runs, "sparse_audit_cosine")
    quant_min = _min_metric(e2e_runs, "quant_audit_cosine")
    value_min = _min_metric(kv_runs, "value_cosine_sim")

    sparse_status = _status(sparse_min, sparse_threshold)
    quant_status = _status(quant_min, quant_threshold)
    value_status = _status(value_min, value_threshold)

    unsafe_for_llm = any(status in {"warn", "fail"} for status in (sparse_status, quant_status, value_status))
    sparse_deployment_safe = bool(sparse_min is not None and sparse_min >= sparse_threshold)
    sparse_default_recommendation = (
        "sparse_allowed" if sparse_deployment_safe else "dense_default"
    )
    sparse_default = "disabled"

    kernel_benchmark_status = "not run"
    kernel_benchmark_path = output_dir / "kernel_benchmark.json"
    if kernel_benchmark_path.exists():
        try:
            kernel_payload = json.loads(kernel_benchmark_path.read_text(encoding="utf-8"))
            if kernel_payload.get("runs"):
                kernel_benchmark_status = "run"
        except Exception:
            kernel_benchmark_status = "not run"

    real_model_status = "not run"
    real_model_path = output_dir / "real_model_validation.json"
    if real_model_path.exists():
        try:
            real_payload = json.loads(real_model_path.read_text(encoding="utf-8"))
            real_model_status = "run" if real_payload.get("status") == "run" else "not run"
        except Exception:
            real_model_status = "not run"

    lines = [
        f"# {profile} Proof Summary",
        "",
        f"Generated: {_metadata()['timestamp']}",
        "",
        "## Files",
        "- kv_cache_runs.json",
        "- e2e_scenarios.json",
        "- kernel_benchmark.json" if kernel_benchmark_status == "run" else "- kernel_benchmark.json (not generated)",
        "- real_model_validation.json" if real_model_path.exists() else "- real_model_validation.json (not generated)",
        "",
        "## Highlights",
    ]

    if best_kv is not None:
        lines.append(
            "- Fastest KV retrieve: "
            f"{best_kv['retrieve_latency_ms']:.2f}ms "
            f"({best_kv['scenario']})"
        )
        lines.append(
            "- KV value quality (same run): "
            f"cos={best_kv['value_cosine_sim']:.4f}, "
            f"rel_mae={best_kv['value_rel_mae']:.4f}, "
            f"max_abs={best_kv['value_max_abs_error']:.4f}"
        )

    if best_sparse is not None:
        lines.append(
            "- Best sparse scenario: "
            f"{best_sparse['scenario']} "
            f"miss={best_sparse['cache_miss_total_latency_ms']:.2f}ms, "
            f"hit={best_sparse['cache_hit_total_latency_ms']:.2f}ms, "
            f"sparse_cos={best_sparse.get('sparse_audit_cosine')}"
        )

    if worst_sparse is not None:
        lines.append(
            "- Worst sparse scenario: "
            f"{worst_sparse['scenario']} "
            f"hit_mode={worst_sparse.get('cache_hit_execution_mode')} "
            f"sparse_cos={worst_sparse.get('sparse_audit_cosine')} "
            f"sparse_rel_mae={worst_sparse.get('sparse_audit_rel_mae')}"
        )

    if best_dense is not None:
        lines.append(
            "- Dense decode path: "
            f"miss={best_dense['cache_miss_total_latency_ms']:.2f}ms, "
            f"hit={best_dense['cache_hit_total_latency_ms']:.2f}ms, "
            f"mode={best_dense.get('cache_hit_execution_mode')}"
        )

    lines.extend([
        "",
        "## Absolute Quality",
        (
            f"- Sparse quality: {sparse_status} "
            f"(min={sparse_min if sparse_min is not None else 'n/a'}, threshold={sparse_threshold:.3f})"
        ),
        (
            f"- Quant quality: {quant_status} "
            f"(min={quant_min if quant_min is not None else 'n/a'}, threshold={quant_threshold:.3f})"
        ),
        (
            f"- KV cache quality: {value_status} "
            f"(min={value_min if value_min is not None else 'n/a'}, threshold={value_threshold:.3f})"
        ),
        f"- Sparse default: {sparse_default}",
        "- WARNING_UNSAFE_FOR_LLM_DEPLOYMENT" if unsafe_for_llm else "- Deployment quality warning: none",
        (
            "- Sparse deployment threshold met: yes"
            if sparse_deployment_safe
            else "- Sparse deployment threshold met: no"
        ),
        (
            "- Recommended default: sparse decode may be enabled for validated profiles"
            if sparse_deployment_safe
            else "- Recommended default: dense (sparse decode remains experimental and should default to disabled)"
        ),
        f"- Kernel benchmark: {kernel_benchmark_status}",
        f"- Real model validation: {real_model_status}",
        "",
        "## Next Checks",
        "- Compare these artifacts against previous runs for trend regressions.",
        "- Keep sparse vs dense and quant vs sparse audit metrics split in analysis.",
    ])

    summary_payload = {
        "profile": profile,
        "generated_at": _metadata()["timestamp"],
        "files": ["kv_cache_runs.json", "e2e_scenarios.json"],
        "highlights": {
            "best_kv": best_kv,
            "best_sparse": best_sparse,
            "worst_sparse": worst_sparse,
            "best_dense": best_dense,
        },
        "absolute_quality": {
            "sparse": {
                "status": sparse_status,
                "min": sparse_min,
                "threshold": sparse_threshold,
            },
            "quant": {
                "status": quant_status,
                "min": quant_min,
                "threshold": quant_threshold,
            },
            "value": {
                "status": value_status,
                "min": value_min,
                "threshold": value_threshold,
            },
            "warning_unsafe_for_llm_deployment": unsafe_for_llm,
            "sparse_deployment_safe": sparse_deployment_safe,
            "sparse_default": sparse_default,
            "sparse_default_recommendation": sparse_default_recommendation,
        },
        "kernel_benchmark": {
            "status": kernel_benchmark_status,
        },
        "real_model_validation": {
            "status": "run" if real_model_status == "run" else "not_run",
        },
    }

    (output_dir / "proof_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate proof artifacts")
    parser.add_argument(
        "--profile",
        default="main12",
        help="Proof profile name for output labeling/default paths",
    )
    parser.add_argument(
        "--output-dir",
        "--out",
        default="",
        help="Directory where JSON/Markdown artifacts are written",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Iterations per scenario",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(f"artifacts/proof/{args.profile}")
    output_dir.mkdir(parents=True, exist_ok=True)

    kv_payload = run_kv_benchmarks(iterations=args.iterations)
    e2e_payload = run_e2e_benchmarks(iterations=args.iterations)

    (output_dir / "kv_cache_runs.json").write_text(
        json.dumps(kv_payload, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "e2e_scenarios.json").write_text(
        json.dumps(e2e_payload, indent=2) + "\n", encoding="utf-8"
    )
    write_summary(output_dir, kv_payload, e2e_payload, profile=args.profile)

    print(f"Wrote proof artifacts to {output_dir}")


if __name__ == "__main__":
    main()
