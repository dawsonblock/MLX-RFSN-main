#!/usr/bin/env python3
"""Generate proof plots from KV and E2E benchmark artifacts."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z6hQAAAAASUVORK5CYII="
)


def _load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("runs", []))


def _write_placeholder_png(path: Path, note: str) -> None:
    path.write_bytes(TINY_PNG)
    path.with_suffix(".txt").write_text(note + "\n", encoding="utf-8")


def _load_optional(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_with_matplotlib(kv_runs: list[dict], e2e_runs: list[dict], output_dir: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    kv_scenarios = [run.get("scenario", "") for run in kv_runs]
    kv_store = [float(run.get("store_latency_ms", 0.0)) for run in kv_runs]
    kv_retrieve = [float(run.get("retrieve_latency_ms", 0.0)) for run in kv_runs]
    kv_value_cos = [float(run.get("value_cosine_sim", 0.0)) for run in kv_runs]
    kv_compression_ratio = [float(run.get("compression_ratio", 0.0)) for run in kv_runs]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    x = list(range(len(kv_scenarios)))
    ax.plot(x, kv_store, marker="o", label="store_latency_ms")
    ax.plot(x, kv_retrieve, marker="o", label="retrieve_latency_ms")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("shape=", "")[:28] for s in kv_scenarios], rotation=45, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("KV Cache Latency by Scenario")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "kv_latency.png", dpi=150)
        fig.savefig(output_dir / "kv_cache_latency.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, kv_value_cos, marker="o", color="#2d6a4f")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("shape=", "")[:28] for s in kv_scenarios], rotation=45, ha="right")
    ax.set_ylabel("Cosine")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("KV Value Cosine Similarity")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "kv_quality_cosine.png", dpi=150)
        fig.savefig(output_dir / "kv_reconstruction_quality.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, kv_compression_ratio, marker="o", color="#6a4c93")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("shape=", "")[:28] for s in kv_scenarios], rotation=45, ha="right")
    ax.set_ylabel("Packed / Original")
    ax.set_title("KV Compression Ratio")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "compression_ratio.png", dpi=150)
    finally:
        plt.close(fig)

    e2e_scenarios = [run.get("scenario", "") for run in e2e_runs]
    e2e_miss = [float(run.get("cache_miss_total_latency_ms", 0.0)) for run in e2e_runs]
    e2e_hit = [float(run.get("cache_hit_total_latency_ms", 0.0)) for run in e2e_runs]
    e2e_quant = [float(run.get("quant_audit_cosine") or 0.0) for run in e2e_runs]
    e2e_sparse = [float(run.get("sparse_audit_cosine") or 0.0) for run in e2e_runs]
    e2e_topk = [float(run.get("top_k_ratio") or 0.0) for run in e2e_runs]

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    x2 = list(range(len(e2e_scenarios)))
    ax.plot(x2, e2e_miss, marker="o", label="cache_miss_total_latency_ms")
    ax.plot(x2, e2e_hit, marker="o", label="cache_hit_total_latency_ms")
    ax.set_xticks(x2)
    ax.set_xticklabels(e2e_scenarios, rotation=35, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("E2E Latency by Scenario")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "e2e_latency.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x2, e2e_quant, marker="o", label="quant_audit_cosine")
    ax.plot(x2, e2e_sparse, marker="o", label="sparse_audit_cosine")
    ax.set_xticks(x2)
    ax.set_xticklabels(e2e_scenarios, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Cosine")
    ax.set_title("E2E Quality Cosine")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "e2e_quality_cosine.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x2, e2e_sparse, marker="o", color="#1d3557")
    ax.set_xticks(x2)
    ax.set_xticklabels(e2e_scenarios, rotation=35, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Cosine")
    ax.set_title("E2E Sparse Quality Cosine")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "e2e_sparse_quality.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(e2e_topk, e2e_sparse, color="#264653")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Top-k ratio")
    ax.set_ylabel("Sparse audit cosine")
    ax.set_title("Sparse Quality vs Top-k")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "sparse_quality_vs_topk.png", dpi=150)
    finally:
        plt.close(fig)

    return True


def _plot_kernel_reference_vs_metal(
    *,
    kernel_payload: dict | None,
    output_dir: Path,
) -> bool:
    if kernel_payload is None:
        return False

    runs = list(kernel_payload.get("runs", []))
    if not runs:
        return False

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    by_shape: dict[str, dict[str, float]] = {}
    speedup_by_shape: dict[str, float] = {}
    error_by_shape: dict[str, float] = {}
    for row in runs:
        shape = str(row.get("shape", ""))
        mode = str(row.get("mode", ""))
        latency = float(
            row.get("latency_ms_p50", row.get("retrieve_latency_ms", 0.0))
        )
        by_shape.setdefault(shape, {})[mode] = latency
        if mode in ("metal_multikernel_dequant_wht_sign", "metal_fused_dequant_wht_sign"):
            speedup_by_shape[shape] = float(row.get("speedup_vs_reference", 0.0))
            error_by_shape[shape] = float(row.get("max_abs_diff_vs_reference", 0.0))

    shapes = sorted(by_shape.keys())
    ref = [by_shape[s].get("sequential_reference", 0.0) for s in shapes]
    metal_multi = [
        by_shape[s].get("metal_multikernel_dequant_wht_sign", 0.0)
        for s in shapes
    ]
    metal_fused = [
        by_shape[s].get("metal_fused_dequant_wht_sign", 0.0)
        for s in shapes
    ]

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    x = list(range(len(shapes)))
    ax.plot(x, ref, marker="o", label="sequential reference")
    ax.plot(x, metal_multi, marker="o", label="multi-kernel Metal")
    if any(metal_fused):
        ax.plot(x, metal_fused, marker="o", label="fused Metal")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [s.replace("(", "").replace(")", "") for s in shapes],
        rotation=35, ha="right",
    )
    ax.set_ylabel("Retrieve latency (ms)")
    ax.set_title("Kernel Reference vs Metal")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "kernel_reference_vs_metal.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    speedup = [speedup_by_shape.get(s, 0.0) for s in shapes]
    ax.bar(list(range(len(shapes))), speedup, color="#2a9d8f")
    ax.set_xticks(list(range(len(shapes))))
    ax.set_xticklabels(
        [s.replace("(", "").replace(")", "") for s in shapes],
        rotation=35, ha="right",
    )
    ax.set_ylabel("Speedup vs reference")
    ax.set_title("Kernel Speedup by Shape")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "kernel_speedup_by_shape.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)
    err = [error_by_shape.get(s, 0.0) for s in shapes]
    ax.bar(list(range(len(shapes))), err, color="#e76f51")
    ax.set_xticks(list(range(len(shapes))))
    ax.set_xticklabels(
        [s.replace("(", "").replace(")", "") for s in shapes],
        rotation=35, ha="right",
    )
    ax.set_ylabel("Max abs diff vs reference")
    ax.set_title("Kernel Error vs Reference")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "kernel_error_vs_reference.png", dpi=150)
    finally:
        plt.close(fig)

    return True


def _plot_real_model(
    *,
    real_payload: dict | None,
    output_dir: Path,
) -> bool:
    if real_payload is None:
        return False
    configs = real_payload.get("configs", [])
    test_configs = [c for c in configs if c.get("name") != "baseline_fp16"]
    if not test_configs:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    names = [c["name"] for c in test_configs]
    cosine = [c.get("logit_cosine_mean", 0.0) for c in test_configs]
    top5 = [c.get("top5_overlap_mean", 0.0) for c in test_configs]
    ppl_delta = [abs(c.get("perplexity_delta", 0.0)) for c in test_configs]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    x = list(range(len(names)))
    ax.bar(x, cosine, color="#2a9d8f")
    ax.axhline(y=0.995, color="r", linestyle="--", label="threshold=0.995")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Logit cosine mean")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Real-Model Logit Cosine")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "real_model_logit_cosine.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, top5, color="#264653")
    ax.axhline(y=0.95, color="r", linestyle="--", label="threshold=0.95")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Top-5 overlap mean")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Real-Model Top-k Overlap")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "real_model_topk_overlap.png", dpi=150)
    finally:
        plt.close(fig)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x, ppl_delta, color="#e76f51")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("|Perplexity delta|")
    ax.set_title("Real-Model Perplexity Delta")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "real_model_perplexity_delta.png", dpi=150)
    finally:
        plt.close(fig)

    return True


def _plot_long_context(
    *,
    long_payload: dict | None,
    output_dir: Path,
) -> bool:
    if long_payload is None:
        return False
    contexts = long_payload.get("contexts", [])
    if not contexts:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    tokens = [c["tokens"] for c in contexts]
    # For each context, average cosine across non-baseline configs
    avg_cosine = []
    for ctx in contexts:
        vals = [
            cfg.get("logit_cosine_mean", 0.0)
            for cfg in ctx.get("configs", [])
            if cfg.get("name") != "baseline_fp16" and not cfg.get("oom")
        ]
        avg_cosine.append(sum(vals) / len(vals) if vals else 0.0)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tokens, avg_cosine, marker="o", color="#2a9d8f")
    ax.axhline(y=0.995, color="r", linestyle="--", label="threshold=0.995")
    ax.set_xlabel("Context tokens")
    ax.set_ylabel("Mean logit cosine")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Long-Context Quality")
    ax.legend()
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "long_context_quality.png", dpi=150)
    finally:
        plt.close(fig)

    # Latency per context
    avg_latency = []
    for ctx in contexts:
        vals = [
            cfg.get("latency_ms", 0.0)
            for cfg in ctx.get("configs", [])
            if cfg.get("name") != "baseline_fp16" and not cfg.get("oom")
        ]
        avg_latency.append(sum(vals) / len(vals) if vals else 0.0)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tokens, avg_latency, marker="o", color="#e76f51")
    ax.set_xlabel("Context tokens")
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Long-Context Latency")
    fig.tight_layout()
    try:
        fig.savefig(output_dir / "long_context_latency.png", dpi=150)
    finally:
        plt.close(fig)

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark plot artifacts")
    parser.add_argument("--input-dir", default="artifacts/proof/main25")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kv_runs = _load_runs(input_dir / "kv_cache_runs.json")
    e2e_runs = _load_runs(input_dir / "e2e_scenarios.json")
    kernel_payload = _load_optional(input_dir / "kernel_benchmark.json")
    real_payload = _load_optional(input_dir / "real_model_validation.json")
    long_payload = _load_optional(input_dir / "long_context_validation.json")

    used_matplotlib = _plot_with_matplotlib(kv_runs, e2e_runs, output_dir)
    if not used_matplotlib:
        _write_placeholder_png(
            output_dir / "kv_latency.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "kv_cache_latency.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "kv_quality_cosine.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "kv_reconstruction_quality.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "compression_ratio.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "e2e_latency.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "e2e_quality_cosine.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "e2e_sparse_quality.png",
            "matplotlib unavailable; placeholder only",
        )
        _write_placeholder_png(
            output_dir / "sparse_quality_vs_topk.png",
            "matplotlib unavailable; placeholder only",
        )

    # Remove obsolete placeholder files if they exist
    (output_dir / "custom_kernel_alpha_pending_benchmark.png").unlink(missing_ok=True)
    (output_dir / "custom_kernel_alpha_pending_benchmark.txt").unlink(missing_ok=True)
    (output_dir / "kernel_reference_vs_custom.png").unlink(missing_ok=True)
    (output_dir / "kernel_reference_vs_custom.txt").unlink(missing_ok=True)

    wrote_kernel_plot = _plot_kernel_reference_vs_metal(
        kernel_payload=kernel_payload,
        output_dir=output_dir,
    )
    if not wrote_kernel_plot:
        _write_placeholder_png(
            output_dir / "kernel_reference_vs_metal.png",
            "Placeholder: kernel reference vs metal comparison "
            "requires dedicated paired kernel benchmark dataset.",
        )

    wrote_real = _plot_real_model(
        real_payload=real_payload,
        output_dir=output_dir,
    )
    if not wrote_real:
        _write_placeholder_png(
            output_dir / "real_model_logit_cosine.png",
            "Placeholder: real-model data not available.",
        )
        _write_placeholder_png(
            output_dir / "real_model_topk_overlap.png",
            "Placeholder: real-model data not available.",
        )
        _write_placeholder_png(
            output_dir / "real_model_perplexity_delta.png",
            "Placeholder: real-model data not available.",
        )

    wrote_long = _plot_long_context(
        long_payload=long_payload,
        output_dir=output_dir,
    )
    if not wrote_long:
        _write_placeholder_png(
            output_dir / "long_context_quality.png",
            "Placeholder: long-context data not available.",
        )
        _write_placeholder_png(
            output_dir / "long_context_latency.png",
            "Placeholder: long-context data not available.",
        )

    print(f"Wrote plot artifacts to {output_dir}")


if __name__ == "__main__":
    main()
