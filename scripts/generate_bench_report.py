#!/usr/bin/env python3
"""Generate consolidated benchmark report from individual benchmark outputs."""
from __future__ import annotations

import json
from pathlib import Path


def extract_json(text: str) -> dict:
    start = text.find("{")
    if start >= 0:
        return json.loads(text[start:])
    return {}


def main() -> None:
    root = Path("artifacts/bench/current")
    root.mkdir(parents=True, exist_ok=True)

    kv = extract_json((root / "kv_cache_results.json").read_text())
    bitpack = extract_json((root / "bitpack_results.json").read_text())
    attn = extract_json((root / "attention_results.json").read_text())

    quality_rows = []
    for row in kv.get("results", []):
        quality_rows.append({
            "shape": row.get("shape", ""),
            "k_cosine": round(row.get("key_cosine_sim", 0), 6),
            "v_cosine": round(row.get("value_cosine_sim", 0), 6),
            "compression": row.get("compression_ratio", 0),
            "original_mb": row.get("original_bytes", 0) // (1024 * 1024),
            "packed_mb": row.get("packed_bytes", 0) // (1024 * 1024),
        })

    results = {
        "metadata": {
            "platform": "Apple Silicon / MLX",
            "date": "2026-06-08",
            "suite": "RFSN v10 Stable Path Validation",
        },
        "kv_cache_quality": quality_rows[:5],
        "quality_gates": {
            "cosine_similarity_min": 0.999,
            "kl_divergence_max": 1e-4,
            "top5_overlap_min": 0.95,
        },
        "bitpack_summary": bitpack.get("summary", {}),
        "attention_summary": attn.get("summary", {}),
    }

    (root / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Generated results.json with {len(quality_rows)} KV cache configs")

    # Generate CSV
    import csv
    with open(root / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shape", "k_cosine", "v_cosine", "compression", "original_mb", "packed_mb"])
        writer.writeheader()
        writer.writerows(quality_rows)
    print(f"Generated results.csv")

    # Generate Markdown summary
    md = ["# RFSN v10 Benchmark Results\n"]
    md.append(f"**Platform**: {results['metadata']['platform']}\n")
    md.append(f"**Date**: {results['metadata']['date']}\n")
    md.append("## KV Cache Quality (8-bit Grouped Quantization)\n")
    md.append("| Shape | K Cosine | V Cosine | Compression | Orig MB | Packed MB |")
    md.append("|-------|----------|----------|-------------|---------|-----------|")
    for row in quality_rows[:5]:
        md.append(f"| {row['shape']} | {row['k_cosine']} | {row['v_cosine']} | {row['compression']:.2f}x | {row['original_mb']} | {row['packed_mb']} |")

    md.append("\n## Quality Thresholds\n")
    for k, v in results["quality_gates"].items():
        md.append(f"- **{k}**: {v}")

    (root / "results.md").write_text("\n".join(md))
    print(f"Generated results.md")


if __name__ == "__main__":
    main()
