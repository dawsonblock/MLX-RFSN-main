#!/usr/bin/env python3
"""Generate layer policy from per-layer sensitivity data.

Input:  artifacts/proof/experimental/per_layer_sensitivity.json
Output: artifacts/proof/experimental/layer_policy.json

Logic:
  - Read sensitivity results
  - For each layer, pick the mode with best quality/compression tradeoff
  - Apply conservative bias for early layers (0-3)
  - Apply aggressive bias for late layers (16+) if safe
  - Always include default fallback

Usage:
  python scripts/generate_layer_policy.py \
      --in artifacts/proof/experimental/per_layer_sensitivity.json \
      --out artifacts/proof/experimental/layer_policy.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Policy generation
# ---------------------------------------------------------------------------

def generate_layer_policy(sensitivity_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a layer policy from sensitivity results.

    Args:
        sensitivity_results: List of per-layer sensitivity dicts.

    Returns:
        Policy dict with default_fallback and per-layer recommendations.
    """
    policy: dict[str, Any] = {
        "description": (
            "layers 0-3: conservative precision; "
            "layers 4-15: standard precision; "
            "layers 16+: aggressive precision if safe"
        ),
        "default_fallback": {
            "k_bits": 8,
            "v_bits": 5,
            "group_size": 64,
            "variant": "k8_v5_gs64",
        },
        "layers": {},
    }

    for row in sensitivity_results:
        layer_id = row.get("layer_id")
        if layer_id is None:
            continue

        rec = {
            "k_bits": row.get("recommended_k_bits", 8),
            "v_bits": row.get("recommended_v_bits", 5),
            "group_size": row.get("recommended_group_size", 64),
            "variant": row.get("recommended_variant", "k8_v5_gs64"),
        }

        # Apply policy biases
        if layer_id <= 3:
            rec["bias"] = "conservative"
            rec["k_bits"] = max(rec["k_bits"], 8)
            rec["v_bits"] = max(rec["v_bits"], 5)
            rec["group_size"] = min(rec.get("group_size", 64), 64)
        elif 4 <= layer_id <= 15:
            rec["bias"] = "standard"
        else:
            rec["bias"] = "aggressive"
            # Only allow aggressive if quality is good enough
            best_variant = next(
                (v for v in row.get("variants", []) if v.get("variant") == rec["variant"]),
                None,
            )
            if best_variant is not None:
                if (
                    best_variant.get("cosine_drop", 1.0) < 0.005
                    and best_variant.get("KL_delta", 1.0) < 0.001
                ):
                    rec["k_bits"] = max(4, rec["k_bits"] - 2)
                    rec["v_bits"] = max(3, rec["v_bits"] - 2)
                else:
                    rec["note"] = "aggressive blocked by quality"
            else:
                rec["note"] = "variant not found; using standard"

        policy["layers"][str(layer_id)] = rec

    return policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate layer policy from sensitivity data")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=Path("artifacts/proof/experimental/per_layer_sensitivity.json"),
        help="Input sensitivity JSON path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/proof/experimental/layer_policy.json"),
        help="Output policy JSON path",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        raise SystemExit(f"Input file not found: {args.input_path}")

    data = json.loads(args.input_path.read_text(encoding="utf-8"))
    sensitivity_results = data.get("layers", [])
    if not sensitivity_results:
        raise SystemExit("No sensitivity layers found in input.")

    print(f"Read {len(sensitivity_results)} layers from {args.input_path}")

    policy = generate_layer_policy(sensitivity_results)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Wrote policy to {args.out}")

    # Print a quick summary
    conservative = sum(1 for v in policy["layers"].values() if v.get("bias") == "conservative")
    standard = sum(1 for v in policy["layers"].values() if v.get("bias") == "standard")
    aggressive = sum(1 for v in policy["layers"].values() if v.get("bias") == "aggressive")
    print(
        f"Summary: {conservative} conservative, {standard} standard, {aggressive} aggressive"
    )


if __name__ == "__main__":
    main()
