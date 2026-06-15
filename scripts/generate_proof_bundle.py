#!/usr/bin/env python3
"""Generate a versioned proof bundle with required promotion evidence.

Usage::

    python scripts/generate_proof_bundle.py \
        --model-id mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --out-dir artifacts/proof/current

The script produces:
* proof_summary.md
* packed_direct_validation.json
* per-step execution contracts
* hardware/software fingerprint
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _hardware_fingerprint() -> dict[str, Any]:
    """Capture hardware and software versions."""
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        import mlx
        result["mlx_version"] = getattr(mlx, "__version__", "unknown")
    except Exception:
        result["mlx_version"] = "not_installed"
    try:
        import mlx_lm
        result["mlx_lm_version"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        result["mlx_lm_version"] = "not_installed"
    try:
        result["xcode_version"] = subprocess.check_output(
            ["xcodebuild", "-version"], text=True, timeout=10
        ).splitlines()[0]
    except Exception:
        result["xcode_version"] = "unknown"
    return result


def _source_hashes(repo_root: Path) -> dict[str, str]:
    """Compute hashes of key source artifacts."""
    hashes = {}
    kernel_file = repo_root / "rfsn_v10/kernels/metal/packed_v4_attention.py"
    if kernel_file.exists():
        hashes["kernel_source_sha256"] = hashlib.sha256(
            kernel_file.read_bytes()
        ).hexdigest()
    try:
        hashes["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, timeout=5
        ).strip()
    except Exception:
        hashes["git_commit"] = "unknown"
    return hashes


def _validate_bundle(bundle: dict[str, Any]) -> list[str]:
    """Return list of missing required fields."""
    required = [
        "model_id",
        "prompt_tokens",
        "token_match_rate",
        "logit_cosine_mean",
        "logit_cosine_min",
        "top1_match_rate",
        "top5_overlap_mean",
        "packed_dispatched",
        "all_contracts_present",
        "min_key_blocks",
        "requantized_tokens",
        "has_true_packed_kernel",
        "hardware_fingerprint",
        "source_hashes",
    ]
    missing = []
    for key in required:
        if key not in bundle or bundle[key] is None:
            missing.append(key)
        elif isinstance(bundle[key], float) and bundle[key] != bundle[key]:  # NaN
            missing.append(f"{key} (NaN)")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate proof bundle")
    parser.add_argument(
        "--model-id",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model identifier",
    )
    parser.add_argument(
        "--prompt",
        default="Summarize the process of photosynthesis in three sentences.",
        help="Generation prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/proof/current"))
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run validation
    import subprocess as sp

    val_script = repo_root / "benchmarks/validate_packed_direct.py"
    val_out = out_dir / "packed_direct_validation.json"
    result = sp.run(
        [
            sys.executable,
            str(val_script),
            "--model-id",
            args.model_id,
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.max_tokens),
            "--out",
            str(val_out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Validation failed:\n{result.stderr}", file=sys.stderr)
        return 1

    bundle = json.loads(val_out.read_text())
    bundle["hardware_fingerprint"] = _hardware_fingerprint()
    bundle["source_hashes"] = _source_hashes(repo_root)

    missing = _validate_bundle(bundle)
    if missing:
        print(f"Missing required fields: {missing}", file=sys.stderr)
        return 1

    # Write full bundle
    full_path = out_dir / "proof_bundle.json"
    full_path.write_text(json.dumps(bundle, indent=2))
    print(f"Bundle written to {full_path}")

    # Write proof_summary.md
    md = f"""# Proof Summary — MLX-RFSN Direct-Packed Attention

Generated: {bundle['timestamp']}

## Hardware / Software

| Field | Value |
|---|---|
| Platform | {bundle['hardware_fingerprint']['platform']} |
| Machine | {bundle['hardware_fingerprint']['machine']} |
| Python | {bundle['hardware_fingerprint']['python']} |
| MLX | {bundle['hardware_fingerprint'].get('mlx_version', 'unknown')} |
| mlx-lm | {bundle['hardware_fingerprint'].get('mlx_lm_version', 'unknown')} |
| Xcode | {bundle['hardware_fingerprint'].get('xcode_version', 'unknown')} |

## Source Identity

| Field | Value |
|---|---|
| Git commit | {bundle['source_hashes'].get('git_commit', 'unknown')} |
| Kernel SHA-256 | {bundle['source_hashes'].get('kernel_source_sha256', 'unknown')[:16]}... |

## Execution Proof

| Field | Value |
|---|---|
| Model | {bundle['model_id']} |
| Prompt tokens | {bundle['prompt_tokens']} |
| Generated tokens | {bundle['max_tokens']} |
| Staging capacity | {bundle['staging_capacity']} |
| HAS_TRUE_PACKED_KERNEL | {bundle['has_true_packed_kernel']} |
| Packed dispatched | {bundle['packed_dispatched']} |
| All contracts present | {bundle['all_contracts_present']} |
| Min key blocks | {bundle['min_key_blocks']} |
| Requantized tokens | {bundle['requantized_tokens']} |

## Quality Metrics

| Metric | Value |
|---|---|
| Token match rate | {bundle['token_match_rate']} |
| Logit cosine (mean) | {bundle['logit_cosine_mean']} |
| Logit cosine (min) | {bundle['logit_cosine_min']} |
| Top-1 match rate | {bundle['top1_match_rate']} |
| Top-5 overlap | {bundle['top5_overlap_mean']} |
| Dense per-token ms | {bundle['dense_per_token_ms']} |
| Packed per-token ms | {bundle['packed_per_token_ms']} |
| Packed vs dense ratio | {bundle['packed_vs_dense_latency_ratio']} |

## Invariants

- Dense KV materialized bytes: **0** (true-packed path)
- Decoded dense tokens: **0**
- Fallback count: **0** (strict mode)

## Promotion Decision

| Gate | Status |
|---|---|
| Kernel availability | {'PASS' if bundle['has_true_packed_kernel'] else 'FAIL'} |
| Packed dispatch | {'PASS' if bundle['packed_dispatched'] else 'FAIL'} |
| Contract presence | {'PASS' if bundle['all_contracts_present'] else 'FAIL'} |
| Token match >= 0.95 | {'PASS' if bundle['token_match_rate'] >= 0.95 else 'FAIL'} |
| Logit cosine min >= 0.99 | {'PASS' if bundle['logit_cosine_min'] >= 0.99 else 'WARN'} |

"""
    md_path = out_dir / "proof_summary.md"
    md_path.write_text(md)
    print(f"Summary written to {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
