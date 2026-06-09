#!/usr/bin/env python3
"""Standalone long-context KV validation runner.

Delegates to validate_real_model_kv.py with --contexts flag.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Long-context KV validation"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--contexts",
        default="512,1024,2048",
        help="Comma-separated token counts",
    )
    parser.add_argument(
        "--positions",
        type=int,
        default=64,
        help="Number of decode positions to evaluate (causal NLL scoring)",
    )
    parser.add_argument(
        "--configs",
        default="baseline_fp16,mixed_L0-1k8v4_restk6v4_gs64,"
        "k8_v4_gs64,k8_v5_gs64,k8_v3_gs64,"
        "k6_v6_gs64,k8_v4_gs32,k8_v5_gs32,k4_v4_gs64",
        help="Comma-separated config names",
    )
    parser.add_argument(
        "--out",
        default="artifacts/proof/main28/long_context_validation.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    script = Path(__file__).parent / "validate_real_model_kv.py"
    cmd = [
        sys.executable, str(script),
        "--model", args.model,
        "--contexts", args.contexts,
        "--positions", str(args.positions),
        "--configs", args.configs,
        "--long-context-out", args.out,
    ]
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
