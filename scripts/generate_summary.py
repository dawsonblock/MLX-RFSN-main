#!/usr/bin/env python3
"""Generate summary.json artifact index for Main 28."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_ARTIFACTS = [
    "kernel_benchmark.json",
    "fused_kernel_benchmark.json",
    "optimization_benchmark.json",
    "real_model_validation.json",
    "long_context_validation.json",
    "generation_smoke.json",
    "generation_throughput.json",
    "proof_summary.md",
    "mlx_test_summary.md",
    "mlx_pytest_raw.log",
    "mlx_pytest_junit.xml",
    "main28_release_manifest.json",
]

OPTIONAL_ARTIFACTS = [
    "per_layer_sensitivity.json",
    "targeted_layer_protection.json",
]


def main() -> int:
    proof_dir = Path("artifacts/proof/main28")

    artifacts = {}
    missing_required = []
    missing_optional = []

    for name in REQUIRED_ARTIFACTS:
        if (proof_dir / name).exists():
            artifacts[name.replace(".", "_")] = name
        else:
            missing_required.append(name)

    for name in OPTIONAL_ARTIFACTS:
        if not (proof_dir / name).exists():
            missing_optional.append(name)

    if missing_required:
        print(f"ERROR: Missing required artifacts: {missing_required}")
        return 1

    summary = {
        "release": "main28",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
        "status": "complete" if not missing_optional else "partial",
        "missing_optional": missing_optional,
    }

    output_path = proof_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Generated {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
