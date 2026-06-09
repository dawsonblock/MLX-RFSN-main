#!/usr/bin/env python3
"""Check production validation results against regression baseline.

Compares production validation results against the quality thresholds
defined in the baseline configuration.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_baseline(baseline_path: Path) -> dict:
    """Load the baseline configuration."""
    with open(baseline_path, encoding="utf-8") as f:
        return json.load(f)


def check_threshold(
    value: float,
    threshold: dict,
    metric_name: str,
    strict: bool = False,
) -> tuple[bool, str]:
    """Check if a value meets the threshold requirements."""
    minimum = threshold.get("minimum")
    maximum = threshold.get("maximum")
    target = threshold.get("target")

    if minimum is not None and value < minimum:
        msg = f"{metric_name} {value:.4f} below minimum {minimum:.4f}"
        if strict:
            return False, f"FAIL: {msg}"
        return True, f"WARN: {msg}"

    if maximum is not None and value > maximum:
        msg = f"{metric_name} {value:.4f} above maximum {maximum:.4f}"
        if strict:
            return False, f"FAIL: {msg}"
        return True, f"WARN: {msg}"

    if target is not None and value < target:
        msg = f"{metric_name} {value:.4f} below target {target:.4f}"
        return True, f"INFO: {msg}"

    return True, f"PASS: {metric_name} {value:.4f}"


def check_results(
    results_path: Path,
    baseline_path: Path,
    strict: bool = False,
) -> tuple[bool, list[str]]:
    """Check validation results against baseline."""
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    baseline = load_baseline(baseline_path)
    thresholds = baseline["thresholds"]

    messages = []
    all_passed = True

    # Check overall metrics
    overall_dense = results.get("overall_dense_cosine", 0)
    overall_sparse = results.get("overall_sparse_cosine", 0)

    passed, msg = check_threshold(
        overall_dense,
        thresholds["overall_dense_cosine"],
        "Overall dense cosine",
        strict,
    )
    messages.append(msg)
    if not passed:
        all_passed = False

    passed, msg = check_threshold(
        overall_sparse,
        thresholds["overall_sparse_cosine"],
        "Overall sparse cosine",
        strict,
    )
    messages.append(msg)
    if not passed:
        all_passed = False

    # Check category-level metrics
    category_stats = results.get("category_statistics", {})
    for category, stats in category_stats.items():
        dense_cosine = stats.get("avg_dense_cosine", 0)
        sparse_cosine = stats.get("avg_sparse_cosine", 0)

        passed, msg = check_threshold(
            dense_cosine,
            thresholds["category_dense_cosine"],
            f"{category} dense cosine",
            strict,
        )
        messages.append(msg)
        if not passed:
            all_passed = False

        passed, msg = check_threshold(
            sparse_cosine,
            thresholds["category_sparse_cosine"],
            f"{category} sparse cosine",
            strict,
        )
        messages.append(msg)
        if not passed:
            all_passed = False

    # Check success rate
    prompts_tested = results.get("prompts_tested", 1)
    success_rate = results.get("prompts_successful", 0) / max(
        prompts_tested, 1
    )
    if success_rate < 0.95:
        msg = f"FAIL: Success rate {success_rate:.2%} below 95%"
        messages.append(msg)
        if strict:
            all_passed = False
    else:
        messages.append(f"PASS: Success rate {success_rate:.2%}")

    return all_passed, messages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check production validation regression"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to production validation results JSON",
    )
    parser.add_argument(
        "--baseline",
        default="benchmarks/production_baseline.json",
        help="Path to baseline configuration",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any threshold breach (default: warn only)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    baseline_path = Path(args.baseline)

    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        sys.exit(1)

    if not baseline_path.exists():
        print(f"ERROR: Baseline file not found: {baseline_path}")
        sys.exit(1)

    all_passed, messages = check_results(
        results_path, baseline_path, args.strict
    )

    print("Production Validation Regression Check")
    print("=" * 50)
    for msg in messages:
        print(msg)

    print("=" * 50)
    if all_passed:
        print("RESULT: All checks passed")
        sys.exit(0)
    else:
        print("RESULT: Some checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
