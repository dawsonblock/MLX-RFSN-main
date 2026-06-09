#!/usr/bin/env python3
"""Compare two proof artifact directories and emit trend reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.proof_regression import compare_proof_dirs, load_thresholds_file, report_to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare proof artifact runs")
    parser.add_argument(
        "--profile",
        default="main10",
        help="Proof profile name used to derive default directories",
    )
    parser.add_argument(
        "--baseline-dir",
        default="",
        help="Directory with baseline proof artifacts",
    )
    parser.add_argument(
        "--current-dir",
        default="",
        help="Directory with current proof artifacts",
    )
    parser.add_argument(
        "--thresholds-file",
        default="scripts/proof_regression_thresholds.json",
        help="Optional thresholds config JSON file",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write JSON trend report",
    )
    parser.add_argument(
        "--output-md",
        default="",
        help="Optional path to write Markdown trend report",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Treat missing baseline scenarios in current run as breaches",
    )
    parser.add_argument(
        "--strict-absolute",
        action="store_true",
        help="Treat absolute quality minima warnings as hard breaches",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else Path(f"benchmarks/proof_baselines/{args.profile}")
    current_dir = Path(args.current_dir) if args.current_dir else Path(f"artifacts/proof/{args.profile}")
    thresholds = load_thresholds_file(Path(args.thresholds_file) if args.thresholds_file else None)

    report = compare_proof_dirs(
        baseline_dir=baseline_dir,
        current_dir=current_dir,
        thresholds=thresholds,
        strict_missing=args.strict_missing,
        strict_absolute=args.strict_absolute,
    )

    report_json = json.dumps(report, indent=2)
    report_md = report_to_markdown(report)

    print(report_md)

    if args.output_json:
        output_json_path = Path(args.output_json)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(report_json + "\n", encoding="utf-8")

    if args.output_md:
        output_md_path = Path(args.output_md)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)
        output_md_path.write_text(report_md, encoding="utf-8")


if __name__ == "__main__":
    main()
