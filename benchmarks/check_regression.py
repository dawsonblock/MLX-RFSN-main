#!/usr/bin/env python3
"""Benchmark regression checker — Week 6.

Compares ``benchmarks/results/latest.json`` against a baseline JSON.
Exits non-zero if any metric regresses by more than the threshold.

Usage:
    python benchmarks/check_regression.py baseline.json latest.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

THRESHOLD = 0.05  # 5 %


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(baseline: dict, current: dict) -> int:
    regressions = 0
    for key in baseline:
        if key == "metadata":
            continue
        if key not in current:
            print(f"MISS  {key}: missing in current run")
            regressions += 1
            continue
        base = baseline[key]
        curr = current[key]
        for metric in base:
            if metric not in curr:
                continue
            b_val = float(base[metric])
            c_val = float(curr[metric])
            # For latency metrics (ms), higher is regression.
            # For speedup, lower is regression.
            if metric == "speedup":
                delta = (b_val - c_val) / max(b_val, 1e-9)
                direction = "slower"
            else:
                delta = (c_val - b_val) / max(b_val, 1e-9)
                direction = "slower"
            if delta > THRESHOLD:
                print(
                    f"REGRESSION  {key}/{metric}: "
                    f"{b_val:.4f} → {c_val:.4f}  "
                    f"({delta:+.1%} {direction})"
                )
                regressions += 1
            else:
                print(
                    f"OK          {key}/{metric}: "
                    f"{b_val:.4f} → {c_val:.4f}  ({delta:+.1%})"
                )
    return regressions


def main() -> int:
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} baseline.json latest.json",
            file=sys.stderr,
        )
        return 2
    baseline = load(Path(sys.argv[1]))
    current = load(Path(sys.argv[2]))
    regressions = compare(baseline, current)
    if regressions:
        print(f"\n{regressions} regression(s) exceed {THRESHOLD:.0%} threshold")
        return 1
    print(f"\nNo regressions exceed {THRESHOLD:.0%} threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
