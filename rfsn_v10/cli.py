"""RFSN v10 CLI entry points."""

from __future__ import annotations

import argparse
import json
import sys

from rfsn_v10.health import get_health_checker


def _health() -> int:
    checker = get_health_checker()
    report = checker.get_health_report()
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("overall_status") == "healthy" else 1


def main(argv: list[str] | None = None) -> int:
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(prog="rfsn-benchmark")
    sub = parser.add_subparsers(dest="command")

    health_parser = sub.add_parser("health", help="Run health checks")
    health_parser.set_defaults(func=_health)

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    return args.func()


if __name__ == "__main__":
    sys.exit(main())
