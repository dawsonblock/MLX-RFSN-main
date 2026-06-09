#!/usr/bin/env python3
"""RFSN v10 Release Gate.

Runs a sequence of checks and produces a machine-readable JSON report.

Usage::

    python scripts/release_gate.py --cpu-only   # CI / no Apple Silicon required
    python scripts/release_gate.py --mlx        # Apple Silicon only
    python scripts/release_gate.py --full       # all checks + benchmark smoke

Exit code 0 = release ready.  Exit code 1 = one or more checks failed.
"""
from __future__ import annotations

# Suppress bytecode generation to avoid dirty artifact false positives
import sys

sys.dont_write_bytecode = True

import argparse
import ast
import importlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python_version() -> dict:
    vi = sys.version_info
    ok = (3, 11) <= vi < (3, 13)
    return {
        "name": "python_version",
        "passed": ok,
        "detail": f"{vi.major}.{vi.minor}.{vi.micro}",
        "message": "OK" if ok else f"Unsupported Python {vi.major}.{vi.minor}. Use 3.11 or 3.12.",
    }


def check_compile_all() -> dict:
    """Byte-compile all project source files; catch syntax errors."""
    compile_targets = [
        "rfsn_v10",
        "rfsn_v11",
        "benchmarks",
        "tests",
        "scripts",
    ]
    all_ok = True
    errors = []
    # Run compileall in subprocess with bytecode disabled
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for target in compile_targets:
        target_path = REPO_ROOT / target
        if not target_path.exists():
            continue
        cmd = [
            sys.executable, "-m", "compileall",
            "-q", "-f", str(target_path)
        ]
        result = subprocess.run(cmd, capture_output=True, env=env, cwd=REPO_ROOT)
        if result.returncode != 0:
            all_ok = False
            errors.append(target)
    return {
        "name": "compileall_full_repo",
        "passed": all_ok,
        "message": "All project .py files compile OK" if all_ok else f"Compilation errors in: {', '.join(errors)}",
    }


def check_import(module: str) -> dict:
    try:
        importlib.import_module(module)
        return {"name": f"import_{module}", "passed": True, "message": "OK"}
    except Exception as exc:
        return {"name": f"import_{module}", "passed": False, "message": str(exc)}


def check_stable_imports() -> dict:
    """Core stable modules must import without MLX."""
    modules = [
        "rfsn_v10.config",
        "rfsn_v10.errors",
        "rfsn_v10.health",
        "rfsn_v10.logging",
        "rfsn_v10.metrics",
        "rfsn_v10.bitpack",
    ]
    failures = []
    for m in modules:
        # Remove cached version for a fresh import
        for key in list(sys.modules):
            if key == m or key.startswith(m + "."):
                del sys.modules[key]
        try:
            importlib.import_module(m)
        except Exception as exc:
            failures.append(f"{m}: {exc}")
    ok = len(failures) == 0
    return {
        "name": "stable_imports",
        "passed": ok,
        "message": "All stable imports OK" if ok else "; ".join(failures),
    }


def check_no_forbidden_v10_imports() -> dict:
    """rfsn_v10 must not import from rfsn_v11, external, research, or agent_core."""
    v10_dir = REPO_ROOT / "rfsn_v10"
    forbidden = ("rfsn_v11", "external", "research", "agent_core")
    violations = []
    for py_file in sorted(v10_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    names = [node.module]
            for name in names:
                for prefix in forbidden:
                    if name == prefix or name.startswith(prefix + "."):
                        rel = py_file.relative_to(REPO_ROOT)
                        violations.append(f"{rel}: {name!r}")
    ok = len(violations) == 0
    return {
        "name": "no_forbidden_v10_imports",
        "passed": ok,
        "message": "No forbidden imports" if ok else f"{len(violations)} violation(s): " + "; ".join(violations[:5]),
    }


def check_no_placeholder_source() -> dict:
    """Scan for TODO/FIXME/NotImplemented placeholders in stable runtime."""
    v10_dir = REPO_ROOT / "rfsn_v10"
    patterns = ["raise NotImplementedError", "TODO: implement", "FIXME: implement", "pass  # TODO"]
    hits = []
    for py_file in sorted(v10_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            continue
        for pat in patterns:
            if pat in source:
                rel = py_file.relative_to(REPO_ROOT)
                hits.append(f"{rel}: {pat!r}")
    ok = len(hits) == 0
    return {
        "name": "no_placeholder_source",
        "passed": ok,
        "message": "No placeholders found" if ok else f"{len(hits)} placeholder(s): " + "; ".join(hits[:5]),
    }


def check_config_defaults() -> dict:
    """All experimental flags must default to False."""
    try:
        for key in list(sys.modules):
            if "rfsn_v10.config" in key:
                del sys.modules[key]
        import os
        for env in ["RFSN_EXPERIMENTAL_QJL", "RFSN_EXPERIMENTAL_POLAR",
                    "RFSN_EXPERIMENTAL_ADAPTIVE", "RFSN_SPARSE_DECODE_ENABLED",
                    "RFSN_QJL_ENABLED"]:
            os.environ.pop(env, None)
        from rfsn_v10.config import RFSNConfig
        cfg = RFSNConfig()
        failures = []
        if cfg.experimental.enable_qjl:
            failures.append("enable_qjl defaults to True")
        if cfg.experimental.enable_polar:
            failures.append("enable_polar defaults to True")
        if cfg.experimental.enable_adaptive:
            failures.append("enable_adaptive defaults to True")
        if cfg.runtime.sparse_decode_enabled:
            failures.append("sparse_decode_enabled defaults to True")
        if cfg.runtime.qjl_enabled:
            failures.append("qjl_enabled defaults to True")
        ok = len(failures) == 0
        return {
            "name": "config_defaults_safe",
            "passed": ok,
            "message": "All experimental flags default to False" if ok else "; ".join(failures),
        }
    except Exception as exc:
        return {"name": "config_defaults_safe", "passed": False, "message": str(exc)}


def check_project_scripts() -> dict:
    """Parse [project.scripts] from pyproject.toml and validate imports.

    Runs in a subprocess to avoid polluting the release-gate process's
    sys.modules with entry-point modules that may have side effects.
    """
    helper = """
import sys, importlib, json, tomllib
from pathlib import Path
REPO_ROOT = Path(sys.argv[1])
data = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text(encoding='utf-8'))
scripts = data.get('project', {}).get('scripts', {})
failures = []
for name, target in scripts.items():
    module_name, _, attr = target.partition(':')
    try:
        module = importlib.import_module(module_name)
        if attr and not hasattr(module, attr):
            failures.append(f"{name}: {target} missing attribute '{attr}'")
    except Exception as exc:
        failures.append(f"{name}: {target} import failed: {exc}")
print(json.dumps(failures))
"""
    try:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", helper, str(REPO_ROOT)],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            return {
                "name": "project_script_entrypoints",
                "passed": False,
                "message": (result.stderr or result.stdout).strip()[:200],
            }
        failures = json.loads(result.stdout.strip())
        ok = len(failures) == 0
        return {
            "name": "project_script_entrypoints",
            "passed": ok,
            "message": "All entry points importable" if ok else "; ".join(failures[:5]),
        }
    except Exception as exc:
        return {"name": "project_script_entrypoints", "passed": False, "message": str(exc)}


def check_no_dirty_artifacts() -> dict:
    """Scan for cache/build artifacts that should not ship."""
    forbidden_dirs = {"__pycache__", ".pytest_cache", ".rfsn_cache", ".tmp", ".mypy_cache", ".ruff_cache"}
    forbidden_suffixes = {".pyc", ".pyo", ".egg-info"}
    # Only check source dirs that should be clean, not runtime-created caches
    source_dirs = ["rfsn_v10", "rfsn_v11", "benchmarks", "tests", "scripts", "tools"]
    bad = []
    for src_dir in source_dirs:
        dir_path = REPO_ROOT / src_dir
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*"):
            if ".git" in path.parts:
                continue
            parts = set(path.parts)
            in_forbidden_dir = bool(parts & forbidden_dirs)
            if in_forbidden_dir:
                bad.append(str(path.relative_to(REPO_ROOT)))
            elif path.suffix in forbidden_suffixes:
                bad.append(str(path.relative_to(REPO_ROOT)))
    ok = len(bad) == 0
    return {
        "name": "dirty_artifact_scan",
        "passed": ok,
        "message": "No dirty artifacts found" if ok else f"{len(bad)} artifact(s): " + "; ".join(bad[:10]),
    }


def run_pytest(markers: str, label: str) -> dict:
    """Run pytest with the given marker expression."""
    cmd = [
        sys.executable, "-m", "pytest",
        "-q", "--tb=short",
        f"-m={markers}",
        f"--rootdir={REPO_ROOT}",
        str(REPO_ROOT / "tests"),
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - t0
    passed = result.returncode == 0
    # Extract counts from last line
    output_tail = (result.stdout + result.stderr).strip().splitlines()
    summary = output_tail[-1] if output_tail else ""
    return {
        "name": f"pytest_{label}",
        "passed": passed,
        "message": summary,
        "duration_s": round(elapsed, 2),
        "returncode": result.returncode,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="RFSN release gate")
    parser.add_argument("--cpu-only", action="store_true", help="Run CPU-safe checks only")
    parser.add_argument("--mlx", action="store_true", help="Include MLX Apple Silicon checks")
    parser.add_argument("--full", action="store_true", help="All checks including benchmark smoke")
    parser.add_argument("--output", default=None, help="Write JSON report to this file")
    args = parser.parse_args()

    checks = []

    # Always run - dirty artifact scan first before any cache-creating operations
    checks.append(check_python_version())
    checks.append(check_no_dirty_artifacts())
    checks.append(check_compile_all())
    checks.append(check_stable_imports())
    checks.append(check_no_forbidden_v10_imports())
    checks.append(check_no_placeholder_source())
    checks.append(check_config_defaults())
    checks.append(check_project_scripts())

    # CPU-safe pytest
    checks.append(run_pytest(
        "not mlx and not slow and not benchmark",
        "cpu_safe",
    ))

    if args.mlx or args.full:
        checks.append(run_pytest("mlx", "mlx"))

    if args.full:
        checks.append(run_pytest("benchmark", "benchmark_smoke"))

    # Summarise
    n_passed = sum(1 for c in checks if c["passed"])
    n_failed = sum(1 for c in checks if not c["passed"])
    release_ready = n_failed == 0

    # Find the pytest_cpu_safe result for test counts
    tests_passed = 0
    tests_failed = 0
    for c in checks:
        if c["name"] == "pytest_cpu_safe":
            msg = c.get("message", "")
            m = re.search(r"(\d+) passed", msg)
            if m:
                tests_passed = int(m.group(1))
            m2 = re.search(r"(\d+) failed", msg)
            if m2:
                tests_failed = int(m2.group(1))

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    report = {
        "release_ready": release_ready,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "checks_passed": n_passed,
        "checks_failed": n_failed,
        "tests_passed": tests_passed,
        "tests_failed": tests_failed,
        "mlx_tests": "included" if (args.mlx or args.full) else "skipped",
        "git_commit": git_commit,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checks": checks,
    }

    print(json.dumps(report, indent=2))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport written to: {args.output}", file=sys.stderr)

    if not release_ready:
        print("\nFAILED checks:", file=sys.stderr)
        for c in checks:
            if not c["passed"]:
                print(f"  [{c['name']}] {c['message']}", file=sys.stderr)

    return 0 if release_ready else 1


if __name__ == "__main__":
    sys.exit(main())
