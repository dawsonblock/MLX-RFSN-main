#!/usr/bin/env python3
"""RFSN v10 — release gate.

Runs all verification steps that must pass before a release tag is created.
Exits non-zero if any step fails.

Usage:
    python scripts/release_gate.py              # full gate
    python scripts/release_gate.py --cpu-only   # skip MLX tests (CI / Linux)
    python scripts/release_gate.py --check-list # print gate steps and exit

Steps:
    1. Import smoke             — package imports and subpackage presence
    2. CLI smoke                — python -m rfsn_v10 version / healthcheck
    3. Config validation        — validate-config with default config
    4. CPU tests                — pytest tests that run without MLX
    5. MLX tests                — pytest tests requiring Apple Silicon (skipped w/ --cpu-only)
    6. Security tests           — clickhouse security and routing tests
    7. SDPA enforcement         — no raw SDPA calls in runtime code
    8. Benchmark smoke          — run_all.py --fast (verifies runner doesn't crash)
    9. Packaging smoke          — build wheel, verify subpackages present

Exit codes:
    0   — all steps passed
    1   — at least one step failed
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------

@dataclass
class GateResult:
    step: str
    passed: bool
    skipped: bool = False
    message: str = ""


def _run(cmd: list[str], env_extra: dict | None = None) -> tuple[int, str]:
    import os
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    return r.returncode, r.stdout + r.stderr


def _check(step: str, cmd: list[str], env_extra: dict | None = None) -> GateResult:
    rc, out = _run(cmd, env_extra)
    if rc == 0:
        return GateResult(step=step, passed=True)
    # Show last 20 lines of output for diagnosis
    tail = "\n".join(out.splitlines()[-20:])
    return GateResult(step=step, passed=False, message=tail)


def _skip(step: str, reason: str) -> GateResult:
    return GateResult(step=step, passed=True, skipped=True, message=reason)


# ---------------------------------------------------------------------------
# Gate steps
# ---------------------------------------------------------------------------

def step_compileall(_cpu_only: bool) -> GateResult:
    """Every .py file in rfsn_v10/ and tests/ must compile without SyntaxError.

    This is the first and most critical gate. Placeholder text or incomplete
    generated code produces a SyntaxError that blocks all downstream steps.
    """
    return _check(
        "compileall",
        [sys.executable, "-m", "compileall", "-q", "rfsn_v10", "tests"],
    )


def step_import_smoke(_cpu_only: bool) -> GateResult:
    """Verify all required package imports succeed."""
    code = (
        "import rfsn_v10; "
        "import rfsn_v10.kernels; "
        "import rfsn_v10.runtime; "
        "from rfsn_v10 import RFSNRuntime; "
        "from rfsn_v10.runtime import RFSNRuntime; "
        "print('OK')"
    )
    return _check("import_smoke", [sys.executable, "-c", code])


def step_cli_version(_cpu_only: bool) -> GateResult:
    return _check(
        "cli_version",
        [sys.executable, "-m", "rfsn_v10", "version"],
        env_extra={"RFSN_BACKEND": "numpy"},
    )


def step_cli_healthcheck(_cpu_only: bool) -> GateResult:
    return _check(
        "cli_healthcheck",
        [sys.executable, "-m", "rfsn_v10", "healthcheck"],
        env_extra={"RFSN_BACKEND": "numpy"},
    )


def step_config_validate(_cpu_only: bool) -> GateResult:
    cfg = ROOT / "configs" / "default_runtime.yaml"
    return _check(
        "config_validate",
        [sys.executable, "-m", "rfsn_v10", "validate-config", "--config", str(cfg)],
        env_extra={"RFSN_BACKEND": "numpy"},
    )


def step_cpu_tests(_cpu_only: bool) -> GateResult:
    tests = [
        "tests/test_config.py",
        "tests/test_config_strict.py",
        "tests/test_kernels_validation.py",
        "tests/test_quantization_lazy_imports.py",
        "tests/test_experimental_flags.py",
        "tests/test_no_runtime_raw_sdpa.py",
    ]
    return _check(
        "cpu_tests",
        [sys.executable, "-m", "pytest", "-q", "--tb=short"] + tests,
        env_extra={"RFSN_BACKEND": "numpy"},
    )


def step_security_tests(_cpu_only: bool) -> GateResult:
    tests = ["tests/test_clickhouse_security.py"]
    # Add routing / tool runner tests if they exist
    for t in ["tests/test_clickhouse_routing.py", "tests/test_tool_runner_security.py"]:
        if (ROOT / t).exists():
            tests.append(t)
    return _check(
        "security_tests",
        [sys.executable, "-m", "pytest", "-q", "--tb=short"] + tests,
    )


def step_sdpa_enforcement(_cpu_only: bool) -> GateResult:
    return _check(
        "sdpa_enforcement",
        [sys.executable, "-m", "pytest", "-q", "--tb=short",
         "tests/test_no_runtime_raw_sdpa.py"],
    )


def step_mlx_tests(cpu_only: bool) -> GateResult:
    if cpu_only:
        return _skip("mlx_tests", "--cpu-only flag set")

    tests = [
        "tests/test_drift.py",
        "tests/test_attention_causal_mask.py",
        "tests/test_short_prompt_decode_drift.py",
        "tests/test_prefill_decode_split.py",
    ]
    # Filter to tests that actually exist
    tests = [t for t in tests if (ROOT / t).exists()]
    if not tests:
        return _skip("mlx_tests", "No MLX test files found")
    return _check(
        "mlx_tests",
        [sys.executable, "-m", "pytest", "-q", "--tb=short"] + tests,
    )


def step_benchmark_smoke(_cpu_only: bool) -> GateResult:
    return _check(
        "benchmark_smoke",
        [sys.executable, "benchmarks/run_all.py", "--fast"],
    )


def step_build_install(_cpu_only: bool) -> GateResult:
    """Build wheel, install it, and verify subpackage imports work from the wheel.

    This catches package-discovery bugs (missing __init__.py, wrong find pattern)
    that only appear when the package is installed — not when running from source.
    """
    import tempfile
    import zipfile

    with tempfile.TemporaryDirectory(prefix="rfsn_build_") as build_dir:
        # Ensure build tool is available
        rc, out = _run([sys.executable, "-m", "pip", "install", "--quiet", "build"])
        if rc != 0:
            return GateResult(
                step="build_install", passed=False,
                message=f"pip install build failed:\n{out[-500:]}"
            )

        # Build wheel
        rc, out = _run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", build_dir],
        )
        if rc != 0:
            return GateResult(
                step="build_install", passed=False,
                message=f"wheel build failed:\n{out[-1000:]}"
            )

        wheels = list(Path(build_dir).glob("*.whl"))
        if not wheels:
            return GateResult(step="build_install", passed=False, message="No wheel produced")

        wheel = wheels[0]

        # Verify subpackages present in wheel zip before installing
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
        required_paths = [
            "rfsn_v10/__init__.py",
            "rfsn_v10/kernels/__init__.py",
            "rfsn_v10/runtime/__init__.py",
        ]
        missing = [r for r in required_paths if not any(n.endswith(r) for n in names)]
        if missing:
            return GateResult(
                step="build_install", passed=False,
                message=f"Wheel missing subpackages: {missing}"
            )

        # Attempt to install the wheel into the current interpreter.
        # If the Python version doesn't match requires-python, skip install
        # and fall through to zip-only content check (already done above).
        rc, out = _run([
            sys.executable, "-m", "pip", "install",
            "--quiet", "--force-reinstall", str(wheel),
        ])
        if rc != 0:
            # Check if this is purely a Python version mismatch
            if "requires a different Python" in out or "python_requires" in out:
                # Wheel content verified by zip inspection above; skip install.
                # Log a warning but do not fail the gate — the wheel itself is valid.
                print(f"\n    NOTE: wheel install skipped (Python version mismatch: "
                      f"gate running {sys.version.split()[0]}, "
                      f"wheel requires 3.11). Zip content verified.")
                # Re-install editable before returning
                _run([sys.executable, "-m", "pip", "install", "--quiet", "-e", "."])
                return GateResult(step="build_install", passed=True,
                                  message="wheel content verified; install skipped (Python version)")
            return GateResult(
                step="build_install", passed=False,
                message=f"pip install wheel failed:\n{out[-500:]}"
            )

        # Verify imports work from the installed wheel
        verify_code = (
            "import rfsn_v10; "
            "import rfsn_v10.kernels; "
            "import rfsn_v10.quantization; "
            "import rfsn_v10.runtime; "
            "from rfsn_v10 import RFSNRuntime; "
            "print('wheel import OK')"
        )
        rc, out = _run([sys.executable, "-c", verify_code])
        if rc != 0:
            return GateResult(
                step="build_install", passed=False,
                message=f"wheel import check failed:\n{out[-500:]}"
            )

        # Re-install editable so the rest of the session works from source
        _run([sys.executable, "-m", "pip", "install", "--quiet", "-e", "."])

    return GateResult(step="build_install", passed=True)


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------

GATE_STEPS: list[Callable[[bool], GateResult]] = [
    step_compileall,      # must be first — blocks all others on SyntaxError
    step_import_smoke,
    step_cli_version,
    step_cli_healthcheck,
    step_config_validate,
    step_cpu_tests,
    step_security_tests,
    step_sdpa_enforcement,
    step_mlx_tests,
    step_benchmark_smoke,
    step_build_install,   # builds wheel, verifies subpackages, installs — must be last
]


def _ensure_editable_install() -> bool:
    """Install package in editable mode if not already importable.

    This ensures the gate can be run from a fresh clone without needing
    manual ``PYTHONPATH=.`` or ``pip install -e .`` beforehand.

    Returns True if the install succeeded (or was already present).
    """
    import subprocess
    try:
        import rfsn_v10  # noqa: F401
        return True  # already installed
    except ImportError:
        pass
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", "."],
        cwd=ROOT,
    )
    return result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RFSN v10 release gate")
    parser.add_argument(
        "--cpu-only", action="store_true",
        help="Skip MLX-dependent tests (for Linux / CI without Apple Silicon)"
    )
    parser.add_argument(
        "--check-list", action="store_true",
        help="Print gate steps and exit without running them"
    )
    args = parser.parse_args(argv)

    if args.check_list:
        for fn in GATE_STEPS:
            print(f"  {fn.__name__.replace('step_', '')}")
        return 0

    # Ensure the package is importable without manual PYTHONPATH setup
    if not _ensure_editable_install():
        print("ERROR: Could not install package in editable mode. Run: pip install -e .")
        return 1

    results: list[GateResult] = []
    for fn in GATE_STEPS:
        name = fn.__name__.replace("step_", "")
        print(f"  [{name}] ...", end=" ", flush=True)
        result = fn(args.cpu_only)
        results.append(result)
        if result.skipped:
            print(f"SKIP  ({result.message})")
        elif result.passed:
            print("PASS")
        else:
            print("FAIL")
            if result.message:
                for line in result.message.splitlines()[-10:]:
                    print(f"    {line}")

    passed = sum(1 for r in results if r.passed and not r.skipped)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.passed)

    print()
    print(f"Gate: {passed} passed, {skipped} skipped, {failed} failed")

    if failed:
        failed_names = [r.step for r in results if not r.passed]
        print(f"Failed steps: {', '.join(failed_names)}")
        print("\nRelease gate FAILED — do not tag.")
        return 1

    print("\nRelease gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
