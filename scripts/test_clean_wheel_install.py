#!/usr/bin/env python3
"""Test clean wheel installation on Python 3.11 and 3.12.

This script builds a wheel and tests it in isolated virtual environments
to ensure the package installs correctly without contaminating the current
environment.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def build_wheel() -> Path:
    """Build the wheel and return its path."""
    print("Building wheel...")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        check=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)

    # Find the most recent wheel
    dist_dir = Path("dist")
    wheels = sorted(dist_dir.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise RuntimeError("No wheel found in dist/")
    latest_wheel = wheels[-1]
    print(f"Built wheel: {latest_wheel}")
    return latest_wheel


def test_wheel_install(wheel_path: Path, python_version: str) -> bool:
    """Test wheel installation in a clean virtual environment.

    Args:
        wheel_path: Path to the wheel file.
        python_version: Python version string (e.g., "3.11", "3.12").

    Returns:
        True if installation and basic import test succeed.
    """
    import tempfile
    import shutil

    print(f"\nTesting wheel installation on Python {python_version}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        venv_dir = Path(tmpdir) / f"venv_{python_version}"
        wheel_copy = Path(tmpdir) / wheel_path.name

        # Copy wheel to temp directory
        shutil.copy(wheel_path, wheel_copy)

        # Create virtual environment
        print(f"  Creating virtual environment...")
        try:
            subprocess.run(
                [f"python{python_version}", "-m", "venv", str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to create venv with Python {python_version}")
            print(f"  {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"  WARNING: Python {python_version} not found, skipping")
            return False

        # Install wheel
        pip_exe = venv_dir / "bin" / "pip"
        python_exe = venv_dir / "bin" / "python"

        print(f"  Installing wheel...")
        try:
            subprocess.run(
                [str(pip_exe), "install", "--no-deps", str(wheel_copy)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Failed to install wheel")
            print(f"  {e.stderr}")
            return False

        # Test basic import
        print(f"  Testing imports...")
        try:
            result = subprocess.run(
                [
                    str(python_exe), "-c",
                    "import rfsn_v10; import rfsn_v11; print('Imports OK')"
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"  {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"  ERROR: Import test failed")
            print(f"  {e.stderr}")
            return False

        print(f"  Python {python_version} test PASSED")
        return True


def main() -> int:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test clean wheel installation on multiple Python versions"
    )
    parser.add_argument(
        "--python-versions",
        nargs="+",
        default=["3.11", "3.12"],
        help="Python versions to test (default: 3.11 3.12)",
    )
    args = parser.parse_args()

    try:
        # Build wheel
        wheel_path = build_wheel()

        # Test on each Python version
        results = {}
        for py_version in args.python_versions:
            results[py_version] = test_wheel_install(wheel_path, py_version)

        # Report results
        print("\n" + "=" * 60)
        print("INSTALLATION TEST RESULTS")
        print("=" * 60)
        for py_version, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"Python {py_version}: {status}")

        # Exit with error if any test failed
        if not all(results.values()):
            print("\nSome installation tests failed")
            return 1

        print("\nAll installation tests passed")
        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
