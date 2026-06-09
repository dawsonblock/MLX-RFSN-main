#!/usr/bin/env python3
"""Check that Python version is supported (3.11 or 3.12).

Run this before install or in the release gate::

    python scripts/check_python_version.py

Exit code 0 = OK.  Exit code 1 = unsupported version with clear message.
"""
import sys

MIN = (3, 11)
MAX = (3, 13)  # exclusive

vi = sys.version_info
if not (MIN <= vi < MAX):
    raise SystemExit(
        f"Unsupported Python {vi.major}.{vi.minor}.{vi.micro}.\n"
        f"MLX-RFSN requires Python 3.11 or 3.12.\n"
        f"Install with: pyenv install 3.12.8 && pyenv local 3.12.8"
    )
print(f"Python version OK: {vi.major}.{vi.minor}.{vi.micro}")
