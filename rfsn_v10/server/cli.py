#!/usr/bin/env python3
"""RFSN v10 server command-line interface.

Entry points::

    rfsn-server --model <model-id>             # start server
    rfsn-health                                # check server health
    rfsn-config-check                          # validate config + env
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def main(argv: list[str] | None = None) -> None:
    """Start the RFSN v10 inference server.

    This is the ``rfsn-server`` entry point.
    """
    parser = argparse.ArgumentParser(
        prog="rfsn-server",
        description="RFSN v10 local MLX inference server (OpenAI-compatible)",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL_ID",
        help="HuggingFace model ID or local path (overrides RFSN_MODEL_ID env)",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind host (default: 127.0.0.1; use 0.0.0.0 for LAN)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--kv-compression",
        action="store_true",
        default=None,
        help="Enable v10 KV compression (default: off; benchmark before enabling)",
    )
    parser.add_argument(
        "--no-kv-compression",
        action="store_true",
        default=False,
        help="Disable KV compression",
    )
    parser.add_argument(
        "--require-api-key",
        action="store_true",
        default=False,
        help="Require Authorization: Bearer <key>",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (required when --require-api-key is set)",
    )
    parser.add_argument(
        "--backend",
        choices=["mlx", "numpy"],
        default=None,
        help="Compute backend (default: mlx)",
    )
    args = parser.parse_args(argv)

    # Apply CLI args as env overrides (server reads from env)
    if args.model:
        os.environ["RFSN_MODEL_ID"] = args.model
    if args.host:
        os.environ["RFSN_HOST"] = args.host
    if args.port:
        os.environ["RFSN_PORT"] = str(args.port)
    if args.backend:
        os.environ["RFSN_BACKEND"] = args.backend
    if args.no_kv_compression:
        os.environ["RFSN_ENABLE_KV_COMPRESSION"] = "false"
    if args.kv_compression:
        os.environ["RFSN_ENABLE_KV_COMPRESSION"] = "true"
    if args.require_api_key:
        os.environ["RFSN_REQUIRE_API_KEY"] = "true"
    if args.api_key:
        os.environ["RFSN_API_KEY"] = args.api_key

    # Validate model is set
    if not os.environ.get("RFSN_MODEL_ID", "").strip():
        parser.error(
            "Model is required.  Pass --model <id> or set RFSN_MODEL_ID.\n"
            "  Example: rfsn-server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit"
        )

    # Early config validation (catches LAN guard, missing api_key, etc.)
    try:
        from ..config import RFSNConfig
        cfg = RFSNConfig.from_env()
    except Exception as exc:
        sys.exit(f"Configuration error: {exc}")

    try:
        import uvicorn
    except ImportError:
        sys.exit(
            "uvicorn is not installed.  Run: pip install 'mlx-rfsn[production]'"
        )

    host = cfg.server.host
    port = cfg.server.port

    print(f"Starting RFSN v10 server on http://{host}:{port}")
    print(f"  model:   {os.environ.get('RFSN_MODEL_ID')}")
    print(f"  backend: {os.environ.get('RFSN_BACKEND', 'mlx')}")
    print(f"  kv:      {os.environ.get('RFSN_ENABLE_KV_COMPRESSION', 'false')}")
    print(f"  api-key: {'required' if cfg.server.require_api_key else 'not required'}")
    print()

    uvicorn.run(
        "rfsn_v10.server.app:app",
        host=host,
        port=port,
        log_level="info",
    )


def health_check(argv: list[str] | None = None) -> None:
    """Check health of a running RFSN server.

    This is the ``rfsn-health`` entry point.
    """
    import urllib.error
    import urllib.request

    parser = argparse.ArgumentParser(
        prog="rfsn-health",
        description="Check health of a running RFSN v10 server",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Server base URL (default: http://127.0.0.1:8000)",
    )
    args = parser.parse_args(argv)

    health_url = args.url.rstrip("/") + "/health"
    try:
        with urllib.request.urlopen(health_url, timeout=5) as resp:
            data = json.loads(resp.read())
        print(json.dumps(data, indent=2))
        status = data.get("status", "unknown")
        sys.exit(0 if status == "ok" else 1)
    except urllib.error.URLError as exc:
        print(f"ERROR: Cannot reach server at {health_url}: {exc}", file=sys.stderr)
        sys.exit(1)


def config_check(argv: list[str] | None = None) -> None:
    """Validate RFSN config and environment.

    This is the ``rfsn-config-check`` entry point.
    """
    parser = argparse.ArgumentParser(
        prog="rfsn-config-check",
        description="Validate RFSN v10 configuration and environment",
    )
    parser.parse_args(argv)

    errors: list[str] = []
    warnings_: list[str] = []

    # Python version
    vi = sys.version_info
    if not ((3, 11) <= vi < (3, 13)):
        errors.append(f"Unsupported Python {vi.major}.{vi.minor}. Use 3.11 or 3.12.")
    else:
        print(f"  Python {vi.major}.{vi.minor}.{vi.micro} OK")

    # Model ID
    model_id = os.environ.get("RFSN_MODEL_ID", "").strip()
    if not model_id:
        warnings_.append("RFSN_MODEL_ID not set (required to start server)")
    else:
        print(f"  RFSN_MODEL_ID = {model_id}")

    # API key consistency
    if os.environ.get("RFSN_REQUIRE_API_KEY", "false").lower() == "true":
        if not os.environ.get("RFSN_API_KEY", "").strip():
            errors.append("RFSN_REQUIRE_API_KEY=true but RFSN_API_KEY is not set")
        else:
            print("  API key: configured")

    # Config import
    try:
        from rfsn_v10.config import RFSNConfig
        cfg = RFSNConfig.from_env()
        print(f"  Config load: OK (host={cfg.server.host}, port={cfg.server.port})")
    except Exception as exc:
        errors.append(f"Config load failed: {exc}")

    # Experimental flags
    try:
        from rfsn_v10.config import RFSNConfig
        cfg = RFSNConfig.from_env()
        if cfg.experimental.enable_qjl:
            warnings_.append("RFSN_EXPERIMENTAL_QJL=true (not validated)")
        if cfg.experimental.enable_polar:
            warnings_.append("RFSN_EXPERIMENTAL_POLAR=true (not validated)")
        if cfg.experimental.enable_adaptive:
            warnings_.append("RFSN_EXPERIMENTAL_ADAPTIVE=true (not validated)")
        if cfg.runtime.sparse_decode_enabled:
            warnings_.append("RFSN_SPARSE_DECODE_ENABLED=true (not benchmark-proven)")
    except Exception:
        pass

    print()
    for w in warnings_:
        print(f"  WARNING: {w}")
    for e in errors:
        print(f"  ERROR:   {e}", file=sys.stderr)

    if errors:
        print("\nConfig check FAILED.", file=sys.stderr)
        sys.exit(1)
    else:
        print("\nConfig check OK.")
