"""Entry point for ``python -m rfsn_v10.server``.

Runs the FastAPI inference server via uvicorn.
Environment variables:
    RFSN_SERVER_HOST — bind host (default: 0.0.0.0)
    RFSN_SERVER_PORT — bind port (default: 8000)
"""
from __future__ import annotations

import os

from .app import app

host = os.environ.get("RFSN_SERVER_HOST", "0.0.0.0")
port = int(os.environ.get("RFSN_SERVER_PORT", "8000"))

import uvicorn  # noqa: E402

uvicorn.run(app, host=host, port=port)
