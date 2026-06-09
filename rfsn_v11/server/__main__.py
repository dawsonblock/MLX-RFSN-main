"""Entry point for ``python -m rfsn_v11.server``."""
import os
import uvicorn
from .app import app

host = os.environ.get("RFSN_SERVER_HOST", "0.0.0.0")
port = int(os.environ.get("RFSN_SERVER_PORT", "8000"))
uvicorn.run(app, host=host, port=port)
