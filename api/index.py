"""Vercel serverless entry point for the CyberSentinel API.

Vercel's Python runtime imports this module and serves the ASGI callable named
``app``. Two things have to be settled before the application package is
imported, because ``Settings`` is cached on first use:

1. ``src`` is added to the import path - the project uses a src layout and the
   serverless bundle installs dependencies only, not the project itself.
2. Any path the application writes to is redirected under ``/tmp``, the only
   writable directory in a Vercel lambda.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# SQLite is the zero-service default, but the bundle is read-only. /tmp is
# per-instance and cleared on cold start, so incident history survives only for
# the life of one lambda; set DATABASE_URL to a managed Postgres for real
# persistence and this default is skipped.
os.environ.setdefault("DATABASE_URL", "sqlite:////tmp/cybersentinel.db")

# A lambda has no Qdrant to reach. The vector store falls back to the JSON
# index committed at data/processed/, which is read-only and read-only is all
# the query path needs.
os.environ.setdefault("QDRANT_IN_MEMORY", "false")

# No GPU, no model weights, no write access: the dependency-free backends are
# the only ones that can run here.
os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("EMBEDDING_BACKEND", "hash")
os.environ.setdefault("APP_ENV", "production")

from cybersentinel.api.main import app  # noqa: E402

__all__ = ["app"]
