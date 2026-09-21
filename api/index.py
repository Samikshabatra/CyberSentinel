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

from urllib.parse import parse_qsl, urlencode  # noqa: E402

from cybersentinel.api.main import app as _fastapi_app  # noqa: E402

# Every request reaches this function through the catch-all rewrite in
# vercel.json, and a rewrite replaces the request path with its destination:
# the application would see "/api/index" for "/health", "/docs" and everything
# else, and match none of them. The rewrite therefore carries the original path
# in a query parameter, which this wrapper puts back into the ASGI scope before
# the application sees the request.
_PATH_PARAM = "__vercel_path"


async def app(scope: dict, receive: object, send: object) -> None:
    """ASGI wrapper that restores the pre-rewrite request path."""
    if scope.get("type") in {"http", "websocket"}:
        params = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        original = next((value for key, value in params if key == _PATH_PARAM), None)
        if original:
            scope = dict(scope)
            scope["path"] = original
            scope["raw_path"] = original.encode("utf-8")
            # The marker is an artefact of the rewrite; a handler reading query
            # parameters should never see it.
            remaining = [(key, value) for key, value in params if key != _PATH_PARAM]
            scope["query_string"] = urlencode(remaining).encode("latin-1")

    await _fastapi_app(scope, receive, send)  # type: ignore[operator]


__all__ = ["app"]
