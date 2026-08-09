"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from cybersentinel import __version__
from cybersentinel.api.routes import analysis, incidents
from cybersentinel.database.connection import init_database
from cybersentinel.utils.config import get_settings
from cybersentinel.utils.logging import configure_logging, get_logger
from cybersentinel.utils.validation import InputValidationError

logger = get_logger(__name__)

DESCRIPTION = """
AI-assisted cybersecurity incident analysis.

CyberSentinel classifies a security event with a cybersecurity-specialised LLM,
grounds the findings in retrieved threat intelligence (MITRE ATT&CK, CWE, NVD),
correlates related events, scores risk with a transparent matrix, and produces a
structured, explainable incident report.

**Safety posture**

* Every output is a recommendation for a human analyst. Nothing is executed.
* High-risk or operationally impactful recommendations pause the workflow for
  explicit analyst approval.
* Threat-intelligence identifiers are only reported when supported by retrieved
  sources; unsupported claims are rejected and listed.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Prepare shared resources on startup."""
    configure_logging()
    settings = get_settings()
    logger.info(
        f"starting CyberSentinel API (env={settings.app_env}, llm={settings.llm_backend}, "
        f"embeddings={settings.embedding_backend})"
    )
    try:
        init_database()
    except Exception as exc:
        logger.warning(f"database initialisation failed: {type(exc).__name__}: {exc}")
    yield
    logger.info("shutting down CyberSentinel API")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="CyberSentinel API",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # The Streamlit UI runs on a different port; in development it needs CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [settings.api_base_url],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(incidents.router)
    app.include_router(analysis.router)

    @app.exception_handler(InputValidationError)
    async def _validation_handler(_request: Request, exc: InputValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc), "error_type": "InputValidationError"},
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # `exc.errors()` can carry a raw exception object in `ctx`, which is not
        # JSON serialisable; report only the fields a client can act on.
        errors = [
            {
                "location": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg", ""),
                "type": error.get("type", ""),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": "invalid request payload", "errors": errors},
        )

    @app.exception_handler(Exception)
    async def _unhandled_handler(_request: Request, exc: Exception) -> JSONResponse:
        # Never leak internals to the client; the full trace goes to the logs.
        logger.exception("unhandled error")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal error", "error_type": type(exc).__name__},
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "name": "CyberSentinel",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()


def run() -> None:
    """Console entry point: ``cybersentinel-api``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "cybersentinel.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
