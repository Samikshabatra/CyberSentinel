"""Database engine and session management.

SQLite is the default so the application runs with no services at all;
PostgreSQL is used by changing ``DATABASE_URL``. Both are supported by the same
SQLAlchemy models.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from cybersentinel.database.models import Base
from cybersentinel.utils.config import Settings, get_settings
from cybersentinel.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_sqlite_path(url: str, settings: Settings) -> str:
    """Make a relative SQLite path absolute against the project root."""
    prefix = "sqlite:///"
    if not url.startswith(prefix) or url.startswith("sqlite:///:memory:"):
        return url

    raw = url[len(prefix) :]
    path = Path(raw)
    if path.is_absolute():
        return url

    resolved = (settings.project_root / raw.lstrip("./")).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{resolved.as_posix()}"


def build_engine(settings: Settings | None = None, url: str | None = None) -> Engine:
    """Create an engine for the configured database."""
    resolved_settings = settings or get_settings()
    database_url = _resolve_sqlite_path(url or resolved_settings.database_url, resolved_settings)

    kwargs: dict[str, Any] = {"pool_pre_ping": True, "future": True}
    if database_url.startswith("sqlite"):
        # check_same_thread=False is required because FastAPI serves requests
        # from a thread pool while sharing one engine.
        kwargs["connect_args"] = {"check_same_thread": False}

    engine = create_engine(database_url, **kwargs)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    logger.debug(f"database engine created for {database_url.split('@')[-1]}")
    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine."""
    return build_engine()


@lru_cache(maxsize=1)
def get_session_factory() -> sessionmaker[Session]:
    """Return the process-wide session factory.

    The schema is created here as well, so any caller - including a health check
    that runs before the first analysis - finds usable tables.
    """
    engine = get_engine()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_database(engine: Engine | None = None) -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(engine or get_engine())
    logger.info("database schema ready")


@contextmanager
def session_scope(factory: sessionmaker[Session] | None = None) -> Iterator[Session]:
    """Transactional session scope: commit on success, roll back on failure."""
    session_factory = factory or get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    with session_scope() as session:
        yield session


def reset_connections() -> None:
    """Drop cached engine and session factory (used by tests)."""
    get_session_factory.cache_clear()
    get_engine.cache_clear()
