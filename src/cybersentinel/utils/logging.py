"""Structured logging for CyberSentinel.

Every LangGraph node emits a record carrying run id, incident id, node name,
status and latency so a full workflow can be reconstructed from logs.
Raw analyst input is never logged in full - only a truncated, redacted preview.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from cybersentinel.utils.config import get_settings

_CONFIGURED = False

# Patterns redacted before anything reaches the logs.
_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|token)\b\s*[:=]\s*\S+"), r"\1=***"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"), "bearer ***"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<email>"),
]


def redact(text: str, max_chars: int = 200) -> str:
    """Redact obvious secrets/PII and truncate for safe logging."""
    cleaned = text
    for pattern, replacement in _REDACTIONS:
        cleaned = pattern.sub(replacement, cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars] + "...[truncated]"
    return cleaned


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    _RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
        "message",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Readable console format that still surfaces node/run context."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<8} {record.name} | {record.getMessage()}"
        extras = []
        for key in ("run_id", "incident_id", "node", "status", "latency_s"):
            value = getattr(record, key, None)
            if value is not None:
                extras.append(f"{key}={value}")
        if extras:
            base = f"{base} [{' '.join(extras)}]"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


def configure_logging(level: str | None = None, json_output: bool | None = None) -> None:
    """Configure root logging once for the whole process."""
    global _CONFIGURED
    settings = get_settings()
    resolved_level = (level or settings.log_level).upper()
    use_json = settings.log_json if json_output is None else json_output

    root = logging.getLogger()
    root.setLevel(resolved_level)

    if _CONFIGURED:
        for handler in root.handlers:
            handler.setFormatter(JsonFormatter() if use_json else HumanFormatter())
        return

    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if use_json else HumanFormatter())
    root.addHandler(handler)

    # Third-party noise control.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    configure_logging()
    return logging.getLogger(name)


def new_run_id() -> str:
    """Generate a short correlation id for one workflow execution."""
    return uuid.uuid4().hex[:12]


@contextmanager
def log_node(
    logger: logging.Logger,
    node: str,
    run_id: str,
    incident_id: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Time a LangGraph node and log its start/finish with status and latency.

    Yields a mutable dict; nodes may add small, non-sensitive fields to it which
    are included in the completion log record.
    """
    started = time.perf_counter()
    context: dict[str, Any] = {}
    logger.debug(
        "node started", extra={"run_id": run_id, "incident_id": incident_id, "node": node}
    )
    try:
        yield context
    except Exception as exc:
        latency = round(time.perf_counter() - started, 3)
        logger.exception(
            f"node failed: {type(exc).__name__}",
            extra={
                "run_id": run_id,
                "incident_id": incident_id,
                "node": node,
                "status": "error",
                "latency_s": latency,
                **context,
            },
        )
        raise
    else:
        latency = round(time.perf_counter() - started, 3)
        logger.info(
            "node completed",
            extra={
                "run_id": run_id,
                "incident_id": incident_id,
                "node": node,
                "status": "success",
                "latency_s": latency,
                **context,
            },
        )
