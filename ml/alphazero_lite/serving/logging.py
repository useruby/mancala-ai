"""Structured stderr logging for the Mancala AI service."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import sys
from typing import Any


_HANDLER_MARKER = "mancala_ai_json_stderr"
_FIELDS = ("request_id", "profile", "outcome", "elapsed_ms")


class JsonLogFormatter(logging.Formatter):
    """Encode service log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, timezone.utc)
        payload: dict[str, Any] = {
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "severity": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for field in _FIELDS:
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_application_logging() -> None:
    """Enable INFO JSON logs for mancala.ai without altering other loggers."""
    logger = logging.getLogger("mancala.ai")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = next(
        (
            handler
            for handler in logger.handlers
            if getattr(handler, "_mancala_ai_handler", False)
        ),
        None,
    )
    if handler is not None:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler(sys.stderr)
    handler._mancala_ai_handler = True  # type: ignore[attr-defined]
    handler.setLevel(logging.INFO)
    handler.setFormatter(JsonLogFormatter())
    logger.addHandler(handler)
