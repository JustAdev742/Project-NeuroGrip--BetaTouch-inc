"""Structured logging.

Built on the standard library, with three device-specific additions:

* **Structured context.** ``log.info("grasp planned", grasp="cylindrical", conf=0.82)``
  keeps machine-readable fields out of the message string so the black-box
  recorder and the log viewer can filter on them.
* **A ring-buffer sink** that the touchscreen Logs screen reads directly — no file
  tailing, no extra process.
* **Rate limiting.** A sensor that fails at 200 Hz must not fill the eMMC with
  200 identical lines per second.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

__all__ = [
    "JsonFormatter",
    "LogRecordView",
    "RingBufferHandler",
    "StructuredLogger",
    "configure_logging",
    "get_logger",
    "log_buffer",
]

_LEVELS = {
    "TRACE": 5,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

logging.addLevelName(5, "TRACE")


@dataclass(frozen=True, slots=True)
class LogRecordView:
    """A log line as consumed by the UI and diagnostics screens."""

    timestamp: float
    level: str
    logger: str
    message: str
    fields: dict[str, Any]

    @property
    def level_no(self) -> int:
        return _LEVELS.get(self.level, logging.INFO)

    def format(self) -> str:
        base = f"{self.level:<8} {self.logger}: {self.message}"
        if self.fields:
            base += " " + " ".join(f"{k}={v}" for k, v in self.fields.items())
        return base


class RingBufferHandler(logging.Handler):
    """Keeps the most recent records in memory for the on-device log viewer."""

    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self._records: deque[LogRecordView] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        view = LogRecordView(
            timestamp=record.created,
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            fields=dict(getattr(record, "fields", {}) or {}),
        )
        with self._lock:
            self._records.append(view)

    def records(self, *, level: int = 0, limit: int = 200, contains: str = "") -> list[LogRecordView]:
        """Filtered view of the buffer, newest last."""
        with self._lock:
            items = list(self._records)
        if level:
            items = [r for r in items if r.level_no >= level]
        if contains:
            needle = contains.lower()
            items = [r for r in items if needle in r.message.lower() or needle in r.logger.lower()]
        return items[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class JsonFormatter(logging.Formatter):
    """One JSON object per line — the on-disk format for post-hoc analysis."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 6),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload["fields"] = _jsonable(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), default=str)


class _ConsoleFormatter(logging.Formatter):
    """Compact, aligned, optionally colourised console output."""

    COLOURS: ClassVar[dict[str, str]] = {
        "TRACE": "\033[2;37m",
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;41m",
    }
    RESET: ClassVar[str] = "\033[0m"

    def __init__(self, colour: bool) -> None:
        super().__init__(datefmt="%H:%M:%S")
        self._colour = colour

    def format(self, record: logging.LogRecord) -> str:
        stamp = self.formatTime(record, self.datefmt)
        level = record.levelname
        prefix = f"{stamp} {level:<8}"
        if self._colour:
            prefix = f"{stamp} {self.COLOURS.get(level, '')}{level:<8}{self.RESET}"
        line = f"{prefix} {record.name:<28} {record.getMessage()}"
        fields = getattr(record, "fields", None)
        if fields:
            line += "  " + " ".join(f"{k}={v}" for k, v in fields.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary log fields to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class StructuredLogger:
    """Thin wrapper adding keyword fields and rate limiting to a stdlib logger."""

    __slots__ = ("_lock", "_logger", "_suppressed")

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger
        self._suppressed: dict[str, tuple[float, int]] = {}
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._logger.name

    def _log(self, level: int, message: str, exc_info: Any = None, **fields: Any) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, message, extra={"fields": fields}, exc_info=exc_info)

    def trace(self, message: str, **fields: Any) -> None:
        self._log(5, message, **fields)

    def debug(self, message: str, **fields: Any) -> None:
        self._log(logging.DEBUG, message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log(logging.INFO, message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log(logging.WARNING, message, **fields)

    def error(self, message: str, exc_info: Any = None, **fields: Any) -> None:
        self._log(logging.ERROR, message, exc_info=exc_info, **fields)

    def critical(self, message: str, exc_info: Any = None, **fields: Any) -> None:
        self._log(logging.CRITICAL, message, exc_info=exc_info, **fields)

    def exception(self, message: str, **fields: Any) -> None:
        self._log(logging.ERROR, message, exc_info=True, **fields)

    def throttled(
        self,
        key: str,
        level: str,
        message: str,
        *,
        now: float,
        interval: float = 5.0,
        **fields: Any,
    ) -> None:
        """Log at most once per ``interval`` seconds for a given ``key``.

        The suppressed count is reported when the next line is finally emitted,
        so no information is lost — only volume.

        ``now`` is passed in explicitly (from the injected clock) so that
        throttling is deterministic under simulation.
        """
        with self._lock:
            last, skipped = self._suppressed.get(key, (-1e18, 0))
            if now - last < interval:
                self._suppressed[key] = (last, skipped + 1)
                return
            self._suppressed[key] = (now, 0)
        if skipped:
            fields["suppressed"] = skipped
        self._log(_LEVELS.get(level.upper(), logging.INFO), message, **fields)

    def child(self, suffix: str) -> StructuredLogger:
        """Derive a sub-logger (``emg`` -> ``emg.pipeline``)."""
        return StructuredLogger(self._logger.getChild(suffix))


#: Process-wide ring buffer; the UI Logs screen reads from this instance.
log_buffer = RingBufferHandler()

_configured = False


def configure_logging(
    *,
    level: str = "INFO",
    console: bool = True,
    colour: bool | None = None,
    file_path: str | Path | None = None,
    max_bytes: int = 4_000_000,
    backups: int = 3,
    buffer_capacity: int = 1000,
    quiet_loggers: Iterable[str] = (),
) -> RingBufferHandler:
    """Install handlers on the root logger. Safe to call more than once.

    Returns the ring-buffer handler so callers can hand it to the UI.
    """
    global _configured

    root = logging.getLogger()
    root.setLevel(_LEVELS.get(level.upper(), logging.INFO))

    if _configured:
        for handler in list(root.handlers):
            root.removeHandler(handler)

    log_buffer.__init__(buffer_capacity)  # type: ignore[misc]  # re-size in place
    root.addHandler(log_buffer)

    if console:
        use_colour = sys.stderr.isatty() if colour is None else colour
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(_ConsoleFormatter(use_colour))
        root.addHandler(stream)

    if file_path:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
        )
        rotating.setFormatter(JsonFormatter())
        root.addHandler(rotating)

    for name in quiet_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    return log_buffer


def get_logger(name: str) -> StructuredLogger:
    """Return the structured logger for ``name`` (conventionally ``__name__``)."""
    return StructuredLogger(logging.getLogger(name))
