"""Black-box recorder and session telemetry.

Two jobs:

* **Black box.** A rolling in-memory buffer of recent state and events that is
  flushed to disk when something goes wrong (a critical fault, an e-stop, a
  crash). After an incident — "it squeezed too hard", "it moved when I didn't
  mean it to" — this is what makes the question answerable. Without it, every
  such report is unfalsifiable, and that is not an acceptable position for a
  device someone wears.
* **Session telemetry.** Optional continuous logging for development and for
  clinical review, written as newline-delimited JSON.

Both are strictly append-and-forget from the caller's perspective. Writing runs
on a :class:`~neurogrip.core.events.QueuedSubscriber` worker thread so disk
latency never reaches the control loop.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .core.clock import Clock
from .core.events import Event, EventBus, QueuedSubscriber
from .core.logging import get_logger
from .core.topics import Topics

__all__ = ["BlackBoxRecorder", "TelemetryWriter", "serialise"]

log = get_logger(__name__)

#: Topics always captured by the black box. Deliberately narrow: this buffer must
#: cover a useful *time* window, and high-rate topics would shrink that to
#: seconds.
BLACKBOX_TOPICS = (
    Topics.DECISION_MADE,
    Topics.GRASP_PLANNED,
    Topics.INTENT_UPDATED,
    Topics.MODE_CHANGED,
    Topics.MOTION_STARTED,
    Topics.MOTION_COMPLETED,
    Topics.MOTION_CANCELLED,
    Topics.GRIP_CONTACT,
    Topics.GRIP_SLIP,
    Topics.SAFETY_FAULT_RAISED,
    Topics.SAFETY_FAULT_CLEARED,
    Topics.ESTOP_ENGAGED,
    Topics.ESTOP_RELEASED,
    Topics.WATCHDOG_EXPIRED,
    Topics.SYSTEM_ERROR,
)

#: Flushing the black box is triggered by any of these.
TRIGGER_TOPICS = (
    Topics.ESTOP_ENGAGED,
    Topics.SAFETY_FAULT_RAISED,
    Topics.WATCHDOG_EXPIRED,
    Topics.SYSTEM_ERROR,
)


def _is_diagnostic(event: Event) -> bool:
    """True for an event a self-check produced deliberately.

    Such events are still *recorded* — they are part of what happened — but they
    do not trigger an incident flush. A periodic e-stop proof test writing an
    incident file every few hours would bury the real ones, which is the only
    thing the black box exists to preserve.
    """
    payload = event.payload
    if isinstance(payload, dict):
        return bool(payload.get("diagnostic"))
    return bool(getattr(payload, "diagnostic", False))


def serialise(value: Any, depth: int = 0) -> Any:
    """Convert arbitrary runtime objects to JSON-safe values.

    Depth-limited, because state objects reference each other and an unbounded
    walk would either recurse forever or serialise the whole system.
    """
    if depth > 4:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return {k: serialise(v, depth + 1) for k, v in asdict(value).items()}
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, dict):
        return {str(k): serialise(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialise(v, depth + 1) for v in value]
    if hasattr(value, "value") and hasattr(value, "name"):  # Enum
        return value.value
    if hasattr(value, "as_dict"):
        try:
            return serialise(value.as_dict(), depth + 1)
        except Exception:
            return str(value)
    return str(value)


class BlackBoxRecorder:
    """Rolling incident recorder."""

    def __init__(
        self,
        bus: EventBus,
        clock: Clock,
        *,
        directory: Path | str = "var/blackbox",
        capacity: int = 2000,
        auto_flush: bool = True,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._directory = Path(directory)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._auto_flush = auto_flush
        self._subscriber: QueuedSubscriber | None = None
        self._flushes = 0
        #: Suppress repeated flushes for one continuing incident.
        self._last_flush_at = -1e18
        self._min_flush_interval = 5.0

    def start(self) -> None:
        """Subscribe and begin recording."""
        self._subscriber = QueuedSubscriber(
            self._bus, BLACKBOX_TOPICS, self._on_event, maxsize=1024, name="blackbox"
        )
        self._subscriber.start()
        log.info("black-box recorder started", capacity=self._buffer.maxlen)

    def stop(self) -> None:
        if self._subscriber is not None:
            self._subscriber.stop()
            self._subscriber = None

    def _on_event(self, event: Event) -> None:
        self._buffer.append(
            {
                "t": round(event.timestamp, 4),
                "seq": event.sequence,
                "topic": event.topic,
                "source": event.source,
                "payload": serialise(event.payload),
            }
        )
        if self._auto_flush and event.topic in TRIGGER_TOPICS and not _is_diagnostic(event):
            self.flush(reason=event.topic)

    def flush(self, reason: str = "manual") -> Path | None:
        """Write the buffer to a timestamped file. Returns the path."""
        now = self._clock.monotonic()
        if now - self._last_flush_at < self._min_flush_interval:
            return None
        self._last_flush_at = now

        if not self._buffer:
            return None

        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock.wall()))
            path = self._directory / f"incident-{stamp}.json"
            payload = {
                "reason": reason,
                "recorded_at": self._clock.wall(),
                "events": list(self._buffer),
            }
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        except OSError as exc:
            log.error("could not write black-box record", error=str(exc))
            return None

        self._flushes += 1
        log.warning("black-box record written", path=str(path), reason=reason, events=len(self._buffer))
        return path

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    @property
    def flushes(self) -> int:
        return self._flushes

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Most recent records, for the on-screen incident viewer."""
        return list(self._buffer)[-limit:]


class TelemetryWriter:
    """Continuous newline-delimited JSON telemetry.

    Off by default. Useful for development, for clinical review, and for building
    datasets. Rotates by size so a long session cannot fill the storage.
    """

    def __init__(
        self,
        bus: EventBus,
        clock: Clock,
        *,
        path: Path | str = "var/telemetry/session.jsonl",
        topics: tuple[str, ...] = ("*",),
        max_bytes: int = 32_000_000,
    ) -> None:
        self._bus = bus
        self._clock = clock
        self._path = Path(path)
        self._topics = topics
        self._max_bytes = max_bytes
        self._handle = None
        self._written = 0
        self._subscriber: QueuedSubscriber | None = None
        self.dropped = 0

    def start(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._subscriber = QueuedSubscriber(
            self._bus, self._topics, self._write, maxsize=4096, name="telemetry"
        )
        self._subscriber.start()
        log.info("telemetry writer started", path=str(self._path))

    def stop(self) -> None:
        if self._subscriber is not None:
            self._subscriber.stop()
            self._subscriber = None
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def _write(self, event: Event) -> None:
        if self._handle is None:
            return
        record = {
            "t": round(event.timestamp, 4),
            "topic": event.topic,
            "source": event.source,
            "payload": serialise(event.payload),
        }
        line = json.dumps(record, separators=(",", ":"), default=str)
        try:
            self._handle.write(line + "\n")
            self._written += len(line) + 1
            if self._written >= self._max_bytes:
                self._rotate()
        except OSError as exc:
            self.dropped += 1
            log.throttled(
                "telemetry-write", "error", "telemetry write failed",
                now=self._clock.monotonic(), error=str(exc),
            )

    def _rotate(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock.wall()))
        rotated = self._path.with_name(f"{self._path.stem}-{stamp}{self._path.suffix}")
        try:
            self._path.rename(rotated)
        except OSError:  # pragma: no cover - best effort
            pass
        self._handle = self._path.open("a", encoding="utf-8")
        self._written = 0
