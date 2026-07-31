"""In-process publish/subscribe event bus.

Design constraints that shaped this implementation:

* **A slow or broken subscriber must never stall the control loop.** Handler
  exceptions are caught, counted and reported; they never propagate to the
  publisher. A handler that raises repeatedly is quarantined.
* **Publishing must be cheap and allocation-light**, because it happens at up to
  200 Hz from the control loop.
* **Delivery is synchronous by default.** Ordering guarantees make the system far
  easier to reason about, and every handler in this codebase is a fast, non-blocking
  state update. Anything expensive subscribes through
  :class:`QueuedSubscriber`, which hands work to its own thread.
* **Recent history is retained** so the diagnostics screen and the black-box
  recorder can show what happened just before a fault.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .clock import Clock, RealClock

__all__ = ["Event", "EventBus", "QueuedSubscriber", "Subscription"]

Handler = Callable[["Event"], None]

#: A handler that raises this many times in a row is unsubscribed automatically.
_QUARANTINE_THRESHOLD = 10


@dataclass(frozen=True, slots=True)
class Event:
    """A single published message."""

    topic: str
    payload: Any = None
    timestamp: float = 0.0
    source: str = ""
    #: Monotonically increasing per-bus sequence number, useful for replay.
    sequence: int = 0

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"[{self.sequence}] {self.topic} from={self.source or '?'}"


@dataclass(slots=True)
class Subscription:
    """Handle returned by :meth:`EventBus.subscribe`; call :meth:`cancel` to detach."""

    pattern: str
    handler: Handler
    bus: EventBus
    active: bool = True
    error_count: int = 0
    consecutive_errors: int = 0
    delivered: int = 0
    name: str = ""

    def cancel(self) -> None:
        """Stop receiving events. Idempotent."""
        if self.active:
            self.active = False
            self.bus._remove(self)


@dataclass(slots=True)
class _BusStats:
    published: int = 0
    delivered: int = 0
    handler_errors: int = 0
    quarantined: int = 0
    per_topic: dict[str, int] = field(default_factory=dict)


class EventBus:
    """Synchronous topic-based event bus with wildcard subscriptions.

    Example::

        bus = EventBus(clock)
        sub = bus.subscribe(Topics.HAND_STATE, lambda ev: print(ev.payload))
        bus.publish(Topics.HAND_STATE, state, source="control")
        sub.cancel()
    """

    def __init__(self, clock: Clock | None = None, *, history: int = 256) -> None:
        self._clock: Clock = clock or RealClock()
        self._exact: dict[str, list[Subscription]] = {}
        self._prefix: list[tuple[str, Subscription]] = []
        self._global: list[Subscription] = []
        self._lock = threading.RLock()
        self._sequence = 0
        self._history: deque[Event] = deque(maxlen=history)
        self._stats = _BusStats()
        #: Set by the application when the logger is available, so that handler
        #: failures are recorded rather than silently swallowed.
        self.error_reporter: Callable[[str, BaseException], None] | None = None

    # -- subscription ---------------------------------------------------------

    def subscribe(self, pattern: str, handler: Handler, *, name: str = "") -> Subscription:
        """Subscribe to ``pattern``.

        ``pattern`` is either an exact topic (``"emg.frame"``), a prefix wildcard
        (``"emg.*"``) or ``"*"`` for everything.
        """
        sub = Subscription(pattern=pattern, handler=handler, bus=self, name=name or pattern)
        with self._lock:
            if pattern == "*":
                self._global.append(sub)
            elif pattern.endswith(".*"):
                self._prefix.append((pattern[:-1], sub))  # keep the trailing dot
            elif pattern.endswith("*"):
                self._prefix.append((pattern[:-1], sub))
            else:
                self._exact.setdefault(pattern, []).append(sub)
        return sub

    def subscribe_many(self, patterns: Iterable[str], handler: Handler) -> list[Subscription]:
        """Subscribe one handler to several topics; returns all handles."""
        return [self.subscribe(p, handler) for p in patterns]

    def _remove(self, sub: Subscription) -> None:
        with self._lock:
            if sub in self._global:
                self._global.remove(sub)
                return
            for entry in list(self._prefix):
                if entry[1] is sub:
                    self._prefix.remove(entry)
                    return
            bucket = self._exact.get(sub.pattern)
            if bucket and sub in bucket:
                bucket.remove(sub)
                if not bucket:
                    del self._exact[sub.pattern]

    # -- publishing -----------------------------------------------------------

    def publish(self, topic: str, payload: Any = None, *, source: str = "") -> Event:
        """Publish ``payload`` on ``topic`` and deliver it synchronously."""
        with self._lock:
            self._sequence += 1
            event = Event(
                topic=topic,
                payload=payload,
                timestamp=self._clock.monotonic(),
                source=source,
                sequence=self._sequence,
            )
            self._history.append(event)
            self._stats.published += 1
            self._stats.per_topic[topic] = self._stats.per_topic.get(topic, 0) + 1
            targets = self._match(topic)

        for sub in targets:
            if not sub.active:
                continue
            try:
                sub.handler(event)
            except Exception as exc:
                self._on_handler_error(sub, exc)
            else:
                sub.delivered += 1
                sub.consecutive_errors = 0
                self._stats.delivered += 1
        return event

    def _match(self, topic: str) -> list[Subscription]:
        """Collect the subscriptions interested in ``topic`` (called under lock)."""
        matched = list(self._exact.get(topic, ()))
        for prefix, sub in self._prefix:
            if topic.startswith(prefix):
                matched.append(sub)
        matched.extend(self._global)
        return matched

    def _on_handler_error(self, sub: Subscription, exc: BaseException) -> None:
        sub.error_count += 1
        sub.consecutive_errors += 1
        self._stats.handler_errors += 1
        if self.error_reporter is not None:
            try:
                self.error_reporter(sub.name, exc)
            except Exception:  # pragma: no cover - reporter must never break the bus
                pass
        if sub.consecutive_errors >= _QUARANTINE_THRESHOLD:
            self._stats.quarantined += 1
            sub.cancel()

    # -- introspection --------------------------------------------------------

    def history(self, topic: str | None = None, limit: int = 50) -> list[Event]:
        """Most recent events, newest last. Used by the diagnostics console."""
        with self._lock:
            events = list(self._history)
        if topic is not None:
            events = [e for e in events if e.topic == topic]
        return events[-limit:]

    @property
    def stats(self) -> dict[str, Any]:
        """Counters for the diagnostics screen."""
        with self._lock:
            return {
                "published": self._stats.published,
                "delivered": self._stats.delivered,
                "handler_errors": self._stats.handler_errors,
                "quarantined": self._stats.quarantined,
                "subscriptions": (
                    sum(len(v) for v in self._exact.values())
                    + len(self._prefix)
                    + len(self._global)
                ),
                "top_topics": sorted(
                    self._stats.per_topic.items(), key=lambda kv: kv[1], reverse=True
                )[:10],
            }

    def clear(self) -> None:
        """Drop every subscription and the history buffer (used between tests)."""
        with self._lock:
            self._exact.clear()
            self._prefix.clear()
            self._global.clear()
            self._history.clear()


class QueuedSubscriber:
    """Adapter that moves event handling onto a dedicated worker thread.

    Used for anything that may block or take a variable amount of time — writing
    black-box records to disk, encoding a camera preview, rendering the UI — so
    that the publisher (often the 200 Hz control loop) is never delayed.

    Overflow policy is *drop oldest*: for telemetry-style consumers, the most
    recent state is what matters, and unbounded queues are how embedded systems
    run out of memory.
    """

    def __init__(
        self,
        bus: EventBus,
        patterns: Iterable[str],
        handler: Handler,
        *,
        maxsize: int = 512,
        name: str = "queued-subscriber",
    ) -> None:
        self._handler = handler
        self._queue: queue.Queue[Event | None] = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._running = False
        self._subs = [bus.subscribe(p, self._enqueue, name=name) for p in patterns]
        #: Number of events dropped because the worker fell behind.
        self.dropped = 0

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        for sub in self._subs:
            sub.cancel()
        if self._running:
            self._running = False
            self._queue.put(None)
            self._thread.join(timeout=timeout)

    def _enqueue(self, event: Event) -> None:
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1
            try:
                self._queue.get_nowait()  # drop the oldest
                self._queue.put_nowait(event)
            except (queue.Empty, queue.Full):  # pragma: no cover - race with worker
                pass

    def _run(self) -> None:
        while self._running:
            event = self._queue.get()
            if event is None:
                break
            try:
                self._handler(event)
            except Exception:
                continue
