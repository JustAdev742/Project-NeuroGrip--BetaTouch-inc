"""Service lifecycle and health reporting.

Every long-lived subsystem (EMG, vision, control, safety, UI, diagnostics)
implements :class:`Service`. Uniform lifecycle gives the application a single,
ordered startup and — more importantly — a single, *reverse*-ordered shutdown, so
the servo bus is always disabled before the transport it depends on is closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "IDLE_TICK",
    "HealthReport",
    "HealthStatus",
    "Service",
    "ServiceBase",
    "TickResult",
]


class HealthStatus(IntEnum):
    """Coarse health of a component, ordered so ``max()`` gives the worst."""

    OK = 0
    DEGRADED = 1
    FAILED = 2
    OFFLINE = 3

    @property
    def label(self) -> str:
        return self.name.title()

    @property
    def is_usable(self) -> bool:
        """Whether callers may still rely on the component's output."""
        return self in (HealthStatus.OK, HealthStatus.DEGRADED)


@dataclass(frozen=True, slots=True)
class HealthReport:
    """A component's self-assessment, surfaced on the diagnostics screen."""

    name: str
    status: HealthStatus = HealthStatus.OK
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, name: str, **metrics: Any) -> HealthReport:
        return cls(name=name, status=HealthStatus.OK, metrics=metrics)

    @classmethod
    def degraded(cls, name: str, detail: str, **metrics: Any) -> HealthReport:
        return cls(name=name, status=HealthStatus.DEGRADED, detail=detail, metrics=metrics)

    @classmethod
    def failed(cls, name: str, detail: str, **metrics: Any) -> HealthReport:
        return cls(name=name, status=HealthStatus.FAILED, detail=detail, metrics=metrics)

    @classmethod
    def offline(cls, name: str, detail: str = "not started") -> HealthReport:
        return cls(name=name, status=HealthStatus.OFFLINE, detail=detail)

    def __str__(self) -> str:  # pragma: no cover - display helper
        suffix = f": {self.detail}" if self.detail else ""
        return f"{self.name} [{self.status.label}]{suffix}"


@dataclass(frozen=True, slots=True)
class TickResult:
    """What a service did during one scheduler tick.

    Returning a value (rather than ``None``) lets the scheduler account for work
    performed, which is what drives the "is this loop actually running?" watchdogs.
    """

    did_work: bool = True
    detail: str = ""


#: Shared instance for the common "nothing to do this tick" case.
IDLE_TICK = TickResult(did_work=False)


@runtime_checkable
class Service(Protocol):
    """A startable, stoppable, self-reporting subsystem."""

    @property
    def name(self) -> str:
        """Stable identifier used in logs, health reports and configuration."""
        ...

    def start(self) -> None:
        """Acquire resources. Must be idempotent and must not block indefinitely."""
        ...

    def stop(self) -> None:
        """Release resources. Must be idempotent and must not raise."""
        ...

    def health(self) -> HealthReport:
        """Report current health. Must be cheap — it is polled at a few hertz."""
        ...


class ServiceBase:
    """Convenience base implementing the boring half of :class:`Service`.

    Subclasses override :meth:`on_start`, :meth:`on_stop` and optionally
    :meth:`health`. Start/stop idempotency and the ``running`` flag are handled
    here so that no subsystem has to reimplement them (and get them subtly wrong).
    """

    #: Overridden by subclasses; falls back to the class name.
    service_name: str = ""

    def __init__(self) -> None:
        self._running = False
        self._start_error: str = ""

    @property
    def name(self) -> str:
        return self.service_name or type(self).__name__

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        self.on_start()
        self._running = True
        self._start_error = ""

    def stop(self) -> None:
        if not self._running:
            return
        try:
            self.on_stop()
        finally:
            self._running = False

    def on_start(self) -> None:
        """Hook for subclasses; default does nothing."""

    def on_stop(self) -> None:
        """Hook for subclasses; default does nothing."""

    def health(self) -> HealthReport:
        if not self._running:
            return HealthReport.offline(self.name, self._start_error or "not started")
        return HealthReport.ok(self.name)

    def __enter__(self) -> ServiceBase:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
