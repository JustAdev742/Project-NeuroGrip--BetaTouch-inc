"""Emergency stop.

The e-stop is a **latch**, not a flag. Once engaged it stays engaged until a
human explicitly clears it. That is the whole point: an automatic reset would
mean a transient fault could re-enable a limb without anyone deciding it was safe
to do so.

Properties this implementation guarantees:

* **Thread-safe and reentrant.** It may be called from the control loop, the UI
  thread, a signal handler or a watchdog callback.
* **Never raises.** An e-stop that can fail is not an e-stop. Listener exceptions
  are caught and logged.
* **Idempotent.** Engaging an already-engaged stop records the additional reason
  and does nothing else.
* **Auditable.** Every engagement and release is recorded with its source,
  reason and timestamp for the incident log.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from ..core.clock import Clock
from ..core.logging import get_logger

__all__ = ["EmergencyStop", "EstopRecord", "EstopSource"]

log = get_logger(__name__)


class EstopSource:
    """Well-known e-stop sources, for consistent audit records."""

    USER_UI = "user:ui"
    USER_EMG = "user:emg"
    HARDWARE_BUTTON = "hardware:button"
    WATCHDOG = "safety:watchdog"
    SAFETY_RULE = "safety:rule"
    OVERCURRENT = "safety:overcurrent"
    THERMAL = "safety:thermal"
    COMMS = "safety:communication"
    SELFTEST = "diagnostics:selftest"
    SHUTDOWN = "system:shutdown"


@dataclass(frozen=True, slots=True)
class EstopRecord:
    """Audit entry for an e-stop engagement, release or rehearsal."""

    engaged: bool
    source: str
    reason: str
    timestamp: float
    #: True for a :meth:`EmergencyStop.rehearse` notification. Listeners must
    #: acknowledge these and do nothing else — a rehearsal proves the
    #: notification path is intact without stopping anything.
    rehearsal: bool = False

    def __str__(self) -> str:  # pragma: no cover - display helper
        if self.rehearsal:
            return f"E-STOP rehearsal by {self.source}"
        verb = "ENGAGED" if self.engaged else "released"
        return f"E-STOP {verb} by {self.source}: {self.reason}"


class EmergencyStop:
    """Latching emergency stop with listener notification."""

    def __init__(self, clock: Clock, *, history: int = 32) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._engaged = False
        self._reasons: list[str] = []
        self._source = ""
        self._engaged_at = 0.0
        self._history: list[EstopRecord] = []
        self._history_limit = history
        self._listeners: list[Callable[[EstopRecord], None]] = []
        #: Counters for the diagnostics screen.
        self.engage_count = 0
        self.release_count = 0
        self.rehearsal_count = 0

    # -- listeners ------------------------------------------------------------

    def add_listener(self, listener: Callable[[EstopRecord], None]) -> None:
        """Register a callback invoked on every engage/release.

        Listeners must be fast and must not raise; exceptions are caught so one
        broken listener cannot prevent the stop from taking effect.
        """
        with self._lock:
            self._listeners.append(listener)

    @property
    def listener_count(self) -> int:
        """How many listeners are registered.

        Zero is a defect, not a configuration: an e-stop nothing listens to is a
        latch that changes a boolean. :mod:`neurogrip.safety.integrity` checks it.
        """
        with self._lock:
            return len(self._listeners)

    def rehearse(self, source: str = "diagnostics:integrity") -> EstopRecord:
        """Notify listeners *without* engaging, so the chain can be verified.

        The e-stop's notification path is the part most likely to break
        silently: it is a registration made once at assembly, and losing it
        changes no behaviour until the day someone needs the stop. A rehearsal
        exercises exactly that path — every listener is called with a record
        marked ``rehearsal`` — and costs the hand nothing.

        This deliberately proves less than it might appear to. It shows the
        record reaches the controller; it does not show that the actuators would
        actually stop. That second half needs the drive, and is what
        :class:`~neurogrip.safety.integrity.EstopSelfCheck` runs its periodic
        proof test for.
        """
        record = EstopRecord(
            engaged=False,
            source=source,
            reason="integrity rehearsal",
            timestamp=self._clock.monotonic(),
            rehearsal=True,
        )
        # Not appended to the history: rehearsals are routine and would bury the
        # real engagements the incident log exists to preserve.
        self.rehearsal_count += 1
        self._notify(record)
        return record

    def _notify(self, record: EstopRecord) -> None:
        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception as exc:
                log.error("e-stop listener raised", error=str(exc))

    # -- state ----------------------------------------------------------------

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def reasons(self) -> tuple[str, ...]:
        """Every reason accumulated during the current engagement."""
        with self._lock:
            return tuple(self._reasons)

    @property
    def source(self) -> str:
        return self._source

    @property
    def duration(self) -> float:
        """Seconds the stop has been engaged; ``0`` when clear."""
        if not self._engaged:
            return 0.0
        return self._clock.monotonic() - self._engaged_at

    @property
    def summary(self) -> str:
        """One line for the UI banner."""
        if not self._engaged:
            return ""
        return f"{self._reasons[0]} ({self._source})" if self._reasons else self._source

    # -- control --------------------------------------------------------------

    def engage(self, source: str, reason: str) -> EstopRecord:
        """Latch the stop. Idempotent; always safe to call."""
        with self._lock:
            now = self._clock.monotonic()
            first = not self._engaged
            if first:
                self._engaged = True
                self._engaged_at = now
                self._source = source
                self._reasons = [reason]
                self.engage_count += 1
            elif reason not in self._reasons:
                # Additional causes during one engagement are all recorded: after
                # an incident, the *set* of things that went wrong matters.
                self._reasons.append(reason)
            record = EstopRecord(engaged=True, source=source, reason=reason, timestamp=now)
            self._append(record)

        if first:
            log.critical("EMERGENCY STOP ENGAGED", source=source, reason=reason)
        else:
            log.warning("additional e-stop cause", source=source, reason=reason)
        self._notify(record)
        return record

    def release(self, source: str, reason: str = "acknowledged by user") -> EstopRecord | None:
        """Clear the latch. Returns ``None`` if it was not engaged.

        Deliberately requires a ``source``: releasing an e-stop is an action with
        an owner, and the audit log records who.
        """
        with self._lock:
            if not self._engaged:
                return None
            now = self._clock.monotonic()
            duration = now - self._engaged_at
            self._engaged = False
            self._reasons = []
            self._source = ""
            self.release_count += 1
            record = EstopRecord(engaged=False, source=source, reason=reason, timestamp=now)
            self._append(record)

        log.warning(
            "emergency stop released", source=source, reason=reason, duration_s=round(duration, 2)
        )
        self._notify(record)
        return record

    def _append(self, record: EstopRecord) -> None:
        self._history.append(record)
        if len(self._history) > self._history_limit:
            del self._history[0 : len(self._history) - self._history_limit]

    # -- introspection --------------------------------------------------------

    def history(self, limit: int = 10) -> list[EstopRecord]:
        with self._lock:
            return self._history[-limit:]

    def require_clear(self) -> None:
        """Raise if the stop is engaged — used to guard actuator enable paths."""
        if self._engaged:
            from ..core.errors import EmergencyStopActive

            raise EmergencyStopActive(
                "emergency stop is engaged",
                context={"source": self._source, "reasons": list(self._reasons)},
            )
