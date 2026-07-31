"""Watchdog timers.

A prosthetic hand has several independent things that must keep happening. If any
of them stops, the hand must notice within milliseconds rather than waiting to be
told. Each gets a named watchdog that its owner must "kick":

======================  ==========  ============================================
watchdog                timeout     what its expiry means
======================  ==========  ============================================
``control``             100 ms      the control loop has stalled
``emg``                 300 ms      no EMG data; user intent is unknown
``servo``               250 ms      no motor telemetry; hand state is unknown
``vision``              2 s         perception is stale (assistance only)
``ui``                  2 s         the interface has hung (non-critical)
======================  ==========  ============================================

Severity is per-watchdog, because the responses differ: losing vision means
degrade to manual, losing the control loop means stop.

Note the layering. These are the *host-side* watchdogs. The ESP32 firmware runs
its own, independent timeout on the command stream (see ``docs/protocol.md``), so
a host that crashes outright — and therefore cannot run any of these — still
results in a hand that safes itself.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.clock import Clock
from ..core.errors import Severity
from ..core.logging import get_logger

__all__ = ["Watchdog", "WatchdogExpiry", "WatchdogGroup"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WatchdogExpiry:
    """Record of a watchdog firing."""

    name: str
    timeout: float
    elapsed: float
    severity: Severity
    timestamp: float
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"watchdog '{self.name}' expired: {self.elapsed * 1000:.0f} ms "
            f"since last kick (limit {self.timeout * 1000:.0f} ms)"
        )


@dataclass(slots=True)
class Watchdog:
    """A single named timeout."""

    name: str
    timeout: float
    severity: Severity = Severity.FALLBACK
    #: Set false for watchdogs that should not run yet (e.g. vision with no camera).
    enabled: bool = True
    #: ``None`` until the first kick. Not ``0.0``: a simulated or freshly booted
    #: clock legitimately reads zero, and using it as a sentinel would leave
    #: every watchdog permanently disarmed.
    last_kick: float | None = None
    expired: bool = False
    expiry_count: int = 0
    #: Longest gap ever observed between kicks — the headline health number.
    worst_gap: float = 0.0
    detail: str = ""

    def kick(self, now: float) -> bool:
        """Record activity. Returns ``True`` if this cleared an expiry."""
        gap = now - self.last_kick if self.last_kick is not None else 0.0
        self.worst_gap = max(self.worst_gap, gap)
        self.last_kick = now
        if self.expired:
            self.expired = False
            return True
        return False

    def check(self, now: float) -> WatchdogExpiry | None:
        """Test for expiry. Returns an expiry record on the *transition* only."""
        if not self.enabled or self.last_kick is None:
            return None
        elapsed = now - self.last_kick
        if elapsed <= self.timeout:
            return None
        if self.expired:
            return None  # already reported; do not spam
        self.expired = True
        self.expiry_count += 1
        return WatchdogExpiry(
            name=self.name,
            timeout=self.timeout,
            elapsed=elapsed,
            severity=self.severity,
            timestamp=now,
            detail=self.detail,
        )

    def elapsed(self, now: float) -> float:
        return now - self.last_kick if self.last_kick is not None else 0.0

    def margin(self, now: float) -> float:
        """Fraction of the timeout budget still unused, ``[0, 1]``."""
        if not self.enabled or self.last_kick is None:
            return 1.0
        return max(0.0, 1.0 - self.elapsed(now) / max(1e-9, self.timeout))

    def reset(self, now: float) -> None:
        self.last_kick = now
        self.expired = False


class WatchdogGroup:
    """Manages a set of watchdogs and reports expiries once each."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._watchdogs: dict[str, Watchdog] = {}
        #: Called for each expiry transition; wired to the safety monitor.
        self.on_expiry: Callable[[WatchdogExpiry], None] | None = None
        #: Called when a previously expired watchdog is kicked again.
        self.on_recovery: Callable[[str], None] | None = None

    def add(
        self,
        name: str,
        timeout: float,
        *,
        severity: Severity = Severity.FALLBACK,
        enabled: bool = True,
        detail: str = "",
    ) -> Watchdog:
        """Register a watchdog."""
        watchdog = Watchdog(
            name=name, timeout=timeout, severity=severity, enabled=enabled, detail=detail
        )
        self._watchdogs[name] = watchdog
        return watchdog

    def kick(self, name: str) -> None:
        """Record activity for ``name``. Unknown names are ignored deliberately —
        a component that kicks a watchdog nobody registered should not crash."""
        watchdog = self._watchdogs.get(name)
        if watchdog is None:
            return
        if watchdog.kick(self._clock.monotonic()) and self.on_recovery is not None:
            log.info("watchdog recovered", watchdog=name)
            self.on_recovery(name)

    def check_all(self) -> list[WatchdogExpiry]:
        """Check every watchdog; returns new expiries."""
        now = self._clock.monotonic()
        expiries: list[WatchdogExpiry] = []
        for watchdog in self._watchdogs.values():
            expiry = watchdog.check(now)
            if expiry is not None:
                log.error(str(expiry), watchdog=watchdog.name, severity=expiry.severity.name)
                expiries.append(expiry)
                if self.on_expiry is not None:
                    self.on_expiry(expiry)
        return expiries

    def enable(self, name: str, enabled: bool = True) -> None:
        """Enable or disable a watchdog (e.g. vision when no camera is fitted)."""
        watchdog = self._watchdogs.get(name)
        if watchdog is not None:
            watchdog.enabled = enabled
            if enabled:
                watchdog.reset(self._clock.monotonic())

    def reset_all(self) -> None:
        """Reset every timer — used after startup and after fault recovery."""
        now = self._clock.monotonic()
        for watchdog in self._watchdogs.values():
            watchdog.reset(now)

    def get(self, name: str) -> Watchdog | None:
        return self._watchdogs.get(name)

    @property
    def expired(self) -> tuple[str, ...]:
        return tuple(w.name for w in self._watchdogs.values() if w.expired)

    @property
    def worst_severity(self) -> Severity | None:
        """Highest severity among currently expired watchdogs."""
        severities = [w.severity for w in self._watchdogs.values() if w.expired]
        return max(severities) if severities else None

    def status(self) -> list[dict[str, object]]:
        """Per-watchdog status for the diagnostics screen."""
        now = self._clock.monotonic()
        return [
            {
                "name": w.name,
                "enabled": w.enabled,
                "expired": w.expired,
                "timeout_ms": round(w.timeout * 1000, 1),
                "elapsed_ms": round(w.elapsed(now) * 1000, 1),
                "margin": round(w.margin(now), 3),
                "worst_gap_ms": round(w.worst_gap * 1000, 1),
                "expiries": w.expiry_count,
            }
            for w in self._watchdogs.values()
        ]
