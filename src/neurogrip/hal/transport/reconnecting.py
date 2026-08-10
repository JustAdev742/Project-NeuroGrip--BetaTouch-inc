"""Transport wrapper that reopens a dropped link.

USB serial links fail in ways that have nothing to do with software: a connector
works loose, a hub browns out, the host's CDC driver renumbers the device, the
controller reboots after a power glitch. Without recovery each of these turns a
working prosthesis into one that needs a restart — while the user is wearing it.

The recovery contract is deliberately narrow:

* **Reopening is never fast-pathed.** Attempts are spaced by exponential backoff
  so a permanently absent device does not spin the control loop.
* **Reconnection never re-arms the actuators.** Reopening the port makes the
  link usable again; it does not decide the hand is safe to move. That decision
  belongs to the safety layer, which sees the same recovery event, and to the
  firmware, whose watchdog has already safed the drive by the time anyone
  notices the link is gone.
* **Recovery is announced, not silent.** ``on_reconnect`` fires after a
  successful reopen so the driver above can resend the state the controller lost
  — limits, calibration, watchdog period — none of which survive a reboot.

Wrapping rather than subclassing keeps this orthogonal: the same wrapper serves
the motor controller, a serial EMG front end, or any future transport, and the
drivers above it stay unaware that reconnection exists.
"""

from __future__ import annotations

from collections.abc import Callable

from ...core.clock import Clock, RealClock
from ...core.errors import CommunicationError, DeviceNotAvailableError
from ...core.logging import get_logger
from ..base import DeviceInfo
from .base import Transport

__all__ = ["ReconnectingTransport"]

log = get_logger(__name__)


class ReconnectingTransport:
    """Wraps a :class:`Transport` and reopens it after a failure.

    Implements :class:`Transport` itself, so it is a drop-in replacement
    everywhere one is accepted.
    """

    def __init__(
        self,
        inner: Transport,
        clock: Clock | None = None,
        *,
        initial_backoff_s: float = 0.5,
        max_backoff_s: float = 8.0,
        backoff_factor: float = 2.0,
        on_reconnect: Callable[[], None] | None = None,
    ) -> None:
        self._inner = inner
        self._clock = clock or RealClock()
        self._initial_backoff = initial_backoff_s
        self._max_backoff = max_backoff_s
        self._factor = backoff_factor
        #: Called after a successful reopen, so the driver can restore any state
        #: the far end lost. Exceptions raised here are logged, not propagated:
        #: a failed restore must not prevent the link from being usable.
        self.on_reconnect = on_reconnect

        self._wanted = False
        self._backoff = initial_backoff_s
        #: ``None`` until the first failure — a clock that starts at zero makes
        #: 0.0 a real timestamp, so it cannot be used as "never".
        self._next_attempt_at: float | None = None
        #: Diagnostics, surfaced through :meth:`info` and the link stats panel.
        self.disconnects = 0
        self.reconnects = 0
        self.failed_attempts = 0

    # -- transport interface --------------------------------------------------

    def open(self) -> None:
        """Open the link.

        A failure here still arms reconnection: a device that is not present at
        startup but appears a moment later — the usual case when the hand is
        powered from the same battery as the host — must recover on its own.
        """
        self._wanted = True
        try:
            self._inner.open()
        except (CommunicationError, DeviceNotAvailableError):
            self._schedule_retry()
            raise
        self._reset_backoff()

    def close(self) -> None:
        self._wanted = False
        self._next_attempt_at = None
        self._inner.close()

    @property
    def is_open(self) -> bool:
        return self._inner.is_open

    def write(self, data: bytes) -> int:
        if not self._inner.is_open:
            self._try_reconnect()
            raise DeviceNotAvailableError("link is down; reconnecting")
        try:
            return self._inner.write(data)
        except (CommunicationError, DeviceNotAvailableError):
            self._handle_failure("write")
            raise

    def read(self, max_bytes: int = 4096) -> bytes:
        if not self._inner.is_open:
            self._try_reconnect()
            return b""
        try:
            return self._inner.read(max_bytes)
        except (CommunicationError, DeviceNotAvailableError):
            self._handle_failure("read")
            return b""

    def info(self) -> DeviceInfo:
        inner = self._inner.info()
        return DeviceInfo(
            name=inner.name,
            kind=inner.kind,
            driver=inner.driver,
            connection=inner.connection,
            firmware_version=inner.firmware_version,
            capabilities=inner.capabilities,
            extra={
                **inner.extra,
                "disconnects": self.disconnects,
                "reconnects": self.reconnects,
                "failed_attempts": self.failed_attempts,
            },
        )

    # -- recovery -------------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._inner.is_open

    @property
    def waiting_to_retry(self) -> bool:
        """True while the link is down and the next attempt is still pending."""
        return self._wanted and not self._inner.is_open

    def _handle_failure(self, operation: str) -> None:
        """Close the link so the next call attempts a clean reopen."""
        self.disconnects += 1
        log.warning("transport failed; will reconnect", operation=operation)
        self._inner.close()
        self._schedule_retry()

    def _schedule_retry(self) -> None:
        self._next_attempt_at = self._clock.monotonic() + self._backoff

    def _reset_backoff(self) -> None:
        self._backoff = self._initial_backoff
        self._next_attempt_at = None

    def _try_reconnect(self) -> None:
        """Attempt a reopen if the backoff has elapsed.

        Called from :meth:`read` and :meth:`write`, so recovery is driven by the
        control loop rather than by a background thread. That keeps every
        transport access on one thread, which is the assumption the drivers above
        are written against.
        """
        if not self._wanted or self._inner.is_open:
            return
        now = self._clock.monotonic()
        if self._next_attempt_at is not None and now < self._next_attempt_at:
            return

        try:
            self._inner.open()
        except (CommunicationError, DeviceNotAvailableError) as exc:
            self.failed_attempts += 1
            self._backoff = min(self._max_backoff, self._backoff * self._factor)
            self._schedule_retry()
            log.throttled(
                "transport-reconnect",
                "warning",
                "reconnect attempt failed",
                now=now,
                error=str(exc),
                next_try_in=round(self._backoff, 2),
            )
            return

        self.reconnects += 1
        self._reset_backoff()
        log.info("transport reconnected", attempts=self.failed_attempts)
        if self.on_reconnect is not None:
            try:
                self.on_reconnect()
            except Exception as exc:
                # A restore failure leaves the far end on its own defaults, which
                # are safe. Losing the link again over it would not be.
                log.error("post-reconnect restore failed", error=str(exc))
