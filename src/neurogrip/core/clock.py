"""Time abstraction.

Nothing in this stack calls :func:`time.monotonic` or :func:`time.sleep` directly.
Every component receives a :class:`Clock`, which buys three things that matter for a
safety-critical device:

* **Deterministic tests.** Watchdog expiry, motion profiles, debounce windows and
  staleness decay can be tested exactly, without sleeping.
* **Faster-than-real-time simulation.** A whole grasp sequence can be replayed in
  microseconds.
* **Replay.** Recorded EMG sessions are re-run against the real pipeline using the
  timestamps that were captured, not wall-clock time.
"""

from __future__ import annotations

import threading
import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "Deadline", "RealClock", "SimulatedClock", "Stopwatch"]


@runtime_checkable
class Clock(Protocol):
    """Monotonic time source with a sleep primitive."""

    def monotonic(self) -> float:
        """Seconds since an arbitrary epoch; never decreases."""
        ...

    def wall(self) -> float:
        """Unix timestamp, for log records and the on-screen clock only.

        Never use this for control logic — it can jump when NTP corrects.
        """
        ...

    def sleep(self, seconds: float) -> None:
        """Block for ``seconds`` (may return early; callers must not assume exact)."""
        ...


class RealClock:
    """Production clock backed by the operating system."""

    __slots__ = ()

    def monotonic(self) -> float:
        return time.monotonic()

    def wall(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimulatedClock:
    """Manually advanced clock for tests, simulation and replay.

    ``sleep`` advances the virtual time instantly, so a loop written against the
    :class:`Clock` protocol runs at full CPU speed under simulation while behaving
    identically in production.

    Thread-safe: the simulated runtime may run services on worker threads.
    """

    __slots__ = ("_lock", "_t", "_wall_epoch")

    def __init__(self, start: float = 0.0, wall_epoch: float = 1_767_225_600.0) -> None:
        self._t = float(start)
        self._wall_epoch = float(wall_epoch)
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            return self._t

    def wall(self) -> float:
        with self._lock:
            return self._wall_epoch + self._t

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> float:
        """Advance virtual time by ``seconds`` and return the new time."""
        if seconds < 0:
            raise ValueError("cannot advance time backwards")
        with self._lock:
            self._t += seconds
            return self._t

    def set(self, value: float) -> None:
        """Jump to an absolute virtual time (used by replay)."""
        with self._lock:
            if value < self._t:
                raise ValueError("cannot move simulated time backwards")
            self._t = float(value)


class Deadline:
    """A point in the future, evaluated against a :class:`Clock`.

    Preferred over storing raw expiry floats because it keeps the clock reference
    and the deadline together, which is what makes watchdog code readable.
    """

    __slots__ = ("_clock", "_expiry")

    def __init__(self, clock: Clock, timeout: float) -> None:
        self._clock = clock
        self._expiry = clock.monotonic() + max(0.0, timeout)

    @property
    def expired(self) -> bool:
        return self._clock.monotonic() >= self._expiry

    @property
    def remaining(self) -> float:
        return max(0.0, self._expiry - self._clock.monotonic())

    def extend(self, seconds: float) -> None:
        self._expiry += seconds

    def reset(self, timeout: float) -> None:
        self._expiry = self._clock.monotonic() + max(0.0, timeout)


class Stopwatch:
    """Elapsed-time helper used by metrics, self-tests and training exercises."""

    __slots__ = ("_clock", "_start")

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._start = clock.monotonic()

    def reset(self) -> None:
        self._start = self._clock.monotonic()

    @property
    def elapsed(self) -> float:
        return self._clock.monotonic() - self._start

    def lap(self) -> float:
        """Return elapsed time and restart."""
        now = self._clock.monotonic()
        elapsed = now - self._start
        self._start = now
        return elapsed
