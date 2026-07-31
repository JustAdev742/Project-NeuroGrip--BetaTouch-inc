"""Loop rate control and timing statistics.

Real-time behaviour on a Linux SBC is "soft real time": we cannot guarantee a
deadline, but we can *measure* how well we are meeting it and react when we are
not. :class:`RateTimer` provides drift-free pacing; :class:`LoopMonitor` records
period, jitter and overruns so the diagnostics screen can show whether the
control loop is healthy — and so the safety layer can degrade the hand to manual
if it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from .clock import Clock
from .ringbuffer import RingBuffer, RunningStats

__all__ = ["LoopMonitor", "LoopStats", "RateTimer"]


class RateTimer:
    """Fixed-rate pacer that does not accumulate drift.

    Naively sleeping for ``1/rate`` each iteration drifts by the duration of the
    work performed. This schedules against absolute deadlines instead. When the
    loop falls behind by more than ``max_catchup`` periods it resynchronises
    rather than spinning to catch up, which prevents a transient stall from
    turning into a burst of back-to-back control cycles.
    """

    __slots__ = ("_clock", "_max_catchup", "_next", "_period", "missed")

    def __init__(self, clock: Clock, rate_hz: float, *, max_catchup: int = 3) -> None:
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self._clock = clock
        self._period = 1.0 / rate_hz
        self._next = clock.monotonic() + self._period
        self._max_catchup = max_catchup
        #: Count of deadlines missed since construction — reported by diagnostics.
        self.missed = 0

    @property
    def period(self) -> float:
        return self._period

    @property
    def rate_hz(self) -> float:
        return 1.0 / self._period

    def set_rate(self, rate_hz: float) -> None:
        """Change the target rate (Sports Mode raises the control rate)."""
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self._period = 1.0 / rate_hz
        self._next = self._clock.monotonic() + self._period

    def sleep_until_next(self) -> float:
        """Sleep until the next deadline; returns the overshoot in seconds.

        A positive return value means the previous iteration ran long.
        """
        now = self._clock.monotonic()
        delay = self._next - now
        if delay > 0:
            self._clock.sleep(delay)
            self._next += self._period
            return 0.0

        # Deadline missed.
        self.missed += 1
        overshoot = -delay
        if overshoot > self._max_catchup * self._period:
            self._next = self._clock.monotonic() + self._period
        else:
            self._next += self._period
        return overshoot

    def due(self) -> bool:
        """Non-blocking variant: has the next deadline arrived?

        Used by the cooperative scheduler, which multiplexes several rate groups
        onto one thread and therefore must never sleep inside a rate timer.
        """
        now = self._clock.monotonic()
        if now < self._next:
            return False
        behind = now - self._next
        if behind > self._max_catchup * self._period:
            self.missed += 1
            self._next = now + self._period
        else:
            self._next += self._period
        return True

    def time_until_due(self) -> float:
        """Seconds until the next deadline (never negative)."""
        return max(0.0, self._next - self._clock.monotonic())


@dataclass(frozen=True, slots=True)
class LoopStats:
    """Snapshot of a loop's timing behaviour."""

    name: str
    target_hz: float
    actual_hz: float
    mean_period_ms: float
    jitter_ms: float
    p95_period_ms: float
    max_period_ms: float
    mean_work_ms: float
    max_work_ms: float
    overruns: int
    iterations: int

    @property
    def healthy(self) -> bool:
        """True when the loop is within 10% of target and rarely overruns."""
        if self.iterations < 10:
            return True
        rate_ok = self.actual_hz >= self.target_hz * 0.9
        overrun_ok = self.overruns <= max(1, self.iterations // 100)
        return rate_ok and overrun_ok

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "target_hz": round(self.target_hz, 2),
            "actual_hz": round(self.actual_hz, 2),
            "period_ms": round(self.mean_period_ms, 3),
            "jitter_ms": round(self.jitter_ms, 3),
            "p95_ms": round(self.p95_period_ms, 3),
            "work_ms": round(self.mean_work_ms, 3),
            "overruns": self.overruns,
            "iterations": self.iterations,
        }


class LoopMonitor:
    """Measures the real period and work time of a periodic loop."""

    __slots__ = (
        "_clock", "_iterations", "_last", "_name", "_overruns",
        "_periods", "_start", "_stats", "_target", "_work",
    )

    def __init__(self, name: str, target_hz: float, clock: Clock, *, window: int = 256) -> None:
        self._name = name
        self._target = target_hz
        self._clock = clock
        self._periods = RingBuffer(window)
        self._work = RingBuffer(window)
        self._stats = RunningStats()
        self._last: float | None = None
        self._start = 0.0
        self._iterations = 0
        self._overruns = 0

    def begin(self) -> None:
        """Call at the top of each iteration."""
        now = self._clock.monotonic()
        if self._last is not None:
            period = now - self._last
            self._periods.append(period)
            self._stats.add(period)
            if period > 1.5 / self._target:
                self._overruns += 1
        self._last = now
        self._start = now
        self._iterations += 1

    def end(self) -> float:
        """Call at the bottom of each iteration; returns the work duration."""
        duration = self._clock.monotonic() - self._start
        self._work.append(duration)
        return duration

    @property
    def name(self) -> str:
        return self._name

    @property
    def overruns(self) -> int:
        return self._overruns

    def stats(self) -> LoopStats:
        mean_period = self._periods.mean()
        return LoopStats(
            name=self._name,
            target_hz=self._target,
            actual_hz=(1.0 / mean_period) if mean_period > 1e-9 else 0.0,
            mean_period_ms=mean_period * 1000.0,
            jitter_ms=self._periods.std() * 1000.0,
            p95_period_ms=self._periods.percentile(0.95) * 1000.0,
            max_period_ms=self._periods.maximum() * 1000.0,
            mean_work_ms=self._work.mean() * 1000.0,
            max_work_ms=self._work.maximum() * 1000.0,
            overruns=self._overruns,
            iterations=self._iterations,
        )

    def reset(self) -> None:
        self._periods.clear()
        self._work.clear()
        self._stats.reset()
        self._last = None
        self._iterations = 0
        self._overruns = 0
