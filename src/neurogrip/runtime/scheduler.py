"""Cooperative rate-group scheduler.

Everything runs on **one thread**. That is a deliberate choice for a
safety-relevant control system: with a single thread there are no data races
between the control loop and the UI, no lock ordering to get wrong, and no
priority inversion. The cost is that a slow task delays the others — which the
scheduler measures and reports, so the cost is visible rather than mysterious.

Rate groups and their typical rates::

    control      200 Hz   read servos, run trajectory, write targets
    emg          200 Hz   drain samples, filter, classify, estimate intent
    decision     100 Hz   safety evaluation, mode update, fusion
    vision        20 Hz   capture and inference (its own group: it is the slow one)
    ui            15 Hz   assemble the view model and render
    diagnostics    2 Hz   health, metrics, resource sampling

Long-running work that genuinely cannot fit — model inference on a large image,
disk writes — belongs on a worker thread behind a
:class:`~neurogrip.core.events.QueuedSubscriber`, not in a rate group.

Each group is measured by a :class:`~neurogrip.core.rate.LoopMonitor`; overruns
and jitter feed the diagnostics screen and the control-loop watchdog.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.clock import Clock
from ..core.lifecycle import HealthReport, HealthStatus
from ..core.logging import get_logger
from ..core.rate import LoopMonitor, LoopStats, RateTimer

__all__ = ["RateGroup", "Scheduler"]

log = get_logger(__name__)


@dataclass(slots=True)
class RateGroup:
    """One periodic task."""

    name: str
    rate_hz: float
    task: Callable[[], object]
    #: Groups with a higher priority run first when several are due at once.
    priority: int = 0
    enabled: bool = True
    #: Consecutive exceptions before the group is disabled to protect the rest.
    max_consecutive_errors: int = 20
    monitor: LoopMonitor | None = None
    timer: RateTimer | None = None
    errors: int = 0
    consecutive_errors: int = 0
    last_error: str = ""
    executions: int = 0


class Scheduler:
    """Runs rate groups on a single thread against deadlines."""

    def __init__(self, clock: Clock, *, idle_sleep: float = 0.0005) -> None:
        self._clock = clock
        self._groups: list[RateGroup] = []
        #: How long to sleep when nothing is due. Small enough to keep jitter
        #: below a millisecond, large enough not to spin a core.
        self._idle_sleep = idle_sleep
        self._running = False
        self._iterations = 0
        #: Called with each group's stats after every execution.
        self.on_stats: Callable[[LoopStats], None] | None = None

    # -- registration ---------------------------------------------------------

    def add(
        self,
        name: str,
        rate_hz: float,
        task: Callable[[], object],
        *,
        priority: int = 0,
        enabled: bool = True,
    ) -> RateGroup:
        """Register a periodic task."""
        group = RateGroup(
            name=name,
            rate_hz=rate_hz,
            task=task,
            priority=priority,
            enabled=enabled,
            monitor=LoopMonitor(name, rate_hz, self._clock),
            timer=RateTimer(self._clock, rate_hz),
        )
        self._groups.append(group)
        self._groups.sort(key=lambda g: -g.priority)
        return group

    def group(self, name: str) -> RateGroup | None:
        return next((g for g in self._groups if g.name == name), None)

    def set_rate(self, name: str, rate_hz: float) -> bool:
        """Change a group's rate at runtime (modes do this for control/vision)."""
        group = self.group(name)
        if group is None or group.timer is None:
            return False
        group.rate_hz = rate_hz
        if rate_hz <= 0:
            group.enabled = False
            return True
        group.enabled = True
        group.timer.set_rate(rate_hz)
        group.monitor = LoopMonitor(name, rate_hz, self._clock)
        return True

    def enable(self, name: str, enabled: bool = True) -> None:
        group = self.group(name)
        if group is not None:
            group.enabled = enabled

    @property
    def groups(self) -> tuple[RateGroup, ...]:
        return tuple(self._groups)

    @property
    def iterations(self) -> int:
        return self._iterations

    # -- execution ------------------------------------------------------------

    def step(self) -> int:
        """Run every group whose deadline has arrived. Returns how many ran.

        Non-blocking: this is the unit the ``run`` loop and the tests both use,
        which means a test can drive the entire application deterministically by
        advancing a :class:`~neurogrip.core.clock.SimulatedClock` and calling
        ``step``.
        """
        executed = 0
        self._iterations += 1

        for group in self._groups:
            if not group.enabled or group.timer is None:
                continue
            if not group.timer.due():
                continue

            if group.monitor is not None:
                group.monitor.begin()
            try:
                group.task()
                group.consecutive_errors = 0
            except Exception as exc:
                group.errors += 1
                group.consecutive_errors += 1
                group.last_error = str(exc)
                log.error(
                    "rate group raised",
                    group=group.name,
                    error=str(exc),
                    consecutive=group.consecutive_errors,
                    exc_info=True,
                )
                if group.consecutive_errors >= group.max_consecutive_errors:
                    group.enabled = False
                    log.critical(
                        "rate group disabled after repeated failures", group=group.name
                    )
            finally:
                if group.monitor is not None:
                    group.monitor.end()
                    if self.on_stats is not None:
                        self.on_stats(group.monitor.stats())
                group.executions += 1
            executed += 1

        return executed

    def run(self, *, until: Callable[[], bool] | None = None) -> None:
        """Run until stopped.

        Sleeps for the shorter of the idle interval and the time until the next
        deadline, so the loop neither spins nor oversleeps a deadline.
        """
        self._running = True
        log.info(
            "scheduler started",
            groups={g.name: g.rate_hz for g in self._groups if g.enabled},
        )
        try:
            while self._running and (until is None or not until()):
                executed = self.step()
                if executed == 0:
                    self._clock.sleep(min(self._idle_sleep, self._time_until_next()))
        finally:
            self._running = False
            log.info("scheduler stopped", iterations=self._iterations)

    def stop(self) -> None:
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def _time_until_next(self) -> float:
        return min(
            (g.timer.time_until_due() for g in self._groups if g.enabled and g.timer),
            default=self._idle_sleep,
        )

    # -- reporting ------------------------------------------------------------

    def stats(self) -> list[LoopStats]:
        return [g.monitor.stats() for g in self._groups if g.monitor is not None]

    def health(self) -> HealthReport:
        disabled = [g.name for g in self._groups if not g.enabled and g.errors]
        if disabled:
            return HealthReport.failed(
                "scheduler", f"groups disabled after repeated errors: {', '.join(disabled)}"
            )
        unhealthy = [s for s in self.stats() if not s.healthy]
        if unhealthy:
            return HealthReport(
                name="scheduler",
                status=HealthStatus.DEGRADED,
                detail="; ".join(
                    f"{s.name} {s.actual_hz:.0f}/{s.target_hz:.0f} Hz "
                    f"({s.overruns} overruns)"
                    for s in unhealthy
                ),
                metrics={"iterations": self._iterations},
            )
        return HealthReport.ok("scheduler", iterations=self._iterations, groups=len(self._groups))
