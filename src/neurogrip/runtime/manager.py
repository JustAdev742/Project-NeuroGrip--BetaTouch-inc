"""Service registry, dependency ordering and supervised restart.

:class:`~neurogrip.runtime.application.Application` already starts and stops the
subsystems in a fixed, hand-written order. That works, but it encodes the
dependency graph implicitly in the order of a function body, which means:

* adding a service requires knowing where in that body it belongs;
* nothing prevents starting a consumer before its producer;
* a service that dies at runtime stays dead.

This module makes the graph explicit and supervises it. It does not replace
``Application`` — the application owns the control loops and the safety
interlocks, which is where the real-time decisions live. It owns *lifecycle*.

Three behaviours are the point of the whole file:

**Dependency order is derived, not declared.** Services state what they need;
the manager topologically sorts them and starts in that order, stopping in
exactly the reverse. That is what guarantees the servo bus is disabled before
the transport it writes to is closed.

**A failing service is contained.** ``required=False`` services that fail to
start are logged and skipped; the rest of the system comes up. Only a required
service aborts startup. This is the difference between "the camera is unplugged"
and "there is no motor controller".

**Restarts back off, and then stop.** A service that crashes is restarted with
exponential backoff, but only a bounded number of times inside a window. A
process that crash-loops forever is worse than one that stays down and says so:
for the motor controller in particular, each restart re-initialises PWM and can
produce a movement transient, so an unbounded restart loop is a safety problem,
not a resilience feature.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from ..core.clock import Clock, RealClock
from ..core.errors import ServiceError
from ..core.events import EventBus
from ..core.lifecycle import HealthReport, HealthStatus, Service
from ..core.logging import get_logger
from ..core.topics import Topics

__all__ = [
    "ManagedService",
    "RestartPolicy",
    "ServiceManager",
    "ServiceState",
]

log = get_logger(__name__)


class ServiceState(str, Enum):
    """Where a service is in its lifecycle, as the manager sees it."""

    REGISTERED = "registered"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    #: Start or restart failed and the retry budget is exhausted. Terminal until
    #: an operator intervenes; distinguished from STOPPED so the diagnostics
    #: screen can show "gave up" rather than "not started".
    FAILED = "failed"
    #: Never started because a dependency did not come up.
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class RestartPolicy:
    """How aggressively to revive a crashed service.

    ``max_restarts`` inside ``window_s`` bounds the crash loop. Defaults are
    conservative: three attempts in a minute, then stop trying.
    """

    enabled: bool = True
    max_restarts: int = 3
    window_s: float = 60.0
    initial_backoff_s: float = 0.5
    max_backoff_s: float = 10.0

    @classmethod
    def never(cls) -> RestartPolicy:
        """For services where an automatic restart is itself hazardous."""
        return cls(enabled=False)


@dataclass(slots=True)
class ManagedService:
    """Registry entry: a service plus the manager's bookkeeping."""

    service: Service
    depends_on: tuple[str, ...] = ()
    required: bool = True
    policy: RestartPolicy = field(default_factory=RestartPolicy)
    state: ServiceState = ServiceState.REGISTERED
    restarts: int = 0
    last_error: str = ""
    #: Monotonic timestamps of recent restart attempts, trimmed to the window.
    _restart_times: list[float] = field(default_factory=list)
    _next_retry_at: float = 0.0

    @property
    def name(self) -> str:
        return self.service.name


class ServiceManager:
    """Owns registration, ordered lifecycle and supervision."""

    def __init__(
        self,
        bus: EventBus | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._bus = bus
        self._clock: Clock = clock or RealClock()
        self._services: dict[str, ManagedService] = {}
        self._order: tuple[str, ...] = ()

    # -- registration ---------------------------------------------------------

    def register(
        self,
        service: Service,
        *,
        depends_on: Iterable[str] = (),
        required: bool = True,
        policy: RestartPolicy | None = None,
    ) -> ManagedService:
        """Add ``service`` to the registry.

        ``depends_on`` names other services that must be running first. Names
        are resolved at :meth:`start_all`, not here, so registration order does
        not matter.
        """
        name = service.name
        if name in self._services:
            raise ServiceError(f"service '{name}' is already registered")
        entry = ManagedService(
            service=service,
            depends_on=tuple(depends_on),
            required=required,
            policy=policy or RestartPolicy(),
        )
        self._services[name] = entry
        self._order = ()  # invalidate the cached topological order
        return entry

    def get(self, name: str) -> ManagedService | None:
        return self._services.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._services)

    # -- ordering -------------------------------------------------------------

    def resolve_order(self) -> tuple[str, ...]:
        """Topologically sort the registry. Raises on cycles or missing deps.

        Deterministic: ties are broken alphabetically so startup order is stable
        across runs, which matters when reading two boot logs side by side.
        """
        if self._order:
            return self._order

        for entry in self._services.values():
            for dep in entry.depends_on:
                if dep not in self._services:
                    raise ServiceError(
                        f"service '{entry.name}' depends on unknown service '{dep}'",
                        context={"known": sorted(self._services)},
                    )

        resolved: list[str] = []
        pending = {n: set(e.depends_on) for n, e in self._services.items()}
        while pending:
            ready = sorted(n for n, deps in pending.items() if not deps - set(resolved))
            if not ready:
                raise ServiceError(
                    "circular service dependency",
                    context={"unresolved": sorted(pending)},
                )
            for name in ready:
                resolved.append(name)
                del pending[name]
        self._order = tuple(resolved)
        return self._order

    # -- lifecycle ------------------------------------------------------------

    def start_all(self) -> bool:
        """Start every service in dependency order.

        Returns ``True`` if every *required* service is running. Optional
        services that fail are logged and skipped.
        """
        ok = True
        for name in self.resolve_order():
            entry = self._services[name]
            if not self._dependencies_satisfied(entry):
                entry.state = ServiceState.SKIPPED
                entry.last_error = "dependency unavailable"
                log.warning("service skipped", service=name, reason=entry.last_error)
                if entry.required:
                    ok = False
                continue
            if not self._start_one(entry) and entry.required:
                ok = False
        return ok

    def _dependencies_satisfied(self, entry: ManagedService) -> bool:
        return all(
            self._services[dep].state is ServiceState.RUNNING for dep in entry.depends_on
        )

    def _start_one(self, entry: ManagedService) -> bool:
        entry.state = ServiceState.STARTING
        try:
            entry.service.start()
        except Exception as exc:
            entry.state = ServiceState.FAILED
            entry.last_error = f"{type(exc).__name__}: {exc}"
            log.error("service failed to start", service=entry.name, error=entry.last_error)
            self._publish_error(entry, "start failed")
            return False
        entry.state = ServiceState.RUNNING
        entry.last_error = ""
        log.info("service started", service=entry.name)
        return True

    def stop_all(self) -> None:
        """Stop every running service in exact reverse dependency order.

        Never raises: shutdown runs on the crash path too, and an exception here
        would mask the original fault and leave later services un-stopped.
        """
        for name in reversed(self.resolve_order()):
            entry = self._services[name]
            if entry.state is not ServiceState.RUNNING:
                continue
            entry.state = ServiceState.STOPPING
            try:
                entry.service.stop()
            except Exception as exc:
                entry.last_error = f"{type(exc).__name__}: {exc}"
                log.error("service failed to stop", service=name, error=entry.last_error)
            entry.state = ServiceState.STOPPED

    def restart(self, name: str, *, force: bool = False) -> bool:
        """Stop and restart one service.

        ``force`` bypasses the restart budget — for an operator-initiated
        restart from the diagnostics screen, where a human has decided.
        """
        entry = self._services.get(name)
        if entry is None:
            raise ServiceError(f"unknown service '{name}'")

        now = self._clock.monotonic()
        if not force:
            if not entry.policy.enabled:
                log.info("restart suppressed by policy", service=name)
                return False
            if not self._within_budget(entry, now):
                entry.state = ServiceState.FAILED
                entry.last_error = "restart budget exhausted"
                log.error("giving up on service", service=name, restarts=entry.restarts)
                self._publish_error(entry, "restart budget exhausted")
                return False

        try:
            entry.service.stop()
        except Exception as exc:
            log.warning("stop during restart failed", service=name, error=str(exc))

        entry.restarts += 1
        entry._restart_times.append(now)
        backoff = min(
            entry.policy.max_backoff_s,
            entry.policy.initial_backoff_s * (2 ** max(0, entry.restarts - 1)),
        )
        entry._next_retry_at = now + backoff

        started = self._start_one(entry)
        if started:
            log.info("service restarted", service=name, attempt=entry.restarts)
            self._publish_health(entry)
        return started

    def _within_budget(self, entry: ManagedService, now: float) -> bool:
        window = entry.policy.window_s
        entry._restart_times = [t for t in entry._restart_times if now - t <= window]
        return len(entry._restart_times) < entry.policy.max_restarts

    # -- supervision ----------------------------------------------------------

    def supervise(self) -> tuple[str, ...]:
        """Check health and restart anything that has failed.

        Called from the application's slow loop (a few hertz is plenty). Returns
        the names of services that were restarted this pass, so the caller can
        log or surface them.

        Only ``HealthStatus.FAILED`` triggers a restart. ``DEGRADED`` explicitly
        does not: a degraded service is still producing usable output, and
        bouncing it would turn a partial outage into a total one.
        """
        restarted: list[str] = []
        now = self._clock.monotonic()
        for name in self.resolve_order():
            entry = self._services[name]
            if entry.state is not ServiceState.RUNNING:
                continue
            if now < entry._next_retry_at:
                continue
            try:
                report = entry.service.health()
            except Exception as exc:
                log.error("health check raised", service=name, error=str(exc))
                continue
            if report.status is HealthStatus.FAILED:
                log.warning("service unhealthy, restarting", service=name, detail=report.detail)
                if self.restart(name):
                    restarted.append(name)
        return tuple(restarted)

    # -- reporting ------------------------------------------------------------

    def health(self) -> list[HealthReport]:
        reports: list[HealthReport] = []
        for name in self.resolve_order():
            entry = self._services[name]
            if entry.state in (ServiceState.FAILED, ServiceState.SKIPPED):
                reports.append(
                    HealthReport.failed(name, entry.last_error or entry.state.value)
                )
                continue
            if entry.state is not ServiceState.RUNNING:
                reports.append(HealthReport.offline(name, entry.state.value))
                continue
            try:
                reports.append(entry.service.health())
            except Exception as exc:
                reports.append(HealthReport.failed(name, f"health check raised: {exc}"))
        return reports

    def overall_health(self) -> HealthStatus:
        """Worst status across all *required* services.

        Optional services are excluded deliberately: a missing camera should not
        make an otherwise healthy hand report FAILED, or the indicator stops
        meaning anything.
        """
        statuses = [
            r.status
            for r in self.health()
            if self._services[r.name].required
        ]
        return max(statuses, default=HealthStatus.OK)

    def describe(self) -> list[dict[str, object]]:
        """Registry snapshot for the diagnostics screen."""
        return [
            {
                "name": name,
                "state": self._services[name].state.value,
                "required": self._services[name].required,
                "depends_on": list(self._services[name].depends_on),
                "restarts": self._services[name].restarts,
                "error": self._services[name].last_error,
            }
            for name in self.resolve_order()
        ]

    # -- bus plumbing ---------------------------------------------------------

    def _publish_health(self, entry: ManagedService) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            Topics.SERVICE_HEALTH,
            {"service": entry.name, "state": entry.state.value, "restarts": entry.restarts},
            source="service-manager",
        )

    def _publish_error(self, entry: ManagedService, detail: str) -> None:
        if self._bus is None:
            return
        self._bus.publish(
            Topics.SERVICE_ERROR,
            {"service": entry.name, "detail": detail, "error": entry.last_error},
            source="service-manager",
        )
