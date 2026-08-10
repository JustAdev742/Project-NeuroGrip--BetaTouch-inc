"""A very small dependency-injection container.

Why not just construct everything in ``main()``? Because the composition root has
to build ~25 collaborating objects whose concrete types depend on configuration
(simulated servo bus vs. ESP32 bus, HGGD-MCU vs. mock vision, Tk vs. text UI). A
container gives us:

* one place where "which implementation" is decided (see
  :mod:`neurogrip.runtime.bootstrap`);
* lazy construction, so an unused backend is never even imported;
* ordered startup/shutdown of everything that implements
  :class:`~neurogrip.core.lifecycle.Service`;
* trivial substitution in tests — register a fake, resolve the real graph.

It intentionally does *not* do autowiring by type annotations. Explicit factories
are easier to follow than reflection, and the wiring is read far more often than
it is written.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, TypeVar

from .errors import ConfigurationError
from .lifecycle import HealthReport, Service

__all__ = ["Container", "ServiceRegistry"]

T = TypeVar("T")


@dataclass(slots=True)
class _Registration:
    factory: Callable[[Container], Any]
    singleton: bool
    instance: Any = None
    resolving: bool = False
    #: Registration order, used to give deterministic startup sequencing.
    order: int = 0


class Container:
    """Keyed service locator with lazy singletons and cycle detection."""

    def __init__(self) -> None:
        self._registry: dict[str, _Registration] = {}
        self._lock = threading.RLock()
        self._counter = 0

    # -- registration ---------------------------------------------------------

    def register(
        self,
        key: str,
        factory: Callable[[Container], T],
        *,
        singleton: bool = True,
        replace: bool = False,
    ) -> None:
        """Register ``factory`` under ``key``.

        Re-registering without ``replace=True`` is an error: silently shadowing a
        service is how you end up with two servo buses fighting over one port.
        """
        with self._lock:
            if key in self._registry and not replace:
                raise ConfigurationError(f"service '{key}' is already registered")
            self._counter += 1
            self._registry[key] = _Registration(
                factory=factory, singleton=singleton, order=self._counter
            )

    def register_instance(self, key: str, instance: Any, *, replace: bool = False) -> None:
        """Register an already-constructed object (config, clock, event bus)."""
        with self._lock:
            if key in self._registry and not replace:
                raise ConfigurationError(f"service '{key}' is already registered")
            self._counter += 1
            self._registry[key] = _Registration(
                factory=lambda _: instance,
                singleton=True,
                instance=instance,
                order=self._counter,
            )

    # -- resolution -----------------------------------------------------------

    def resolve(self, key: str) -> Any:
        """Construct (or return the cached) service for ``key``."""
        with self._lock:
            registration = self._registry.get(key)
            if registration is None:
                raise ConfigurationError(
                    f"service '{key}' is not registered",
                    context={"available": sorted(self._registry)},
                )
            if registration.singleton and registration.instance is not None:
                return registration.instance
            if registration.resolving:
                raise ConfigurationError(f"circular dependency while resolving '{key}'")
            registration.resolving = True
        try:
            instance = registration.factory(self)
        finally:
            with self._lock:
                registration.resolving = False
        if registration.singleton:
            with self._lock:
                registration.instance = instance
        return instance

    def try_resolve(self, key: str, default: Any = None) -> Any:
        """Resolve ``key`` or return ``default`` when it is not registered."""
        if key not in self._registry:
            return default
        return self.resolve(key)

    def __contains__(self, key: str) -> bool:
        return key in self._registry

    def keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._registry))

    def instantiated(self) -> dict[str, Any]:
        """Every singleton that has actually been constructed so far."""
        with self._lock:
            return {
                key: reg.instance for key, reg in self._registry.items() if reg.instance is not None
            }

    def eager_resolve(self, keys: tuple[str, ...]) -> None:
        """Force construction of ``keys`` in order (used at startup)."""
        for key in keys:
            self.resolve(key)


class ServiceRegistry:
    """Ordered collection of :class:`Service` objects with grouped lifecycle.

    Startup order is registration order; shutdown is exactly the reverse. That
    single rule removes an entire class of teardown bugs, e.g. closing the serial
    transport while the servo bus still wants to send a "disable" frame.
    """

    def __init__(self) -> None:
        self._services: list[Service] = []
        self._started: list[Service] = []

    def add(self, service: Service) -> Service:
        """Register a service and return it (so calls can be chained inline)."""
        self._services.append(service)
        return service

    def extend(self, services: Iterator[Service]) -> None:
        for service in services:
            self.add(service)

    def __iter__(self) -> Iterator[Service]:
        return iter(self._services)

    def __len__(self) -> int:
        return len(self._services)

    def start_all(self) -> None:
        """Start every service in order.

        If one fails, everything already started is stopped in reverse order
        before the error propagates — a partially initialised device must never
        be left holding actuator power.
        """
        for service in self._services:
            try:
                service.start()
            except Exception:
                self.stop_all()
                raise
            self._started.append(service)

    def stop_all(self) -> None:
        """Stop started services in reverse order; never raises."""
        errors: list[tuple[str, BaseException]] = []
        while self._started:
            service = self._started.pop()
            try:
                service.stop()
            except Exception as exc:
                errors.append((service.name, exc))
        if errors:
            # Surfaced through health(), not raised: shutdown has no caller to
            # meaningfully handle it, and the remaining services still stopped.
            self.shutdown_errors = errors  # type: ignore[attr-defined]

    def health(self) -> list[HealthReport]:
        """Collect health from every registered service."""
        reports: list[HealthReport] = []
        for service in self._services:
            try:
                reports.append(service.health())
            except Exception as exc:
                reports.append(HealthReport.failed(service.name, f"health check raised: {exc}"))
        return reports
