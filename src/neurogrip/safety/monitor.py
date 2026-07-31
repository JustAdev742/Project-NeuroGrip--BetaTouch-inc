"""The safety monitor.

Evaluates every rule and every watchdog each cycle, folds the results into one
:class:`SafetyState`, and takes the corresponding action. It is the only
component that decides whether motion and AI assistance are permitted, and its
verdict is consumed by :mod:`neurogrip.fusion` as gate 1 — which nothing can
bypass.

Design points worth calling out:

* **Fail closed for motion, fail open for the user.** A ``CRITICAL`` fault stops
  the actuators. A ``FALLBACK`` fault removes the *AI*, not the hand: direct
  manual control continues, because a person mid-task with a failed camera needs
  their hand more than they need assistance.
* **Faults latch until they clear *and* settle.** A condition that oscillates
  around its threshold would otherwise produce a hand that flickers between
  states. A cleared fault must stay clear for ``clear_hold_s`` before the system
  acts on the recovery.
* **Critical faults require acknowledgement.** They engage the e-stop, and the
  e-stop is a latch (see :mod:`neurogrip.safety.estop`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.clock import Clock
from ..core.errors import Severity
from ..core.events import EventBus
from ..core.lifecycle import HealthReport, HealthStatus, ServiceBase
from ..core.logging import get_logger
from ..core.topics import Topics
from ..core.types import clamp
from .estop import EmergencyStop, EstopSource
from .rules import DEFAULT_RULES, Fault, SafetyContext, SafetyRule
from .watchdog import WatchdogExpiry, WatchdogGroup

__all__ = ["SafetyMonitor", "SafetyState"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SafetyState:
    """The safety verdict for one cycle."""

    #: Whether the hand may move at all.
    motion_allowed: bool = True
    #: Whether the AI assistive path may contribute.
    ai_allowed: bool = True
    #: Hard ceiling on grip force, ``[0, 1]``.
    force_ceiling: float = 1.0
    #: Worst severity currently active.
    severity: Severity = Severity.MINOR
    #: All currently active faults, worst first.
    faults: tuple[Fault, ...] = field(default_factory=tuple)
    estop_engaged: bool = False
    timestamp: float = 0.0

    @property
    def is_nominal(self) -> bool:
        return self.motion_allowed and self.ai_allowed and not self.faults

    @property
    def primary_reason(self) -> str:
        """One-line explanation for the fusion layer and the UI banner."""
        if self.estop_engaged:
            return "emergency stop engaged"
        if self.faults:
            return self.faults[0].message
        return ""

    @property
    def remedies(self) -> tuple[str, ...]:
        return tuple(f.remedy for f in self.faults if f.remedy)

    def fault(self, code: str) -> Fault | None:
        for item in self.faults:
            if item.code == code:
                return item
        return None

    @classmethod
    def nominal(cls, timestamp: float = 0.0) -> SafetyState:
        return cls(timestamp=timestamp)


@dataclass(slots=True)
class _ActiveFault:
    fault: Fault
    raised_at: float
    #: When the underlying condition stopped being reported; ``0`` while active.
    cleared_at: float = 0.0
    occurrences: int = 1


class SafetyMonitor(ServiceBase):
    """Evaluates rules and watchdogs; owns the motion/AI permissions."""

    service_name = "safety"

    def __init__(
        self,
        clock: Clock,
        bus: EventBus,
        estop: EmergencyStop,
        watchdogs: WatchdogGroup,
        *,
        rules: tuple[SafetyRule, ...] | None = None,
        clear_hold_s: float = 1.0,
    ) -> None:
        super().__init__()
        self._clock = clock
        self._bus = bus
        self._estop = estop
        self._watchdogs = watchdogs
        self._rules: list[SafetyRule] = list(
            rules if rules is not None else tuple(rule() for rule in DEFAULT_RULES)
        )
        #: A cleared fault must stay clear this long before it is dropped.
        self._clear_hold = clear_hold_s
        self._active: dict[str, _ActiveFault] = {}
        self._state = SafetyState.nominal()
        #: Fault codes seen this session, for the incident log.
        self._history: list[tuple[float, str, str]] = []
        self.evaluations = 0

        self._watchdogs.on_expiry = self._on_watchdog_expiry

    # -- lifecycle ------------------------------------------------------------

    def on_start(self) -> None:
        self._watchdogs.reset_all()
        self._state = SafetyState.nominal(self._clock.monotonic())
        log.info("safety monitor started", rules=[r.name for r in self._rules])
        disabled = [r.name for r in self._rules if not r.enabled]
        if disabled:
            # Never let a disabled safety rule be invisible.
            log.warning("SAFETY RULES DISABLED", rules=disabled)

    # -- configuration --------------------------------------------------------

    @property
    def rules(self) -> tuple[SafetyRule, ...]:
        return tuple(self._rules)

    def add_rule(self, rule: SafetyRule) -> None:
        self._rules.append(rule)

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def estop(self) -> EmergencyStop:
        return self._estop

    @property
    def watchdogs(self) -> WatchdogGroup:
        return self._watchdogs

    # -- evaluation -----------------------------------------------------------

    def evaluate(self, context: SafetyContext) -> SafetyState:
        """Run every rule and watchdog; update and return the safety state."""
        self.evaluations += 1
        now = context.timestamp or self._clock.monotonic()

        self._watchdogs.check_all()

        reported: dict[str, Fault] = {}
        for rule in self._rules:
            if not rule.enabled:
                continue
            try:
                fault = rule.evaluate(context)
            except Exception as exc:
                log.error("safety rule raised", rule=rule.name, error=str(exc))
                fault = Fault(
                    code=f"rule_error:{rule.name}",
                    severity=Severity.DEGRADED,
                    message=f"safety rule '{rule.name}' failed to evaluate",
                    rule=rule.name,
                )
            if fault is not None:
                reported[fault.code] = fault

        for expiry in self._expired_watchdog_faults(context, now):
            reported[expiry.code] = expiry

        self._reconcile(reported, now)
        state = self._build_state(now)
        self._apply(state)
        if state.estop_engaged != self._estop.engaged:
            # _apply may have engaged the e-stop; rebuild so the published state
            # reflects the world as it is *after* the response, not before it.
            state = self._build_state(now)
        self._state = state
        self._bus.publish(Topics.SAFETY_STATE, state, source=self.name)
        return state

    def _expired_watchdog_faults(self, context: SafetyContext, now: float) -> list[Fault]:
        """Turn currently-expired watchdogs into faults."""
        faults: list[Fault] = []
        for name in self._watchdogs.expired:
            watchdog = self._watchdogs.get(name)
            if watchdog is None:
                continue
            faults.append(
                Fault(
                    code=f"watchdog:{name}",
                    severity=watchdog.severity,
                    message=(
                        f"'{name}' watchdog expired "
                        f"({watchdog.elapsed(now) * 1000:.0f} ms > "
                        f"{watchdog.timeout * 1000:.0f} ms)"
                    ),
                    rule="watchdog",
                    detail={"watchdog": name},
                    force_ceiling=0.0 if watchdog.severity >= Severity.CRITICAL else 1.0,
                    remedy=watchdog.detail or "Transient stall; the system will recover.",
                )
            )
        return faults

    def _reconcile(self, reported: dict[str, Fault], now: float) -> None:
        """Latch new faults and retire ones that have stayed clear."""
        for code, fault in reported.items():
            existing = self._active.get(code)
            if existing is None:
                self._active[code] = _ActiveFault(fault=fault, raised_at=now)
                self._history.append((now, code, fault.message))
                log.error(
                    "safety fault raised",
                    code=code,
                    severity=fault.severity.name,
                    detail=fault.message,
                )
                self._bus.publish(Topics.SAFETY_FAULT_RAISED, fault, source=self.name)
            else:
                existing.fault = fault
                existing.cleared_at = 0.0
                existing.occurrences += 1

        for code, active in list(self._active.items()):
            if code in reported:
                continue
            if active.cleared_at == 0.0:
                active.cleared_at = now
                continue
            if now - active.cleared_at >= self._clear_hold:
                del self._active[code]
                log.info("safety fault cleared", code=code)
                self._bus.publish(Topics.SAFETY_FAULT_CLEARED, {"code": code}, source=self.name)

    def _build_state(self, now: float) -> SafetyState:
        faults = sorted(
            (a.fault for a in self._active.values()), key=lambda f: f.severity, reverse=True
        )
        severity = max((f.severity for f in faults), default=Severity.MINOR)
        force_ceiling = min((f.force_ceiling for f in faults), default=1.0)

        estop_engaged = self._estop.engaged
        motion_allowed = not estop_engaged and severity < Severity.CRITICAL
        ai_allowed = motion_allowed and severity < Severity.FALLBACK

        return SafetyState(
            motion_allowed=motion_allowed,
            ai_allowed=ai_allowed,
            force_ceiling=clamp(force_ceiling),
            severity=severity,
            faults=tuple(faults),
            estop_engaged=estop_engaged,
            timestamp=now,
        )

    def _apply(self, state: SafetyState) -> None:
        """Engage the e-stop for critical faults."""
        if state.severity >= Severity.CRITICAL and not self._estop.engaged:
            critical = next(
                (f for f in state.faults if f.severity >= Severity.CRITICAL), None
            )
            self._estop.engage(
                EstopSource.SAFETY_RULE,
                critical.message if critical else "critical safety fault",
            )

    def _on_watchdog_expiry(self, expiry: WatchdogExpiry) -> None:
        """Watchdog callback — critical expiries stop the hand immediately.

        Handled here rather than waiting for the next :meth:`evaluate` so that a
        stalled control loop (whose expiry means ``evaluate`` may not run again
        promptly) still triggers the stop.
        """
        self._bus.publish(Topics.WATCHDOG_EXPIRED, expiry, source=self.name)
        if expiry.severity >= Severity.CRITICAL:
            self._estop.engage(EstopSource.WATCHDOG, str(expiry))

    # -- operations -----------------------------------------------------------

    def acknowledge(self, source: str = "user:ui") -> bool:
        """Clear latched faults and release the e-stop, if conditions allow.

        Refuses while a critical condition is still being reported: the user can
        acknowledge a fault, but they cannot acknowledge away a hand that is
        still over-temperature.
        """
        still_critical = [
            a.fault
            for a in self._active.values()
            if a.fault.severity >= Severity.CRITICAL and a.cleared_at == 0.0
        ]
        if still_critical:
            log.warning(
                "cannot acknowledge: critical faults are still active",
                faults=[f.code for f in still_critical],
            )
            return False

        self._active.clear()
        self._watchdogs.reset_all()
        released = self._estop.release(source)
        self._state = SafetyState.nominal(self._clock.monotonic())
        log.info("safety faults acknowledged", source=source, estop_released=released is not None)
        return True

    def trigger_estop(self, reason: str, source: str = EstopSource.USER_UI) -> None:
        """Engage the emergency stop from the UI or a hardware button."""
        self._estop.engage(source, reason)
        self._state = self._build_state(self._clock.monotonic())
        self._bus.publish(Topics.SAFETY_STATE, self._state, source=self.name)

    # -- reporting ------------------------------------------------------------

    def health(self) -> HealthReport:
        state = self._state
        if state.estop_engaged:
            return HealthReport.failed(self.name, f"emergency stop: {self._estop.summary}")
        if state.severity >= Severity.CRITICAL:
            return HealthReport.failed(self.name, state.primary_reason)
        if state.severity >= Severity.FALLBACK:
            return HealthReport.degraded(self.name, state.primary_reason)
        if state.faults:
            return HealthReport(
                name=self.name,
                status=HealthStatus.DEGRADED,
                detail=state.primary_reason,
                metrics={"faults": len(state.faults)},
            )
        return HealthReport.ok(
            self.name, evaluations=self.evaluations, rules=len(self._rules)
        )

    def fault_history(self, limit: int = 20) -> list[tuple[float, str, str]]:
        """Recent fault occurrences for the diagnostics log."""
        return self._history[-limit:]
