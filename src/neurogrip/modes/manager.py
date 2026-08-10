"""Mode manager.

Owns which mode is active and arbitrates changes. Rules:

* **Safety can veto.** A mode change is refused while a critical fault is active,
  and the manager force-switches to Manual when safety withdraws AI permission.
* **Transitions are declared**, using the shared
  :class:`~neurogrip.core.state.StateMachine`, so an unexpected transition is an
  explicit refusal rather than an undefined state.
* **Exit before enter, always.** The outgoing mode stops its motion before the
  incoming one configures the controller, so no command can straddle a change.
* **Debounced.** A minimum dwell prevents the hands-free double-pulse gesture
  from cycling modes faster than the user can read the screen.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.clock import Clock
from ..core.errors import ModeTransitionError, Severity
from ..core.events import EventBus
from ..core.lifecycle import HealthReport, ServiceBase
from ..core.logging import get_logger
from ..core.state import StateMachine
from ..core.topics import Topics
from ..core.types import ModeId
from ..emg.intent import IntentEngine
from ..fusion.fusion import Decision
from ..safety.monitor import SafetyState
from .base import ModeBase, ModeContext

__all__ = ["ModeChange", "ModeManager"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModeChange:
    """Record of a mode transition, published on the event bus."""

    previous: ModeId | None
    current: ModeId
    reason: str
    timestamp: float
    forced: bool = False


class ModeManager(ServiceBase):
    """Selects and runs the active operating mode."""

    service_name = "modes"

    def __init__(
        self,
        modes: dict[ModeId, ModeBase],
        clock: Clock,
        bus: EventBus,
        intent_engine: IntentEngine,
        *,
        default_mode: ModeId = ModeId.AI_ASSIST,
        min_dwell_s: float = 1.0,
    ) -> None:
        super().__init__()
        if not modes:
            raise ModeTransitionError("no operating modes were registered")
        self._modes = modes
        self._clock = clock
        self._bus = bus
        self._intent = intent_engine
        #: Minimum time in a mode before another change is accepted.
        self._min_dwell = min_dwell_s
        self._default = default_mode if default_mode in modes else next(iter(modes))
        self._current: ModeId | None = None
        self._changed_at = 0.0
        self._history: list[ModeChange] = []
        self._machine = self._build_machine()
        #: Order used by the hands-free toggle gesture and the UI carousel.
        self._cycle = tuple(m for m in (ModeId.AI_ASSIST, ModeId.MANUAL, ModeId.SPORTS) if m in modes)
        #: Mode the user had selected before an automatic fallback, so it can be
        #: restored when the cause clears. ``None`` when the current mode was
        #: chosen by the user.
        self._preferred: ModeId | None = None
        self._nominal_since: float | None = None
        #: Safety must stay nominal this long before an automatic restore.
        self._restore_settle_s = 2.0
        self.changes = 0
        self.rejections = 0

    def _build_machine(self) -> StateMachine[ModeId]:
        machine: StateMachine[ModeId] = StateMachine(self._default, self._clock, name="modes")
        # Every mode can reach every other mode; the guards below are what
        # actually restrict transitions. Declaring them explicitly keeps the
        # generated diagram honest (see `neurogrip diagnose --dot`).
        for source in self._modes:
            for target in self._modes:
                if source is not target:
                    machine.allow(source, target, trigger=f"select:{target.value}")
        return machine

    # -- lifecycle ------------------------------------------------------------

    def on_start(self) -> None:
        self.activate(self._default, reason="startup", force=True)

    def on_stop(self) -> None:
        if self._current is not None:
            self._modes[self._current].on_exit(None)
            self._current = None

    # -- selection ------------------------------------------------------------

    @property
    def current(self) -> ModeId | None:
        return self._current

    @property
    def active(self) -> ModeBase | None:
        return self._modes[self._current] if self._current is not None else None

    @property
    def available(self) -> tuple[ModeId, ...]:
        return tuple(self._modes)

    @property
    def selectable(self) -> tuple[ModeId, ...]:
        return tuple(m for m, mode in self._modes.items() if mode.profile.user_selectable)

    def mode(self, mode_id: ModeId) -> ModeBase | None:
        return self._modes.get(mode_id)

    def activate(
        self,
        mode_id: ModeId,
        *,
        reason: str = "user request",
        force: bool = False,
        safety: SafetyState | None = None,
    ) -> bool:
        """Switch to ``mode_id``. Returns ``True`` when the change happened."""
        now = self._clock.monotonic()

        if mode_id not in self._modes:
            self.rejections += 1
            self._reject(mode_id, f"mode '{mode_id.value}' is not available")
            return False

        if mode_id is self._current:
            return False

        if not force:
            if now - self._changed_at < self._min_dwell and self._current is not None:
                self.rejections += 1
                self._reject(mode_id, "mode changed too recently")
                return False
            if safety is not None and safety.severity >= Severity.CRITICAL:
                self.rejections += 1
                self._reject(mode_id, f"safety fault active: {safety.primary_reason}")
                return False
            if (
                safety is not None
                and not safety.ai_allowed
                and self._modes[mode_id].profile.policy.ai_enabled
            ):
                self.rejections += 1
                self._reject(
                    mode_id, f"AI assistance is unavailable: {safety.primary_reason}"
                )
                return False
            if not self._machine.can(mode_id):
                self.rejections += 1
                self._reject(mode_id, "transition is not permitted from the current mode")
                return False

        previous = self._current
        if previous is not None:
            self._modes[previous].on_exit(mode_id)

        self._machine.transition_to(mode_id, reason=reason, force=force)
        self._current = mode_id
        self._changed_at = now
        self.changes += 1

        incoming = self._modes[mode_id]
        incoming.on_enter(previous)
        # Intent timing is part of the mode's feel; apply it here so the mode
        # classes do not each have to remember to.
        self._intent.set_settings(incoming.profile.intent_settings)

        change = ModeChange(
            previous=previous, current=mode_id, reason=reason, timestamp=now, forced=force
        )
        self._history.append(change)
        del self._history[:-32]
        log.info(
            "mode changed",
            previous=previous.value if previous else None,
            current=mode_id.value,
            reason=reason,
            forced=force,
        )
        self._bus.publish(Topics.MODE_CHANGED, change, source=self.name)
        return True

    def _reject(self, mode_id: ModeId, reason: str) -> None:
        log.warning("mode change rejected", requested=mode_id.value, reason=reason)
        self._bus.publish(
            Topics.MODE_REJECTED, {"requested": mode_id.value, "reason": reason}, source=self.name
        )

    def cycle(self, *, safety: SafetyState | None = None) -> bool:
        """Advance to the next mode in the quick-switch order.

        Bound to the double-pulse EMG gesture so a user can change mode without
        reaching the touchscreen — which matters when the hand they would reach
        with is the prosthesis.
        """
        if not self._cycle:
            return False
        try:
            index = self._cycle.index(self._current) if self._current in self._cycle else -1
        except ValueError:  # pragma: no cover - defensive
            index = -1
        target = self._cycle[(index + 1) % len(self._cycle)]
        return self.activate(target, reason="hands-free toggle", safety=safety)

    def fall_back_to_manual(self, reason: str) -> bool:
        """Force Manual Mode. Used when safety withdraws AI permission.

        Forced, because this transition must always succeed: it is the response
        to something already having gone wrong.
        """
        if ModeId.MANUAL not in self._modes or self._current is ModeId.MANUAL:
            return False
        # Remember what the user actually chose, so it can be given back.
        self._preferred = self._current
        log.warning("falling back to manual mode", reason=reason)
        self._bus.publish(
            Topics.UI_NOTIFICATION,
            {"level": "warning", "text": f"Switched to Manual: {reason}", "sticky": True},
            source=self.name,
        )
        return self.activate(ModeId.MANUAL, reason=reason, force=True)

    # -- cycle ----------------------------------------------------------------

    def update(self, context: ModeContext) -> Decision | None:
        """Run the active mode for one cycle, applying safety supervision first."""
        if self._current is None:
            return None

        profile = self._modes[self._current].profile
        if profile.policy.ai_enabled and not context.safety.ai_allowed:
            # The active mode wants AI but safety has withdrawn it. Rather than
            # running a mode whose premise no longer holds, switch to the mode
            # that is honest about what the hand can currently do.
            self.fall_back_to_manual(context.safety.primary_reason or "AI unavailable")
            self._nominal_since = None
        elif self._preferred is not None and context.safety.ai_allowed:
            self._maybe_restore(context.timestamp)

        active = self.active
        return active.update(context) if active is not None else None

    def _maybe_restore(self, now: float) -> None:
        """Return to the user's chosen mode once safety has settled.

        Restoring is announced, and only happens after the condition has been
        clear for a couple of seconds — a hand that flips between modes as a
        marginal fault chatters would be worse than one that stays in Manual.
        """
        if self._nominal_since is None:
            self._nominal_since = now
            return
        if now - self._nominal_since < self._restore_settle_s:
            return
        target, self._preferred = self._preferred, None
        self._nominal_since = None
        if target is None or target is self._current:
            return
        log.info("restoring the previously selected mode", mode=target.value)
        self._bus.publish(
            Topics.UI_NOTIFICATION,
            {"level": "info", "text": f"{target.label} mode restored", "sticky": False},
            source=self.name,
        )
        self.activate(target, reason="fault cleared", force=True)

    # -- reporting ------------------------------------------------------------

    def history(self, limit: int = 10) -> list[ModeChange]:
        return self._history[-limit:]

    def health(self) -> HealthReport:
        if self._current is None:
            return HealthReport.offline(self.name, "no mode active")
        return HealthReport.ok(
            self.name,
            mode=self._current.value,
            changes=self.changes,
            rejections=self.rejections,
            dwell_s=round(self._clock.monotonic() - self._changed_at, 1),
        )

    def to_dot(self) -> str:  # pragma: no cover - documentation helper
        return self._machine.to_dot()
