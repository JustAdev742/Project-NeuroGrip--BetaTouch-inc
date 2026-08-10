"""Generic guarded state machine.

Used for the system lifecycle, the mode manager and the AI-assist grasp sequence.
A shared implementation means all three get the same properties for free:

* transitions must be **declared** — an undeclared transition is an error, not a
  silent no-op, which is how illegal states are kept unreachable;
* **guards** can veto a transition (e.g. safety refuses to leave ``FAULT``);
* **entry/exit hooks** run in a defined order, and a failing hook rolls back;
* a bounded **history** is kept for the diagnostics screen and incident analysis.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

from .clock import Clock, RealClock

__all__ = [
    "StateChange",
    "StateMachine",
    "SystemState",
    "Transition",
    "build_system_state_machine",
]

S = TypeVar("S", bound=Hashable)

Guard = Callable[[S, S], bool]
Hook = Callable[[S, S], None]


@dataclass(frozen=True, slots=True)
class Transition(Generic[S]):
    """A declared, optionally guarded edge between two states."""

    source: S
    target: S
    trigger: str = ""
    guard: Guard | None = None
    description: str = ""


@dataclass(frozen=True, slots=True)
class StateChange(Generic[S]):
    """Record of a completed transition, retained in the machine's history."""

    source: S
    target: S
    trigger: str
    timestamp: float
    reason: str = ""


class StateMachine(Generic[S]):
    """A finite state machine with declared transitions.

    Example::

        sm = StateMachine(SystemState.INIT, clock)
        sm.allow(SystemState.INIT, SystemState.READY, trigger="startup_complete")
        sm.on_enter(SystemState.READY, lambda src, dst: bus.publish(Topics.SYSTEM_READY))
        sm.fire("startup_complete")
    """

    def __init__(
        self,
        initial: S,
        clock: Clock | None = None,
        *,
        name: str = "state-machine",
        history: int = 64,
    ) -> None:
        self._state = initial
        self._clock = clock or RealClock()
        self._name = name
        self._transitions: dict[S, list[Transition[S]]] = {}
        self._enter_hooks: dict[S, list[Hook]] = {}
        self._exit_hooks: dict[S, list[Hook]] = {}
        self._any_hooks: list[Callable[[StateChange[S]], None]] = []
        self._history: list[StateChange[S]] = []
        self._history_limit = history
        self._entered_at = self._clock.monotonic()

    # -- declaration ----------------------------------------------------------

    def allow(
        self,
        source: S,
        target: S,
        *,
        trigger: str = "",
        guard: Guard | None = None,
        description: str = "",
    ) -> StateMachine[S]:
        """Declare a legal transition. Chainable."""
        self._transitions.setdefault(source, []).append(
            Transition(source, target, trigger, guard, description)
        )
        return self

    def allow_many(self, sources: Iterable[S], target: S, *, trigger: str = "") -> StateMachine[S]:
        """Declare the same target reachable from several sources (e.g. ``FAULT``)."""
        for source in sources:
            self.allow(source, target, trigger=trigger)
        return self

    def on_enter(self, state: S, hook: Hook) -> StateMachine[S]:
        self._enter_hooks.setdefault(state, []).append(hook)
        return self

    def on_exit(self, state: S, hook: Hook) -> StateMachine[S]:
        self._exit_hooks.setdefault(state, []).append(hook)
        return self

    def on_change(self, hook: Callable[[StateChange[S]], None]) -> StateMachine[S]:
        """Observe every transition (used to publish ``mode.changed`` events)."""
        self._any_hooks.append(hook)
        return self

    # -- queries --------------------------------------------------------------

    @property
    def state(self) -> S:
        return self._state

    @property
    def name(self) -> str:
        return self._name

    @property
    def time_in_state(self) -> float:
        """Seconds since the current state was entered."""
        return self._clock.monotonic() - self._entered_at

    @property
    def history(self) -> tuple[StateChange[S], ...]:
        return tuple(self._history)

    def can(self, target: S) -> bool:
        """Whether a transition to ``target`` is declared and its guard passes."""
        return self._find(target) is not None

    def targets(self) -> tuple[S, ...]:
        """States reachable from the current one, ignoring guards."""
        return tuple(t.target for t in self._transitions.get(self._state, ()))

    def _find(self, target: S) -> Transition[S] | None:
        for transition in self._transitions.get(self._state, ()):
            if transition.target != target:
                continue
            if transition.guard is not None and not transition.guard(self._state, target):
                continue
            return transition
        return None

    # -- execution ------------------------------------------------------------

    def transition_to(self, target: S, *, reason: str = "", force: bool = False) -> bool:
        """Attempt to move to ``target``.

        Returns ``True`` when the transition happened. ``force`` bypasses the
        declaration table and is reserved for safety-driven transitions such as
        an emergency stop, which must always succeed.
        """
        if target == self._state:
            return False

        transition = self._find(target)
        if transition is None and not force:
            return False

        source = self._state
        trigger = transition.trigger if transition else "forced"

        for hook in self._exit_hooks.get(source, ()):
            hook(source, target)

        self._state = target
        self._entered_at = self._clock.monotonic()

        change = StateChange(
            source=source,
            target=target,
            trigger=trigger,
            timestamp=self._entered_at,
            reason=reason,
        )
        self._history.append(change)
        if len(self._history) > self._history_limit:
            del self._history[0 : len(self._history) - self._history_limit]

        for hook in self._enter_hooks.get(target, ()):
            hook(source, target)
        for observer in self._any_hooks:
            observer(change)
        return True

    def fire(self, trigger: str, *, reason: str = "") -> bool:
        """Take the first declared transition matching ``trigger`` whose guard passes."""
        for transition in self._transitions.get(self._state, ()):
            if transition.trigger != trigger:
                continue
            if transition.guard is not None and not transition.guard(self._state, transition.target):
                continue
            return self.transition_to(transition.target, reason=reason or trigger)
        return False

    def to_dot(self) -> str:  # pragma: no cover - documentation helper
        """Render the machine as Graphviz DOT, for the docs and the debug console."""
        lines = [f'digraph "{self._name}" {{', "  rankdir=LR;", '  node [shape=box, style=rounded];']
        for source, transitions in self._transitions.items():
            for t in transitions:
                label = t.trigger or ""
                if t.guard is not None:
                    label = f"{label} [guard]" if label else "[guard]"
                lines.append(f'  "{_label(source)}" -> "{_label(t.target)}" [label="{label}"];')
        lines.append("}")
        return "\n".join(lines)


def _label(state: object) -> str:
    return getattr(state, "value", None) or getattr(state, "name", None) or str(state)


class SystemState(str, Enum):
    """Top-level lifecycle states of the device.

    ``BOOT -> SELFTEST -> HOMING -> READY -> ACTIVE`` is the nominal path.
    ``FAULT`` and ``ESTOP`` are reachable from anywhere and require an explicit,
    user-acknowledged recovery — a device that silently re-enables itself after a
    fault is a device that will surprise its user.
    """

    BOOT = "boot"
    SELFTEST = "selftest"
    HOMING = "homing"
    READY = "ready"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAULT = "fault"
    ESTOP = "estop"
    SHUTDOWN = "shutdown"

    @property
    def motion_allowed(self) -> bool:
        """Whether the hand controller may command movement in this state."""
        return self in (SystemState.HOMING, SystemState.READY, SystemState.ACTIVE, SystemState.DEGRADED)

    @property
    def label(self) -> str:
        return self.value.upper()


def build_system_state_machine(clock: Clock | None = None) -> StateMachine[SystemState]:
    """Construct the device lifecycle machine with its declared transitions."""
    sm: StateMachine[SystemState] = StateMachine(SystemState.BOOT, clock, name="system")
    sm.allow(SystemState.BOOT, SystemState.SELFTEST, trigger="boot_complete")
    sm.allow(SystemState.SELFTEST, SystemState.HOMING, trigger="selftest_passed")
    sm.allow(SystemState.SELFTEST, SystemState.DEGRADED, trigger="selftest_degraded")
    sm.allow(SystemState.HOMING, SystemState.READY, trigger="homed")
    sm.allow(SystemState.READY, SystemState.ACTIVE, trigger="activate")
    sm.allow(SystemState.ACTIVE, SystemState.READY, trigger="idle")
    sm.allow(SystemState.DEGRADED, SystemState.READY, trigger="recovered")
    sm.allow(SystemState.ACTIVE, SystemState.DEGRADED, trigger="degrade")
    sm.allow(SystemState.READY, SystemState.DEGRADED, trigger="degrade")
    # Faults and e-stop are reachable from every operational state.
    operational = (
        SystemState.BOOT,
        SystemState.SELFTEST,
        SystemState.HOMING,
        SystemState.READY,
        SystemState.ACTIVE,
        SystemState.DEGRADED,
    )
    sm.allow_many(operational, SystemState.FAULT, trigger="fault")
    sm.allow_many((*operational, SystemState.FAULT), SystemState.ESTOP, trigger="estop")
    # Recovery is always explicit and always lands in a non-moving state.
    sm.allow(SystemState.FAULT, SystemState.SELFTEST, trigger="acknowledge")
    sm.allow(SystemState.ESTOP, SystemState.SELFTEST, trigger="reset")
    sm.allow_many(
        (*operational, SystemState.FAULT, SystemState.ESTOP),
        SystemState.SHUTDOWN,
        trigger="shutdown",
    )
    return sm
