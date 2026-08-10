"""Operating-mode interface.

A mode is a *policy bundle*, not a separate control implementation. Every mode
runs the same pipeline — EMG → intent → fusion → controller — and differs only in
the parameters it supplies:

* the :class:`~neurogrip.fusion.policy.FusionPolicy` (confidence thresholds,
  whether AI may contribute at all);
* the :class:`~neurogrip.control.motion.MotionLimits` and force ceiling;
* intent timing (dwell, release windows);
* UI presentation flags.

Writing modes this way means a new mode cannot introduce a new way to move the
hand, and therefore cannot introduce a new way to move it unsafely. It also means
the safety and fusion code has no ``if mode == ...`` branches to get out of date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..control.controller import HandController, HandState
from ..control.motion import MotionLimits
from ..core.clock import Clock
from ..core.events import EventBus
from ..core.types import ModeId
from ..emg.intent import IntentEstimate, IntentSettings
from ..emg.pipeline import EmgFrame
from ..fusion.fusion import Decision, DecisionAction, DecisionFusion, FusionInputs
from ..fusion.policy import FusionPolicy
from ..safety.monitor import SafetyState
from ..vision.types import VisionResult

__all__ = ["ModeBase", "ModeContext", "ModeProfile", "OperatingMode"]


@dataclass(frozen=True, slots=True)
class ModeContext:
    """Everything a mode sees on one update.

    A single immutable snapshot rather than live references, so a mode cannot
    observe state changing underneath it mid-decision.
    """

    timestamp: float
    hand: HandState
    intent: IntentEstimate | None
    emg: EmgFrame | None
    vision: VisionResult | None
    safety: SafetyState
    #: Seconds since the previous update of this mode.
    dt: float = 0.0


@dataclass(frozen=True, slots=True)
class ModeProfile:
    """The parameter bundle that *is* the mode."""

    mode: ModeId
    policy: FusionPolicy
    motion_limits: MotionLimits
    intent_settings: IntentSettings
    #: Displayed in the UI header.
    title: str = ""
    subtitle: str = ""
    #: Drives the prominent "AI DISABLED" banner required in Manual Mode.
    show_ai_disabled_banner: bool = False
    #: Accent colour key resolved by the theme.
    accent: str = "primary"
    #: Whether this mode is offered in the quick-switch carousel.
    user_selectable: bool = True
    #: Enable jerk limiting; Sports Mode disables it for responsiveness.
    s_curve: bool = True
    #: Vision processing rate for this mode, Hz. ``0`` disables vision entirely.
    vision_rate_hz: float = 20.0
    #: Control-loop rate for this mode, Hz.
    control_rate_hz: float = 200.0
    notes: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class OperatingMode(Protocol):
    """A selectable operating mode."""

    @property
    def id(self) -> ModeId: ...

    @property
    def profile(self) -> ModeProfile: ...

    def on_enter(self, previous: ModeId | None) -> None:
        """Called when this mode becomes active."""
        ...

    def on_exit(self, following: ModeId | None) -> None:
        """Called when leaving. Must leave the hand in a safe, stable state."""
        ...

    def update(self, context: ModeContext) -> Decision | None:
        """Run one cycle; returns the decision taken, or ``None`` if idle."""
        ...


class ModeBase:
    """Shared implementation for the bundled modes.

    Subclasses supply a :class:`ModeProfile` and, if needed, override
    :meth:`on_enter` / :meth:`on_exit` / :meth:`execute`. The default
    :meth:`update` runs fusion and hands the decision to
    :meth:`execute`, which every mode inherits — so the "decide then execute"
    split is uniform.
    """

    def __init__(
        self,
        profile: ModeProfile,
        controller: HandController,
        fusion: DecisionFusion,
        clock: Clock,
        bus: EventBus,
    ) -> None:
        self._profile = profile
        self._controller = controller
        self._fusion = fusion
        self._clock = clock
        self._bus = bus
        self._entered_at = 0.0
        self._last_decision: Decision | None = None
        #: Diagnostics counters.
        self.updates = 0

    # -- identity -------------------------------------------------------------

    @property
    def id(self) -> ModeId:
        return self._profile.mode

    @property
    def profile(self) -> ModeProfile:
        return self._profile

    @property
    def last_decision(self) -> Decision | None:
        return self._last_decision

    @property
    def active_duration(self) -> float:
        return self._clock.monotonic() - self._entered_at

    # -- lifecycle ------------------------------------------------------------

    def on_enter(self, previous: ModeId | None) -> None:
        """Apply this mode's envelope to the controller and fusion layer."""
        self._entered_at = self._clock.monotonic()
        self._last_decision = None
        self._fusion.set_policy(self._profile.policy)
        self._fusion.reset()
        self._controller.configure(
            motion_limits=self._profile.motion_limits,
            force_ceiling=self._profile.policy.force_ceiling,
            speed_scale=self._profile.policy.speed_ceiling,
            s_curve=self._profile.s_curve,
        )

    def on_exit(self, following: ModeId | None) -> None:
        """Stop any motion this mode started.

        Position is *held*, not opened: switching mode while carrying something
        must not drop it.
        """
        self._controller.cancel(f"leaving {self.id.value} mode")

    # -- cycle ----------------------------------------------------------------

    def update(self, context: ModeContext) -> Decision | None:
        self.updates += 1
        decision = self._fusion.evaluate(
            FusionInputs(
                intent=context.intent,
                vision=context.vision if self._profile.vision_rate_hz > 0 else None,
                current_pose=context.hand.pose,
                mode=self.id,
                timestamp=context.timestamp,
                motion_allowed=context.safety.motion_allowed,
                ai_allowed=context.safety.ai_allowed,
                safety_reason=context.safety.primary_reason,
                holding=context.hand.holding,
                safety_force_ceiling=context.safety.force_ceiling,
            )
        )
        self._last_decision = decision
        self.execute(decision, context)
        return decision

    def execute(self, decision: Decision, context: ModeContext) -> None:
        """Turn a decision into motion commands.

        Shared by every mode. The differences between modes are already encoded
        in the decision (via the policy), so there is nothing mode-specific left
        to do here — which is exactly the property we want.
        """
        from ..control.queue import MotionCommand, Priority

        controller = self._controller

        if decision.action is DecisionAction.CANCEL:
            controller.cancel("user cancelled")
            return

        if decision.action in (DecisionAction.IDLE, DecisionAction.BLOCKED):
            return

        if decision.action is DecisionAction.RELEASE:
            controller.submit(
                MotionCommand(
                    target=decision.target or context.hand.pose,
                    priority=Priority.USER_OVERRIDE,
                    force=decision.force,
                    speed=decision.speed,
                    source=f"mode:{self.id.value}",
                    description="Open",
                    issued_at=context.timestamp,
                )
            )
            return

        if decision.action is DecisionAction.ASSISTED and decision.plan is not None:
            plan = decision.plan
            controller.submit(
                MotionCommand(
                    target=plan.target,
                    preshape=plan.preshape,
                    priority=Priority.ASSISTED,
                    force=decision.force,
                    speed=decision.speed,
                    source=f"mode:{self.id.value}",
                    description=plan.grasp.label,
                    issued_at=context.timestamp,
                    tag=plan.source.value,
                )
            )
            return

        if decision.action is DecisionAction.DIRECT and decision.target is not None:
            controller.submit(
                MotionCommand(
                    target=decision.target,
                    priority=Priority.USER_DIRECT,
                    force=decision.force,
                    speed=decision.speed,
                    source=f"mode:{self.id.value}",
                    description="Direct control",
                    issued_at=context.timestamp,
                    # Direct control is continuously retargeted from EMG, so a
                    # long timeout would let a stale setpoint linger.
                    timeout_s=2.0,
                )
            )
