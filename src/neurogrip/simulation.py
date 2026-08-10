"""Simulation harness: scenarios and a scripted world.

Turns "the user reaches for a bottle, contracts, and the hand grasps it" into a
*runnable, deterministic* artefact. That matters more than it might sound: it is
what allows the safety properties in ``docs/safety.md`` to be tested rather than
asserted.

Two pieces:

* :class:`SimulatedWorld` — drives the synthetic EMG source, the synthetic
  camera's scene, and the contact model of the simulated actuators, so the plant
  and the perception stay consistent with each other. When the scene says the
  bottle is 7 cm wide, the fingers actually stop at the closure that corresponds
  to 7 cm.
* :class:`Scenario` — a timeline of events (set the object, contract, release,
  inject a fault) that the runner plays against the real application under a
  :class:`~neurogrip.core.clock.SimulatedClock`, faster than real time.

Everything above the HAL is the production code path: real filters, real
classifier, real fusion gates, real controller, real safety rules.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .core.clock import SimulatedClock
from .core.logging import get_logger
from .core.types import clamp
from .hal.camera.simulated import SceneObject, SimulatedCamera
from .hal.emg.simulated import SimulatedEmgSource
from .hal.servo.simulated import ContactModel, SimulatedServoBus
from .runtime.application import Application

__all__ = ["DEMO_SCENARIOS", "Scenario", "ScenarioRunner", "ScenarioStep", "SimulatedWorld"]

log = get_logger(__name__)


@dataclass(slots=True)
class SimulatedWorld:
    """Keeps the simulated sensors and the simulated plant consistent.

    The important detail is :meth:`_apply_contact`: an object's *physical* width
    is converted into the finger closure at which contact occurs, so the grip the
    vision system describes and the grip the fingers experience are the same
    object. Without that coupling, a simulation proves nothing about grasping.
    """

    emg: SimulatedEmgSource | None = None
    camera: SimulatedCamera | None = None
    plant: SimulatedServoBus | None = None
    #: Aperture of the fully open hand, in metres. Matches HandKinematics.
    max_aperture_m: float = 0.11
    #: The object currently in front of the hand, or ``None`` for an empty scene.
    object_present: SceneObject | None = None
    object_width_m: float = 0.07
    object_stiffness: float = 1.0

    # -- object -----------------------------------------------------------

    def place_object(
        self,
        label: str,
        *,
        width_m: float = 0.07,
        shape: str = "cylinder",
        distance_m: float = 0.32,
        stiffness: float = 1.0,
        visible: bool = True,
    ) -> None:
        """Put an object in front of the hand, updating both camera and plant."""
        # Apparent size in the image, from the physical width and distance
        # (small-angle pinhole approximation, adequate at these distances).
        apparent = clamp(width_m / max(0.05, distance_m) * 1.2, 0.05, 0.9)
        aspect = 2.4 if shape == "cylinder" else (1.0 if shape == "sphere" else 0.5)
        scene = SceneObject(
            label=label,
            width=apparent,
            height=clamp(apparent * aspect, 0.05, 0.95),
            distance_m=distance_m,
            shape=shape,
            visible=visible,
        )
        self.object_present = scene
        self.object_width_m = width_m
        self.object_stiffness = stiffness
        if self.camera is not None:
            self.camera.set_scene(scene)
        self._apply_contact()

    def clear_object(self) -> None:
        """Remove the object: empty scene, fingers free to close fully."""
        self.object_present = None
        if self.camera is not None:
            self.camera.set_scene(SceneObject(visible=False))
        if self.plant is not None:
            self.plant.set_contact(ContactModel.free())

    def _apply_contact(self) -> None:
        if self.plant is None or self.object_present is None:
            return
        # Closure at which the fingers meet the object's surface.
        closure = clamp(1.0 - self.object_width_m / self.max_aperture_m)
        self.plant.set_contact(
            ContactModel.uniform(closure, stiffness=self.object_stiffness)
        )

    # -- user -------------------------------------------------------------

    def contract(self, level: float = 0.7) -> None:
        """The user flexes (asking to close)."""
        if self.emg is not None:
            self.emg.set_flexor(clamp(level))
            self.emg.set_extensor(0.0)

    def extend(self, level: float = 0.7) -> None:
        """The user extends (asking to open)."""
        if self.emg is not None:
            self.emg.set_flexor(0.0)
            self.emg.set_extensor(clamp(level))

    def co_contract(self, level: float = 0.6) -> None:
        """The user co-contracts — the cancel gesture."""
        if self.emg is not None:
            self.emg.set_flexor(clamp(level))
            self.emg.set_extensor(clamp(level))

    def relax(self) -> None:
        if self.emg is not None:
            self.emg.set_flexor(0.0)
            self.emg.set_extensor(0.0)

    # -- faults -----------------------------------------------------------

    def detach_electrode(self, channel: int = 0) -> None:
        """Simulate an electrode coming loose."""
        if self.emg is not None:
            self.emg.set_contact_quality(channel, 0.05)

    def reattach_electrode(self, channel: int = 0) -> None:
        if self.emg is not None:
            self.emg.set_contact_quality(channel, 1.0)

    def bump_electrode(self, channel: int = 0) -> None:
        """Inject a motion artefact."""
        if self.emg is not None:
            self.emg.inject_motion_artefact(channel)

    @classmethod
    def from_application(cls, application: Application) -> SimulatedWorld:
        """Extract the simulated devices from a built application."""
        hardware = application.hardware
        emg = hardware.emg_source if isinstance(hardware.emg_source, SimulatedEmgSource) else None
        camera = hardware.camera if isinstance(hardware.camera, SimulatedCamera) else None
        plant = hardware.simulated_plant
        max_aperture = application.controller.kinematics.max_aperture_m
        return cls(emg=emg, camera=camera, plant=plant, max_aperture_m=max_aperture)


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    """One timed action in a scenario."""

    #: Seconds from the start of the scenario.
    at: float
    #: What to do. Receives the world and the application.
    action: Callable[[SimulatedWorld, Application], None]
    description: str = ""


@dataclass(frozen=True, slots=True)
class Scenario:
    """A named timeline."""

    name: str
    description: str
    steps: tuple[ScenarioStep, ...]
    duration_s: float = 10.0
    #: Assertions evaluated at the end; each returns ``(passed, message)``.
    checks: tuple[Callable[[SimulatedWorld, Application], tuple[bool, str]], ...] = field(
        default_factory=tuple
    )


@dataclass(slots=True)
class ScenarioResult:
    """Outcome of running a scenario."""

    scenario: str
    duration_s: float
    steps_executed: int
    checks: tuple[tuple[bool, str], ...] = ()
    #: Sampled timeline, for plotting and for post-hoc inspection.
    timeline: tuple[dict, ...] = ()

    @property
    def passed(self) -> bool:
        return all(ok for ok, _ in self.checks)

    def report(self) -> str:
        lines = [f"{self.scenario}: {'PASS' if self.passed else 'FAIL'} ({self.duration_s:.1f} s)"]
        lines.extend(f"  {'✓' if ok else '✗'} {message}" for ok, message in self.checks)
        return "\n".join(lines)


class ScenarioRunner:
    """Plays scenarios against a real application under a simulated clock."""

    def __init__(self, application: Application, clock: SimulatedClock) -> None:
        if not isinstance(clock, SimulatedClock):
            raise TypeError("scenarios require a SimulatedClock for deterministic timing")
        self._application = application
        self._clock = clock
        self._world = SimulatedWorld.from_application(application)

    @property
    def world(self) -> SimulatedWorld:
        return self._world

    def run(self, scenario: Scenario, *, step_s: float = 0.005, sample_hz: float = 20.0) -> ScenarioResult:
        """Run one scenario to completion."""
        log.info("running scenario", scenario=scenario.name)
        start = self._clock.monotonic()
        pending = list(scenario.steps)
        executed = 0
        timeline: list[dict] = []
        next_sample = start
        sample_interval = 1.0 / sample_hz

        while self._clock.monotonic() - start < scenario.duration_s:
            elapsed = self._clock.monotonic() - start

            while pending and pending[0].at <= elapsed:
                step = pending.pop(0)
                log.debug("scenario step", at=round(elapsed, 3), detail=step.description)
                step.action(self._world, self._application)
                executed += 1

            self._application.scheduler.step()

            if self._clock.monotonic() >= next_sample:
                next_sample += sample_interval
                timeline.append(self._sample(elapsed))

            self._clock.advance(step_s)

        checks = tuple(check(self._world, self._application) for check in scenario.checks)
        result = ScenarioResult(
            scenario=scenario.name,
            duration_s=self._clock.monotonic() - start,
            steps_executed=executed,
            checks=checks,
            timeline=tuple(timeline),
        )
        log.info("scenario complete", scenario=scenario.name, passed=result.passed)
        return result

    def _sample(self, elapsed: float) -> dict:
        hand = self._application.controller.state
        intent = self._application.emg.intent
        mode = self._application.modes.active
        decision = mode.last_decision if mode else None
        return {
            "t": round(elapsed, 3),
            "pose": [round(v, 3) for v in hand.pose],
            "holding": hand.holding,
            "force": round(hand.force, 3),
            "intent": intent.kind.value if intent else "none",
            "intent_confidence": round(intent.confidence, 3) if intent else 0.0,
            "action": decision.action.value if decision else "none",
            "ai": bool(decision and decision.ai_contributed),
            "grasp": decision.plan.grasp.value if decision and decision.plan else "",
        }


# ---------------------------------------------------------------------------
# Bundled scenarios
# ---------------------------------------------------------------------------


def _grasp_bottle() -> Scenario:
    def place(world: SimulatedWorld, app: Application) -> None:
        world.place_object("bottle", width_m=0.068, shape="cylinder", distance_m=0.30)

    def contract(world: SimulatedWorld, app: Application) -> None:
        world.contract(0.75)

    def hold(world: SimulatedWorld, app: Application) -> None:
        world.contract(0.5)

    def check_holding(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        hand = app.controller.state
        contacts = hand.grip.contact_count if hand.grip else 0
        return (hand.holding, f"hand is holding the object (contacts: {contacts})")

    def check_ai(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        mode = app.modes.active
        decision = mode.last_decision if mode else None
        used_ai = bool(decision and decision.plan is not None)
        grasp = decision.plan.grasp.value if decision and decision.plan else "none"
        return (used_ai, f"AI selected a grasp ({grasp})")

    def check_force(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        hand = app.controller.state
        return (hand.force <= 0.85, f"grip force {hand.force:.2f} within the ceiling")

    return Scenario(
        name="grasp-bottle",
        description="User points at a bottle and contracts; the AI selects a cylindrical grasp.",
        steps=(
            ScenarioStep(0.5, place, "bottle enters view"),
            ScenarioStep(2.0, contract, "user contracts to grasp"),
            ScenarioStep(4.0, hold, "user maintains a lighter hold"),
        ),
        duration_s=8.0,
        checks=(check_holding, check_ai, check_force),
    )


def _no_intent_no_motion() -> Scenario:
    """The core safety property: perception alone must never move the hand."""

    def place(world: SimulatedWorld, app: Application) -> None:
        world.place_object("bottle", width_m=0.068)

    def relax(world: SimulatedWorld, app: Application) -> None:
        world.relax()

    def check_still(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        hand = app.controller.state
        moved = hand.pose.max_difference(hand.pose.open_hand())
        return (moved < 0.05, f"hand did not move without user intent (max travel {moved:.3f})")

    def check_saw_object(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        vision = app.vision.latest if app.vision else None
        seen = bool(vision and vision.primary)
        return (seen, "the camera did see the object (so inaction was a decision, not blindness)")

    return Scenario(
        name="no-intent-no-motion",
        description="An object is clearly visible but the user never contracts. The hand must not move.",
        steps=(
            ScenarioStep(0.5, place, "bottle enters view"),
            ScenarioStep(1.0, relax, "user stays relaxed"),
        ),
        duration_s=6.0,
        checks=(check_still, check_saw_object),
    )


def _user_cancel() -> Scenario:
    def place(world: SimulatedWorld, app: Application) -> None:
        world.place_object("bottle", width_m=0.068)

    def contract(world: SimulatedWorld, app: Application) -> None:
        world.contract(0.8)

    def cancel(world: SimulatedWorld, app: Application) -> None:
        world.co_contract(0.7)

    def relax(world: SimulatedWorld, app: Application) -> None:
        world.relax()

    def check_cancelled(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        return (app.fusion.cancels > 0, f"cancel was registered ({app.fusion.cancels} times)")

    def check_stopped(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        hand = app.controller.state
        return (not hand.moving, "motion stopped after the cancel")

    return Scenario(
        name="user-cancel",
        description="User starts a grasp, then co-contracts to abort mid-motion.",
        steps=(
            ScenarioStep(0.5, place, "bottle enters view"),
            ScenarioStep(1.5, contract, "user starts the grasp"),
            ScenarioStep(2.2, cancel, "user co-contracts to cancel"),
            ScenarioStep(3.0, relax, "user relaxes"),
        ),
        duration_s=6.0,
        checks=(check_cancelled, check_stopped),
    )


def _vision_lost() -> Scenario:
    """Losing the camera must degrade assistance, never the hand."""

    def contract(world: SimulatedWorld, app: Application) -> None:
        world.contract(0.8)

    def clear(world: SimulatedWorld, app: Application) -> None:
        world.clear_object()

    def check_still_moves(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        hand = app.controller.state
        closed = max(hand.pose)
        return (closed > 0.25, f"the hand still closes under direct control ({closed:.2f})")

    return Scenario(
        name="vision-lost",
        description=(
            "No object is recognised, but the user still wants to grasp. "
            "Direct control must continue."
        ),
        steps=(
            ScenarioStep(0.3, clear, "empty scene"),
            ScenarioStep(1.5, contract, "user contracts anyway"),
        ),
        duration_s=6.0,
        checks=(check_still_moves,),
    )


def _fragile_object() -> Scenario:
    def place(world: SimulatedWorld, app: Application) -> None:
        world.place_object("fruit", width_m=0.072, shape="sphere", stiffness=0.4)

    def contract(world: SimulatedWorld, app: Application) -> None:
        world.contract(0.9)

    def check_gentle(world: SimulatedWorld, app: Application) -> tuple[bool, str]:
        hand = app.controller.state
        return (hand.force <= 0.45, f"force limited for a fragile object ({hand.force:.2f} ≤ 0.45)")

    return Scenario(
        name="fragile-object",
        description="A hard contraction on a fragile object must still produce a gentle grip.",
        steps=(
            ScenarioStep(0.5, place, "fruit enters view"),
            ScenarioStep(2.0, contract, "user contracts hard"),
        ),
        duration_s=7.0,
        checks=(check_gentle,),
    )


#: Scenarios available from ``neurogrip simulate``.
DEMO_SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "grasp-bottle": _grasp_bottle,
    "no-intent-no-motion": _no_intent_no_motion,
    "user-cancel": _user_cancel,
    "vision-lost": _vision_lost,
    "fragile-object": _fragile_object,
}


def build_scenario(name: str) -> Scenario:
    """Look up a bundled scenario by name."""
    factory = DEMO_SCENARIOS.get(name)
    if factory is None:
        raise KeyError(f"unknown scenario '{name}'; available: {', '.join(DEMO_SCENARIOS)}")
    return factory()


def all_scenarios() -> Sequence[Scenario]:
    return [factory() for factory in DEMO_SCENARIOS.values()]
