"""Servo calibration: data model, persistence and the guided bring-up wizard.

A tendon-driven finger is not a servo. Between the horn and the fingertip there
is a length of fishing line whose effective length changes — it is cut by hand
during assembly, it stretches under load, and it creeps over weeks of use. Two
consequences drive everything in this module:

* **Slack must be measured, not assumed.** Some fraction of the servo's travel
  is consumed taking up line slack before the finger moves at all. If the host
  assumes zero slack, the first 20% of every commanded motion does nothing and
  the finger arrives late; if it assumes too much, the finger starts moving
  before the controller expects it to.
* **The usable end of travel must be found, not commanded.** Driving a finger
  to a closure the tendon cannot reach stalls the servo against a hard stop.
  Held there it will strip the horn, snap the line, or cook the motor.

What the wizard *can* determine is exactly what the host can observe: at what
commanded closure does the finger start to load (slack), and at what commanded
closure does it stop making progress (end of travel). Pulse-width endpoints and
inversion are **not** discovered here — they are a property of how the hand was
built, they live in ``config/hardware.toml``, and the wizard's job is to verify
them rather than guess them. Discovering them would need raw-pulse jogging,
which would mean the host could command a pulse the firmware has not
range-checked; that trade is not worth making for a value that changes once, at
build time.

The wizard drives the hand through :class:`~neurogrip.control.controller.HandController`,
never through the servo bus directly, so the "one writer to the actuators"
invariant holds during calibration exactly as it does in normal operation.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from ..core.clock import Clock
from ..core.errors import CalibrationError
from ..core.logging import get_logger
from ..core.types import FINGER_COUNT, Finger, HandPose, clamp
from ..hal.servo.base import ServoCalibration
from .controller import HandController
from .queue import Priority

__all__ = [
    "FingerCalibrationResult",
    "ServoCalibrationPhase",
    "ServoCalibrationProgress",
    "ServoCalibrationSet",
    "ServoCalibrationWizard",
]

log = get_logger(__name__)

#: Force ceiling used throughout calibration. Low enough that a finger driven
#: into a hard stop stalls harmlessly rather than damaging the mechanism.
CALIBRATION_FORCE = 0.18

#: Speed scale during the creep phases. Slow motion is what makes the current
#: rise attributable to tendon load rather than to acceleration.
CALIBRATION_SPEED = 0.22

#: A finger whose measured slack exceeds this has a tendon that is too long to
#: calibrate around; it needs re-stringing, not a software correction.
MAX_ACCEPTABLE_SLACK = 0.55

#: A finger that stops making progress below this closure binds too early —
#: the tendon is too short or the routing is fouled.
MIN_ACCEPTABLE_TRAVEL = 0.80


@dataclass(frozen=True, slots=True)
class FingerCalibrationResult:
    """What the wizard measured for one finger."""

    finger: Finger
    #: Commanded closure at which the tendon went taut.
    slack: float
    #: Commanded closure beyond which the finger stopped making progress.
    travel_end: float
    #: Motor current with the tendon slack, mA — the no-load baseline.
    baseline_current_ma: int
    #: Motor current at the end of travel, mA.
    loaded_current_ma: int
    #: Measured position change across the whole sweep.
    observed_travel: float
    problems: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.problems

    def describe(self) -> str:
        status = "ok" if self.ok else "; ".join(self.problems)
        return (
            f"{self.finger.name.lower()}: slack {self.slack:.2f}, "
            f"travel to {self.travel_end:.2f}, "
            f"{self.baseline_current_ma}→{self.loaded_current_ma} mA — {status}"
        )


@dataclass(slots=True)
class ServoCalibrationSet:
    """Calibration for all five fingers, plus provenance.

    Persisted separately from the EMG calibration because the two have different
    lifetimes: EMG calibration is per user and per session, servo calibration is
    per physical hand and survives users.
    """

    fingers: dict[int, ServoCalibration] = field(default_factory=dict)
    hand_id: str = "default"
    created_at: float = 0.0
    updated_at: float = 0.0
    notes: str = ""
    version: int = 1

    def get(self, finger: Finger) -> ServoCalibration:
        """Calibration for ``finger``, or a conservative zero-slack default.

        The fallback assumes *no* slack. That is the safe direction to be wrong
        in: the finger under-travels and grips weakly, rather than over-driving
        into a stop that the calibration was supposed to keep it away from.
        """
        found = self.fingers.get(int(finger))
        if found is not None:
            return found
        return ServoCalibration(finger=finger)

    def set(self, calibration: ServoCalibration) -> None:
        self.fingers[int(calibration.finger)] = calibration

    @property
    def is_complete(self) -> bool:
        return len(self.fingers) == FINGER_COUNT

    @property
    def age_days(self) -> float:
        """Days since calibration. Fishing line creeps; the UI nags after a month."""
        if not self.updated_at:
            return float("inf")
        return (time.time() - self.updated_at) / 86400.0

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "hand_id": self.hand_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
            "fingers": [
                {**asdict(c), "finger": int(c.finger)}
                for c in sorted(self.fingers.values(), key=lambda c: int(c.finger))
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ServoCalibrationSet:
        result = cls(
            hand_id=data.get("hand_id", "default"),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            notes=data.get("notes", ""),
            version=int(data.get("version", 1)),
        )
        for entry in data.get("fingers", []):
            fields = dict(entry)
            fields["finger"] = Finger(int(fields["finger"]))
            result.set(ServoCalibration(**fields))
        return result

    def save(self, path: Path | str) -> None:
        """Write atomically — a half-written calibration must never load."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(target)
        log.info("servo calibration saved", path=str(target), fingers=len(self.fingers))

    @classmethod
    def load(cls, path: Path | str) -> ServoCalibrationSet:
        file_path = Path(path)
        if not file_path.exists():
            raise CalibrationError(f"servo calibration file not found: {file_path}")
        try:
            return cls.from_dict(json.loads(file_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid servo calibration file: {exc}") from exc

    @classmethod
    def from_config(cls, config) -> ServoCalibrationSet:
        """Build endpoints from ``[servo.fingers.<name>]`` in the hardware config.

        This is the *build spec* — pulse endpoints and inversion as wired — with
        zero slack. The wizard then measures slack on top of it.
        """
        result = cls(hand_id=config.get_str("servo.hand_id", "default"))
        sections = config.sections("servo.fingers")
        for finger in Finger:
            section = sections.get(finger.name.lower())
            if section is None:
                result.set(ServoCalibration(finger=finger))
                continue
            result.set(
                ServoCalibration(
                    finger=finger,
                    min_pulse_us=section.get_int("min_pulse_us", 1000),
                    max_pulse_us=section.get_int("max_pulse_us", 2000),
                    inverted=section.get_bool("inverted", False),
                    slack=section.get_float("slack", 0.0),
                )
            )
        return result


class ServoCalibrationPhase(str, Enum):
    """Wizard phases, in execution order."""

    IDLE = "idle"
    #: Open the hand and let it settle, establishing the no-load current.
    BASELINE = "baseline"
    #: Creep the finger closed, watching for the current rise that means taut.
    TAKE_UP = "take_up"
    #: Continue closing until the finger stops making progress.
    TRAVEL = "travel"
    #: Return the finger to open before moving on to the next one.
    RELEASE = "release"
    #: All fingers measured; validating the result.
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ServoCalibrationProgress:
    """Live wizard state for the UI and the CLI to render."""

    phase: ServoCalibrationPhase
    finger: Finger | None
    finger_index: int
    finger_count: int
    instruction: str
    title: str
    #: Commanded closure right now, for the progress bar.
    commanded: float = 0.0
    measured: float = 0.0
    current_ma: int = 0
    message: str = ""
    results: tuple[FingerCalibrationResult, ...] = field(default_factory=tuple)

    @property
    def fraction(self) -> float:
        if self.finger_count <= 0:
            return 0.0
        return clamp((self.finger_index + self.commanded) / self.finger_count)

    @property
    def finished(self) -> bool:
        return self.phase in (ServoCalibrationPhase.COMPLETE, ServoCalibrationPhase.FAILED)


class ServoCalibrationWizard:
    """Guided per-finger servo calibration.

    Usage mirrors :class:`~neurogrip.emg.calibration.CalibrationWizard`: call
    :meth:`start`, then call :meth:`update` once per control cycle and render the
    returned progress. The wizard commands motion through the controller and
    reads back measured state; the caller only pumps it.

    Unlike EMG calibration this procedure *moves the hand*, so it refuses to
    start unless the drive is enabled and no emergency stop is latched, and it
    aborts on any fault that appears mid-run.
    """

    def __init__(
        self,
        controller: HandController,
        clock: Clock,
        *,
        base: ServoCalibrationSet | None = None,
        creep_rate: float = 0.10,
        settle_s: float = 0.6,
        takeup_current_ma: int = 35,
        stall_current_ma: int = 140,
        stall_dwell_s: float = 0.35,
        progress_epsilon: float = 0.004,
    ) -> None:
        self._controller = controller
        self._clock = clock
        self._base = base or ServoCalibrationSet()
        #: Closure units per second during the creep phases.
        self._creep_rate = creep_rate
        self._settle_s = settle_s
        #: Current rise above baseline that means the tendon has gone taut.
        self._takeup_current_ma = takeup_current_ma
        #: Current at which the finger is considered to be against a stop.
        self._stall_current_ma = stall_current_ma
        self._stall_dwell_s = stall_dwell_s
        #: Position change below which the finger counts as making no progress.
        self._progress_epsilon = progress_epsilon

        self._phase = ServoCalibrationPhase.IDLE
        self._order: tuple[Finger, ...] = tuple(Finger)
        self._index = 0
        self._phase_started = 0.0
        self._commanded = 0.0
        self._baseline_ma = 0
        self._slack: float | None = None
        self._last_position = 0.0
        self._start_position = 0.0
        #: Position and time of the last observed advance, for stall detection.
        self._progress_reference = 0.0
        self._progress_since = 0.0
        self._results: list[FingerCalibrationResult] = []
        self._message = ""
        self._result: ServoCalibrationSet | None = None

    # -- control --------------------------------------------------------------

    def start(self, fingers: tuple[Finger, ...] | None = None) -> ServoCalibrationProgress:
        """Begin calibration, optionally for a subset of fingers.

        Raises :class:`~neurogrip.core.errors.CalibrationError` if the hand is
        not in a state where it may be moved.
        """
        state = self._controller.state
        if state.estop:
            raise CalibrationError("cannot calibrate while the emergency stop is latched")
        if not state.enabled:
            raise CalibrationError("cannot calibrate while the drive is disabled")
        if not state.comms_ok:
            raise CalibrationError("cannot calibrate without a link to the motor controller")

        self._order = fingers or tuple(Finger)
        self._index = 0
        self._results = []
        self._result = None
        self._message = ""
        self._begin_finger()
        log.info("servo calibration started", fingers=[f.name for f in self._order])
        return self.progress()

    def cancel(self, reason: str = "cancelled") -> None:
        """Abort and return the hand to open."""
        if self._phase is not ServoCalibrationPhase.IDLE:
            self._controller.cancel(f"servo calibration {reason}")
            self._open_hand()
        self._phase = ServoCalibrationPhase.IDLE
        self._message = reason
        log.info("servo calibration cancelled", reason=reason)

    def update(self) -> ServoCalibrationProgress:
        """Advance the wizard by one control cycle."""
        if self._phase in (
            ServoCalibrationPhase.IDLE,
            ServoCalibrationPhase.COMPLETE,
            ServoCalibrationPhase.FAILED,
        ):
            return self.progress()

        state = self._controller.state
        if state.estop or not state.comms_ok:
            self._fail("hand became unavailable during calibration")
            return self.progress()

        finger = self._order[self._index]
        position = state.pose[finger]
        current = state.currents[int(finger)] if state.currents else 0
        elapsed = self._clock.monotonic() - self._phase_started

        if self._phase is ServoCalibrationPhase.BASELINE:
            self._update_baseline(elapsed, position, current)
        elif self._phase is ServoCalibrationPhase.TAKE_UP:
            self._update_take_up(position, current)
        elif self._phase is ServoCalibrationPhase.TRAVEL:
            self._update_travel(finger, position, current)
        elif self._phase is ServoCalibrationPhase.RELEASE:
            self._update_release(elapsed, position)

        self._last_position = position
        return self.progress()

    # -- phases ---------------------------------------------------------------

    def _begin_finger(self) -> None:
        """Open the hand and start the baseline measurement for the next finger."""
        self._phase = ServoCalibrationPhase.BASELINE
        self._phase_started = self._clock.monotonic()
        self._commanded = 0.0
        self._slack = None
        self._baseline_ma = 0
        self._open_hand()

    def _update_baseline(self, elapsed: float, position: float, current: int) -> None:
        # Take the baseline only once the finger has stopped moving: current
        # measured while decelerating reflects inertia, not tendon load. Velocity
        # comes from the controller rather than a position difference, which at
        # this loop rate would be dominated by quantisation.
        velocities = self._controller.state.velocities
        moving = bool(velocities) and abs(velocities[int(self._order[self._index])]) > 1e-3
        if elapsed < self._settle_s or moving:
            return
        self._baseline_ma = current
        self._start_position = position
        self._phase = ServoCalibrationPhase.TAKE_UP
        self._phase_started = self._clock.monotonic()

    def _update_take_up(self, position: float, current: int) -> None:
        self._creep()
        if current - self._baseline_ma >= self._takeup_current_ma:
            # The tendon is taut. The *commanded* closure is what matters here,
            # not the measured one: slack is defined as the command interval
            # over which the finger does not respond.
            self._slack = self._commanded
            self._begin_travel(position)
        elif self._commanded >= MAX_ACCEPTABLE_SLACK:
            # Never went taut within the acceptable range. Record it as a
            # problem rather than continuing to drive a possibly unstrung finger.
            self._slack = self._commanded
            self._begin_travel(position)

    def _begin_travel(self, position: float) -> None:
        self._phase = ServoCalibrationPhase.TRAVEL
        self._phase_started = self._clock.monotonic()
        self._progress_reference = position
        self._progress_since = self._phase_started

    def _update_travel(self, finger: Finger, position: float, current: int) -> None:
        self._creep()
        now = self._clock.monotonic()

        # Progress is measured over a window, never tick to tick. At the creep
        # rate a single 5 ms cycle advances the finger by ~0.0005 closure units,
        # which is below any sensible noise floor — a tick-to-tick comparison
        # would report "not moving" on every healthy cycle.
        if position > self._progress_reference + self._progress_epsilon:
            self._progress_reference = position
            self._progress_since = now

        over_current = current >= self._stall_current_ma
        stopped = now - self._progress_since >= self._stall_dwell_s
        if over_current or stopped or self._commanded >= 1.0:
            self._finish_finger(finger, position, current)

    def _update_release(self, elapsed: float, position: float) -> None:
        if elapsed < self._settle_s or position > 0.05:
            return
        self._index += 1
        if self._index >= len(self._order):
            self._finish()
        else:
            self._begin_finger()

    def _finish_finger(self, finger: Finger, position: float, current: int) -> None:
        """Record the measurement for the finger and start releasing it."""
        slack = self._slack if self._slack is not None else 0.0
        travel_end = self._commanded
        observed = position - self._start_position

        problems: list[str] = []
        if slack >= MAX_ACCEPTABLE_SLACK:
            problems.append(
                f"tendon never went taut below {MAX_ACCEPTABLE_SLACK:.0%} closure — re-string it"
            )
        if travel_end < MIN_ACCEPTABLE_TRAVEL:
            problems.append(
                f"binds at {travel_end:.0%} closure (want ≥ {MIN_ACCEPTABLE_TRAVEL:.0%}) — "
                "check tendon routing"
            )
        if observed < 0.2:
            problems.append(f"finger barely moved ({observed:.2f}) — check the horn and linkage")

        result = FingerCalibrationResult(
            finger=finger,
            slack=slack,
            travel_end=travel_end,
            baseline_current_ma=self._baseline_ma,
            loaded_current_ma=current,
            observed_travel=observed,
            problems=tuple(problems),
        )
        self._results.append(result)
        log.info("finger calibrated", detail=result.describe())

        self._phase = ServoCalibrationPhase.RELEASE
        self._phase_started = self._clock.monotonic()
        self._commanded = 0.0
        self._open_hand()

    def _finish(self) -> None:
        """Turn the per-finger measurements into a calibration set."""
        self._phase = ServoCalibrationPhase.VERIFY
        wall = time.time()
        calibration = ServoCalibrationSet(
            hand_id=self._base.hand_id,
            created_at=self._base.created_at or wall,
            updated_at=wall,
        )
        # Carry forward every finger the run did not touch, so calibrating a
        # single finger does not discard the other four.
        for finger in Finger:
            calibration.set(self._base.get(finger))

        for result in self._results:
            existing = self._base.get(result.finger)
            calibration.set(
                ServoCalibration(
                    finger=result.finger,
                    min_pulse_us=existing.min_pulse_us,
                    max_pulse_us=existing.max_pulse_us,
                    inverted=existing.inverted,
                    slack=round(result.slack, 4),
                )
            )

        failures = [r for r in self._results if not r.ok]
        if failures:
            self._phase = ServoCalibrationPhase.FAILED
            self._message = "; ".join(r.describe() for r in failures)
            log.warning("servo calibration failed", problems=self._message)
            return

        self._result = calibration
        self._phase = ServoCalibrationPhase.COMPLETE
        self._message = "Servo calibration complete"
        log.info(
            "servo calibration complete",
            slack={r.finger.name.lower(): round(r.slack, 3) for r in self._results},
        )

    def _fail(self, message: str) -> None:
        self._phase = ServoCalibrationPhase.FAILED
        self._message = message
        self._controller.cancel("servo calibration aborted")
        log.warning("servo calibration aborted", detail=message)

    # -- motion ---------------------------------------------------------------

    def _creep(self) -> None:
        """Advance the commanded closure of the active finger by one step."""
        period = self._clock.monotonic() - self._phase_started
        # Derive the command from elapsed time rather than accumulating per
        # tick, so a slow or jittery loop changes the rate, never the endpoint.
        target = clamp(self._creep_start + self._creep_rate * period)
        self._commanded = target
        self._command_finger(target)

    @property
    def _creep_start(self) -> float:
        """Closure the current creep phase began from."""
        if self._phase is ServoCalibrationPhase.TRAVEL and self._slack is not None:
            return self._slack
        return 0.0

    def _command_finger(self, closure: float) -> None:
        finger = self._order[self._index]
        pose = HandPose.open_hand().masked([finger], HandPose.uniform(closure))
        self._controller.move_to(
            pose,
            priority=Priority.USER_DIRECT,
            force=CALIBRATION_FORCE,
            speed=CALIBRATION_SPEED,
            source="servo-calibration",
            description=f"calibrating {finger.name.lower()}",
        )

    def _open_hand(self) -> None:
        self._controller.move_to(
            HandPose.open_hand(),
            priority=Priority.USER_DIRECT,
            force=CALIBRATION_FORCE,
            speed=CALIBRATION_SPEED,
            source="servo-calibration",
            description="returning to open",
        )

    # -- inspection -----------------------------------------------------------

    def progress(self) -> ServoCalibrationProgress:
        finger = self._order[self._index] if self._index < len(self._order) else None
        state = self._controller.state
        instructions = {
            ServoCalibrationPhase.BASELINE: "Hold still — measuring the unloaded current.",
            ServoCalibrationPhase.TAKE_UP: "Taking up tendon slack.",
            ServoCalibrationPhase.TRAVEL: "Finding the end of travel.",
            ServoCalibrationPhase.RELEASE: "Releasing.",
        }
        return ServoCalibrationProgress(
            phase=self._phase,
            finger=finger,
            finger_index=self._index,
            finger_count=len(self._order),
            instruction=instructions.get(self._phase, ""),
            title=finger.name.title() if finger else self._phase.value.title(),
            commanded=self._commanded,
            measured=state.pose[finger] if finger else 0.0,
            current_ma=state.currents[int(finger)] if finger and state.currents else 0,
            message=self._message,
            results=tuple(self._results),
        )

    @property
    def result(self) -> ServoCalibrationSet | None:
        """The finished calibration, or ``None`` if the run did not complete."""
        return self._result

    @property
    def phase(self) -> ServoCalibrationPhase:
        return self._phase

    @property
    def active(self) -> bool:
        """True only while a run is in progress. See the EMG wizard's ``active``."""
        return self._phase not in (
            ServoCalibrationPhase.IDLE,
            ServoCalibrationPhase.COMPLETE,
            ServoCalibrationPhase.FAILED,
        )

    @property
    def results(self) -> tuple[FingerCalibrationResult, ...]:
        return tuple(self._results)
