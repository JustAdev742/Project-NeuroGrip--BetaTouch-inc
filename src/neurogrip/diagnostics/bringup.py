"""Hardware bring-up tools.

The self-tests in :mod:`neurogrip.diagnostics.selftest` answer "is this device
healthy enough to run?" and must finish in well under a second, because they gate
every startup. These tools answer a different question — "is this hardware built
and wired correctly?" — and are allowed to take a minute, move every finger, and
deliberately trigger a safety system to see whether it responds.

Three tools, each corresponding to a class of bring-up failure that is painful to
diagnose from normal operation:

* :class:`LinkTester` — the link is *nominally* working but marginal. Bad crimp,
  cable too long, baud rate too high, ground loop. Shows up as intermittent
  motion glitches that look like software bugs.
* :class:`ServoSweepTest` — the bench test for a hand that is not built yet.
  Moves each servo in turn and then all five together, so a human can confirm
  the wiring, identify which servo is on which channel, and see whether the
  supply survives five starting at once.
* :class:`RangeTester` — a finger does not reach the travel the software assumes.
  Wrong horn spline, tendon routed through the wrong guide, servo mounted
  mirrored. Shows up as a grip that never closes properly on one digit. Assumes
  a *finished* hand; use ``ServoSweepTest`` before the tendons are strung.
* :class:`EstopTester` — the emergency stop path is not actually connected. Shows
  up when someone needs it.

The last one is the reason this module exists. An emergency stop that has never
been *tested* is not a safety system, it is an assumption; and the only honest
way to test it is to command motion and verify it stops.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..core.clock import Clock
from ..core.logging import get_logger
from ..core.ringbuffer import RunningStats, percentile
from ..core.types import Finger, HandPose, clamp
from ..hal.servo.base import ServoBus
from .selftest import TestOutcome, TestResult

__all__ = ["EstopTester", "LinkTester", "RangeTester", "ServoSweepTest", "ToolReport"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ToolReport:
    """Outcome of one bring-up tool."""

    tool: str
    results: tuple[TestResult, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(r.outcome is not TestOutcome.FAIL for r in self.results)

    @property
    def failures(self) -> tuple[TestResult, ...]:
        return tuple(r for r in self.results if r.outcome is TestOutcome.FAIL)

    def summary(self) -> str:
        passed = sum(1 for r in self.results if r.outcome is TestOutcome.PASS)
        skipped = sum(1 for r in self.results if r.outcome is TestOutcome.SKIP)
        total = len(self.results) - skipped
        suffix = f", {skipped} skipped" if skipped else ""
        return f"{self.tool}: {passed}/{total} checks passed{suffix}"

    def describe(self) -> tuple[str, ...]:
        lines = [f"  {r.outcome.symbol} {r.name}: {r.message}" for r in self.results]
        lines.extend(f"  · {note}" for note in self.notes)
        return tuple(lines)


class LinkTester:
    """Measures the quality of the motor-controller link.

    Round-trip latency matters more than throughput here: the control loop sends
    one command per cycle and needs the reply before the next one. A link with
    good average latency and a bad tail is worse than a uniformly slower one,
    which is why this reports p95 and worst-case rather than a mean.
    """

    def __init__(self, bus: ServoBus, clock: Clock, *, samples: int = 200) -> None:
        self._bus = bus
        self._clock = clock
        self._samples = samples

    def run(self) -> ToolReport:
        if not hasattr(self._bus, "ping"):
            # A simulated bus has no link to measure. Reporting a skip is honest;
            # reporting a pass would suggest the wiring had been checked.
            return ToolReport(
                tool="link",
                results=(
                    TestResult(
                        name="Round trip",
                        outcome=TestOutcome.SKIP,
                        message=f"{self._bus.info().driver} has no physical link to test",
                        remedy="Run against real hardware, or the 'emulator' driver.",
                    ),
                ),
            )
        if not self._bus.is_open:
            try:
                self._bus.open()
            except Exception as exc:
                return ToolReport(
                    tool="link",
                    results=(
                        TestResult(
                            name="Open link",
                            outcome=TestOutcome.FAIL,
                            message=str(exc),
                            remedy="Check the cable and that the controller is powered.",
                        ),
                    ),
                )

        results: list[TestResult] = []
        notes: list[str] = []

        before = self._link_stats()
        latencies: list[float] = []
        stats = RunningStats()
        lost = 0

        for index in range(self._samples):
            sent = self._clock.monotonic()
            try:
                self._bus.ping(index & 0xFFFF)
                # The reply is picked up by the next telemetry pump, so the
                # measured interval includes one poll period. That is the number
                # the control loop actually experiences.
                self._bus.read_state()
            except Exception:
                lost += 1
                continue
            elapsed = (self._clock.monotonic() - sent) * 1000.0
            latencies.append(elapsed)
            stats.add(elapsed)
            # Real hardware needs wall-clock spacing between pings; a simulated
            # clock advances only when someone asks it to.
            self._clock.sleep(0.002)

        after = self._link_stats()

        if not latencies:
            results.append(
                TestResult(
                    name="Round trip",
                    outcome=TestOutcome.FAIL,
                    message="no replies received",
                    remedy="The controller is not responding. Check power and firmware.",
                )
            )
        else:
            p95 = percentile(latencies, 0.95)
            worst = max(latencies)
            outcome = TestOutcome.PASS
            if p95 > 12.0:
                outcome = TestOutcome.WARN
            if p95 > 30.0:
                outcome = TestOutcome.FAIL
            results.append(
                TestResult(
                    name="Round trip",
                    outcome=outcome,
                    message=f"median {percentile(latencies, 0.5):.1f} ms, "
                    f"p95 {p95:.1f} ms, worst {worst:.1f} ms",
                    measurements={
                        "median_ms": round(percentile(latencies, 0.5), 2),
                        "p95_ms": round(p95, 2),
                        "worst_ms": round(worst, 2),
                        "jitter_ms": round(stats.std, 2),
                    },
                    remedy="Shorten the cable or lower the baud rate."
                    if outcome is not TestOutcome.PASS
                    else "",
                )
            )

        loss_pct = 100.0 * lost / max(1, self._samples)
        results.append(
            TestResult(
                name="Reply loss",
                outcome=TestOutcome.PASS if loss_pct == 0 else TestOutcome.FAIL,
                message=f"{lost}/{self._samples} lost ({loss_pct:.1f}%)",
                remedy="Any loss at all points at wiring or a ground problem."
                if lost
                else "",
            )
        )

        # Framing errors are the sensitive indicator: a link can deliver every
        # frame and still be corrupting bytes that the CRC catches.
        crc_errors = after.get("crc_errors", 0) - before.get("crc_errors", 0)
        desyncs = after.get("resyncs", 0) - before.get("resyncs", 0)
        results.append(
            TestResult(
                name="Framing",
                outcome=TestOutcome.PASS if crc_errors == 0 and desyncs == 0 else TestOutcome.FAIL,
                message=f"{crc_errors} CRC error(s), {desyncs} resync(s)",
                measurements={"crc_errors": crc_errors, "resyncs": desyncs},
                remedy="Corrupted frames mean electrical noise, not software."
                if crc_errors or desyncs
                else "",
            )
        )

        info = self._bus.info()
        if info.firmware_version:
            notes.append(f"firmware {info.firmware_version} on {info.connection}")
        for key in ("disconnects", "reconnects"):
            value = info.extra.get(key)
            if value:
                notes.append(f"{key}: {value}")

        return ToolReport(tool="link", results=tuple(results), notes=tuple(notes))

    def _link_stats(self) -> dict[str, int]:
        getter = getattr(self._bus, "link_stats", None)
        return dict(getter()) if callable(getter) else {}


class RangeTester:
    """Drives each finger through its full range and reports what it achieved.

    Runs one finger at a time. Moving all five and reading back the pose would be
    faster but would not distinguish "the ring finger is stuck" from "the ring
    finger's feedback is stuck", because a fouled tendon can drag its neighbour.
    """

    def __init__(
        self,
        controller,
        clock: Clock,
        *,
        settle_s: float = 0.25,
        timeout_s: float = 8.0,
        force: float = 0.25,
        speed: float = 0.4,
        min_travel: float = 0.85,
    ) -> None:
        self._controller = controller
        self._clock = clock
        #: How long the hand must be still before it counts as arrived.
        self._settle_s = settle_s
        #: Upper bound on one move, so a jammed finger cannot hang the test.
        self._timeout_s = timeout_s
        self._force = force
        self._speed = speed
        #: Fraction of nominal travel a healthy finger must achieve.
        self._min_travel = min_travel

    def run(self, fingers: tuple[Finger, ...] = tuple(Finger)) -> ToolReport:
        state = self._controller.state
        if state.estop:
            return ToolReport(
                tool="range",
                results=(
                    TestResult(
                        name="Precondition",
                        outcome=TestOutcome.FAIL,
                        message="emergency stop is latched",
                        remedy="Acknowledge the stop before running a motion test.",
                    ),
                ),
            )
        if not state.enabled:
            self._controller.enable()

        results: list[TestResult] = []
        for finger in fingers:
            results.append(self._test_finger(finger))
        self._move(HandPose.open_hand(), "returning to open")
        return ToolReport(tool="range", results=tuple(results))

    def _test_finger(self, finger: Finger) -> TestResult:
        self._move(HandPose.open_hand(), f"opening before {finger.name.lower()}")
        opened = self._controller.state.pose[finger]

        target = HandPose.open_hand().masked([finger], HandPose.closed_hand())
        self._move(target, f"closing {finger.name.lower()}")
        closed = self._controller.state.pose[finger]

        travel = closed - opened
        neighbours = self._neighbour_motion(finger)

        if travel < self._min_travel * 0.5:
            return TestResult(
                name=f"{finger.name.title()} range",
                outcome=TestOutcome.FAIL,
                message=f"travelled {travel:.2f} of 1.00",
                measurements={"travel": round(travel, 3), "coupling": round(neighbours, 3)},
                remedy="Check the horn, the tendon anchor, and that the servo is powered.",
            )
        if travel < self._min_travel:
            return TestResult(
                name=f"{finger.name.title()} range",
                outcome=TestOutcome.WARN,
                message=f"travelled {travel:.2f} of 1.00",
                measurements={"travel": round(travel, 3), "coupling": round(neighbours, 3)},
                remedy="Tendon is slightly short or the routing binds. Run servo calibration.",
            )
        if neighbours > 0.08:
            # Real coupling: one tendon dragging another is a routing fault that
            # full-hand tests hide, because everything moves anyway.
            return TestResult(
                name=f"{finger.name.title()} range",
                outcome=TestOutcome.WARN,
                message=f"travelled {travel:.2f}, but moved other fingers by {neighbours:.2f}",
                measurements={"travel": round(travel, 3), "coupling": round(neighbours, 3)},
                remedy="Tendons are rubbing. Check the routing guides.",
            )
        return TestResult(
            name=f"{finger.name.title()} range",
            outcome=TestOutcome.PASS,
            message=f"travelled {travel:.2f} of 1.00",
            measurements={"travel": round(travel, 3), "coupling": round(neighbours, 3)},
        )

    def _neighbour_motion(self, finger: Finger) -> float:
        pose = self._controller.state.pose
        return max(
            (pose[other] for other in Finger if other is not finger),
            default=0.0,
        )

    def _move(self, pose: HandPose, description: str) -> None:
        self._controller.move_to(
            pose,
            force=self._force,
            speed=self._speed,
            source="range-test",
            description=description,
        )
        self._settle()

    def _settle(self) -> None:
        """Run the control loop until the hand stops moving.

        Waits for arrival rather than a fixed duration. A fixed wait has to be
        long enough for the slowest configured speed, and if it ever is not, the
        test reports a short range for a healthy finger — the exact false alarm
        this tool exists to rule out.
        """
        deadline = self._clock.monotonic() + self._timeout_s
        still_since: float | None = None
        while self._clock.monotonic() < deadline:
            self._controller.tick()
            self._clock.sleep(0.005)
            now = self._clock.monotonic()
            if self._controller.state.moving:
                still_since = None
                continue
            if still_since is None:
                still_since = now
            elif now - still_since >= self._settle_s:
                return


class ServoSweepTest:
    """Drives every servo through its full range so you can watch it.

    Distinct from :class:`RangeTester`, and the difference matters. RangeTester
    is an *acceptance* test for a finished hand: it measures travel against a
    threshold and flags cross-coupling between tendons. Run it on five bare
    servos sitting on a bench and it fails everything, because there are no
    tendons to travel and nothing to couple.

    This is the bench test for a hand that is not built yet. Its job is to make
    each servo move, one at a time and then together, so a human can confirm the
    wiring, identify which physical servo is on which channel, check the
    direction of travel, and see whether the supply holds up when all five move
    at once. It reports what it commanded and what came back, and it only
    *fails* a channel that did not move at all.

    That last distinction is the point. On a board without position feedback —
    the micro:bit — "what came back" is the firmware's own open-loop estimate, so
    the test cannot tell you the servo physically moved. It says so rather than
    reporting a pass that means nothing. Watching the hardware is the
    measurement; this tool is what makes the hardware move in a defined,
    repeatable order while you do.
    """

    def __init__(
        self,
        controller,
        clock: Clock,
        *,
        cycles: int = 1,
        hold_s: float = 0.6,
        travel: float = 1.0,
        force: float = 0.25,
        speed: float = 0.5,
        settle_s: float = 0.2,
        timeout_s: float = 8.0,
        has_feedback: bool = True,
    ) -> None:
        self._controller = controller
        self._clock = clock
        self._cycles = max(1, cycles)
        #: Pause at each end of travel, so a human can see where it stopped.
        self._hold_s = hold_s
        #: Fraction of full closure to sweep. Lower it when the mechanism is
        #: partly assembled and full travel would collide with something.
        self._travel = clamp(travel, 0.05, 1.0)
        self._force = force
        self._speed = speed
        self._settle_s = settle_s
        self._timeout_s = timeout_s
        #: False on a board that reports commanded position rather than measured.
        self._has_feedback = has_feedback

    def run(self, fingers: tuple[Finger, ...] = tuple(Finger)) -> ToolReport:
        state = self._controller.state
        if state.estop:
            return ToolReport(
                tool="servos",
                results=(
                    TestResult(
                        name="Precondition",
                        outcome=TestOutcome.FAIL,
                        message="emergency stop is latched",
                        remedy="Acknowledge the stop before running a motion test.",
                    ),
                ),
            )
        if not state.enabled:
            self._controller.enable()

        results: list[TestResult] = []
        notes: list[str] = [
            f"{self._cycles} cycle(s) to {self._travel:.0%} closure "
            f"at {self._speed:.0%} speed"
        ]
        if not self._has_feedback:
            notes.append(
                "this controller has no position feedback — positions below are "
                "the firmware's open-loop estimate. Watch the hardware."
            )

        # One at a time first: this is what tells you which servo is on which
        # channel, and it is impossible to work out from an all-together sweep.
        for finger in fingers:
            results.append(self._sweep_one(finger))

        results.append(self._sweep_all(fingers))

        self._move(HandPose.open_hand(), "returning to open")
        return ToolReport(tool="servos", results=tuple(results), notes=tuple(notes))

    # -- individual -----------------------------------------------------------

    def _sweep_one(self, finger: Finger) -> TestResult:
        name = finger.name.title()
        self._move(HandPose.open_hand(), f"opening before {name.lower()}")
        opened = self._controller.state.pose[finger]

        reached: list[float] = []
        for _ in range(self._cycles):
            closed_pose = HandPose.open_hand().masked(
                [finger], HandPose.uniform(self._travel)
            )
            self._move(closed_pose, f"closing {name.lower()}")
            reached.append(self._controller.state.pose[finger])
            self._dwell()

            self._move(HandPose.open_hand(), f"opening {name.lower()}")
            self._dwell()

        returned = self._controller.state.pose[finger]
        travelled = max(reached, default=0.0) - opened

        if travelled < 0.05:
            return TestResult(
                name=f"{name} servo",
                outcome=TestOutcome.FAIL,
                message=f"did not move (commanded {self._travel:.2f}, reached {travelled:.2f})",
                measurements={"travelled": round(travelled, 3)},
                remedy=(
                    "Check the signal wire, the servo's power, and that the "
                    "channel is the one you think it is."
                ),
            )
        if returned > 0.1:
            return TestResult(
                name=f"{name} servo",
                outcome=TestOutcome.WARN,
                message=f"swept {travelled:.2f} but did not return to open ({returned:.2f})",
                measurements={"travelled": round(travelled, 3), "returned": round(returned, 3)},
                remedy="The servo may be binding, or the travel range is wrong for it.",
            )
        return TestResult(
            name=f"{name} servo",
            outcome=TestOutcome.PASS,
            message=f"swept 0.00 → {travelled:.2f} → {returned:.2f}"
            + ("" if self._has_feedback else " (commanded)"),
            measurements={"travelled": round(travelled, 3)},
        )

    # -- all together ---------------------------------------------------------

    def _sweep_all(self, fingers: tuple[Finger, ...]) -> TestResult:
        """The one that finds an undersized power supply.

        Five servos starting together draw their inrush at the same instant. A
        supply that copes with one at a time will brown out here, and the symptom
        is the controller resetting rather than anything the software can see —
        which is exactly why a human needs to be watching.
        """
        self._move(HandPose.open_hand(), "opening all")
        before = self._controller.state

        worst = 1.0
        for _ in range(self._cycles):
            target = HandPose.from_iterable(
                self._travel if f in fingers else 0.0 for f in Finger
            )
            self._move(target, "closing all")
            self._dwell()
            pose = self._controller.state.pose
            worst = min(worst, min(pose[f] for f in fingers))
            self._move(HandPose.open_hand(), "opening all")
            self._dwell()

        after = self._controller.state
        if not after.comms_ok:
            return TestResult(
                name="All servos together",
                outcome=TestOutcome.FAIL,
                message="lost the controller during the sweep",
                remedy=(
                    "Almost always an undersized supply: five servos starting "
                    "together brown out the board. Use a separate 5 V supply "
                    "with a common ground."
                ),
            )
        if worst < 0.05:
            return TestResult(
                name="All servos together",
                outcome=TestOutcome.FAIL,
                message=f"at least one channel did not move (worst {worst:.2f})",
                remedy="Re-run the individual sweeps to find which.",
            )
        return TestResult(
            name="All servos together",
            outcome=TestOutcome.PASS,
            message=f"all {len(fingers)} moved together, worst {worst:.2f}",
            measurements={
                "worst": round(worst, 3),
                "voltage_before": round(before.bus_voltage_v, 2),
                "voltage_after": round(after.bus_voltage_v, 2),
            },
        )

    # -- motion ---------------------------------------------------------------

    def _move(self, pose: HandPose, description: str) -> None:
        self._controller.move_to(
            pose,
            force=self._force,
            speed=self._speed,
            source="servo-sweep",
            description=description,
        )
        self._settle()

    def _settle(self) -> None:
        """Run the control loop until the hand stops moving, or time out."""
        deadline = self._clock.monotonic() + self._timeout_s
        still_since: float | None = None
        while self._clock.monotonic() < deadline:
            self._controller.tick()
            self._clock.sleep(0.005)
            now = self._clock.monotonic()
            if self._controller.state.moving:
                still_since = None
                continue
            if still_since is None:
                still_since = now
            elif now - still_since >= self._settle_s:
                return

    def _dwell(self) -> None:
        """Hold at the end of travel so a human can see where it stopped."""
        deadline = self._clock.monotonic() + self._hold_s
        while self._clock.monotonic() < deadline:
            self._controller.tick()
            self._clock.sleep(0.005)


class EstopTester:
    """Verifies that the emergency stop actually stops the hand.

    Commands a slow closing motion, triggers the stop mid-travel, and checks
    three things:

    1. motion ceases;
    2. the drive is de-energised;
    3. the latch holds — motion stays refused until someone acknowledges.

    Point 3 is the one worth testing. A stop that engages but can be cleared by
    the next command is not a stop, and nothing in normal operation reveals that.
    """

    def __init__(self, controller, safety, clock: Clock, *, settle_s: float = 0.8) -> None:
        self._controller = controller
        self._safety = safety
        self._clock = clock
        self._settle_s = settle_s

    def run(self) -> ToolReport:
        results: list[TestResult] = []

        if self._controller.state.estop:
            self._safety.acknowledge("estop-test")
            self._controller.clear_emergency_stop()
        if not self._controller.state.enabled:
            self._controller.enable()

        # Start a slow motion so there is something to interrupt.
        self._controller.move_to(
            HandPose.closed_hand(),
            force=0.2,
            speed=0.3,
            source="estop-test",
            description="e-stop test motion",
        )
        self._run_for(0.4)
        moving_before = self._controller.state.moving
        pose_before = self._controller.state.pose

        self._safety.trigger_estop("emergency stop test", source="selftest")
        self._run_for(self._settle_s)
        after = self._controller.state

        results.append(
            TestResult(
                name="Motion started",
                outcome=TestOutcome.PASS if moving_before else TestOutcome.WARN,
                message="hand was moving before the stop"
                if moving_before
                else "hand was not moving; the stop was not exercised under load",
                remedy="" if moving_before else "Check that the drive is enabled and homed.",
            )
        )

        travelled = max(
            abs(after.pose[f] - pose_before[f]) for f in Finger
        )
        results.append(
            TestResult(
                name="Motion stopped",
                outcome=TestOutcome.PASS if not after.moving else TestOutcome.FAIL,
                message=f"moving={after.moving} after stop, drifted {travelled:.3f}",
                measurements={"drift": round(travelled, 4)},
                remedy="The stop did not halt motion. Do not use this hand."
                if after.moving
                else "",
            )
        )
        results.append(
            TestResult(
                name="Drive de-energised",
                outcome=TestOutcome.PASS if not after.enabled else TestOutcome.FAIL,
                message="drive off" if not after.enabled else "drive still energised",
                remedy="The stop must cut drive, not merely stop commanding motion."
                if after.enabled
                else "",
            )
        )

        # The latch: a command issued now must be refused.
        rejected = self._command_is_refused()
        results.append(
            TestResult(
                name="Latch holds",
                outcome=TestOutcome.PASS if rejected else TestOutcome.FAIL,
                message="motion refused while stopped"
                if rejected
                else "a motion command was accepted while stopped",
                remedy="The stop must latch until acknowledged." if not rejected else "",
            )
        )

        cleared = self._safety.acknowledge("estop-test")
        self._controller.clear_emergency_stop()
        self._run_for(0.2)
        results.append(
            TestResult(
                name="Recovery",
                outcome=TestOutcome.PASS if cleared else TestOutcome.WARN,
                message="cleared after acknowledgement"
                if cleared
                else "could not clear; another fault may still be active",
            )
        )

        return ToolReport(
            tool="estop",
            results=tuple(results),
            notes=(f"tested at {time.strftime('%Y-%m-%d %H:%M:%S')}",),
        )

    def _command_is_refused(self) -> bool:
        result = self._controller.move_to(
            HandPose.closed_hand(),
            force=0.2,
            speed=0.3,
            source="estop-test",
            description="post-stop probe",
        )
        return not result.accepted

    def _run_for(self, seconds: float) -> None:
        deadline = self._clock.monotonic() + seconds
        while self._clock.monotonic() < deadline:
            self._controller.tick()
            self._clock.sleep(0.005)
