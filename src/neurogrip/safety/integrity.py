"""Periodic emergency-stop integrity checking.

`neurogrip test estop` verifies the whole stop path, but only when a human runs
it. Between two such runs — which in practice means between two bring-up
sessions, months apart — the path can break with no symptom whatsoever. A lost
listener registration, a refactor that drops a call, a firmware regression: none
of them change any observable behaviour until the day the stop is needed, and
that is the worst possible day to discover it.

That is not a hypothetical failure mode. The registration that connects
:class:`~neurogrip.safety.estop.EmergencyStop` to the controller was missing from
this codebase for its entire first version. Everything worked. Every test passed.

The path has two halves, and they need different treatment because only one of
them can be exercised for free:

**The software chain** — ``EmergencyStop`` → listeners → ``HandController``.
Verified by :meth:`EmergencyStop.rehearse`, which calls every listener with a
record marked ``rehearsal``. The controller acknowledges and does nothing else.
Costs nothing, so it runs every 30 seconds.

**The hardware chain** — ``HandController`` → ``ServoBus`` → firmware → drive
de-energised. This one cannot be faked. Proving the actuators would actually stop
means stopping them, so the proof test really does cut drive, confirm from
*telemetry* that the firmware honoured it, and re-arm. It is therefore heavily
gated and infrequent.

What "heavily gated" means, concretely: the hand must be open, idle, holding
nothing, under no motion command, and in communication with the controller.
Under those conditions cutting drive moves nothing — the fingers are already at
rest and a tendon hand's return springs hold them open.

The window is one control cycle: drive goes down, telemetry confirms it, drive
comes back. Measured at 5 ms with zero finger movement, which is why there is no
machinery here to abort on user intent appearing mid-test. A command issued
inside that window is refused and re-submitted by the next decision cycle 10 ms
later, which is below anything a person can perceive. If the window ever grows —
a slower link, a firmware that acknowledges lazily — that reasoning stops
holding, and an intent gate would need adding to :meth:`EstopSelfCheck._safe_to_prove`.

A failure degrades to Manual rather than stopping the hand. That asymmetry is
deliberate and worth stating, because the opposite reading is tempting: a broken
e-stop sounds like grounds for refusing to run at all. But the stop exists to
catch motion the user did not ask for, and in Manual every motion is directly
driven by muscle — releasing the contraction stops it. Disabling assistance
removes the hazard the stop was guarding against, while stopping the hand
outright would take someone's limb away because a backup mechanism is unproven.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..control.controller import HandController
from ..core.clock import Clock
from ..core.errors import Severity
from ..core.events import EventBus
from ..core.logging import get_logger
from ..core.topics import Topics
from .estop import EmergencyStop
from .rules import Fault, SafetyContext, _BaseRule

__all__ = ["EstopCheckResult", "EstopIntegrityRule", "EstopSelfCheck", "IntegrityStatus"]

log = get_logger(__name__)


class IntegrityStatus(str, Enum):
    """Confidence in the emergency-stop path."""

    #: No check has completed yet. Not a failure — the system has just started.
    UNKNOWN = "unknown"
    #: The software chain is verified; the hardware chain has not been proven
    #: this session, usually because the hand has never been idle enough.
    CHAIN_OK = "chain_ok"
    #: Both halves verified.
    VERIFIED = "verified"
    #: A check failed. The stop cannot be relied on.
    FAILED = "failed"

    @property
    def trustworthy(self) -> bool:
        return self is not IntegrityStatus.FAILED

    @property
    def label(self) -> str:
        return {
            IntegrityStatus.UNKNOWN: "not yet checked",
            IntegrityStatus.CHAIN_OK: "signalling path verified",
            IntegrityStatus.VERIFIED: "fully verified",
            IntegrityStatus.FAILED: "FAILED",
        }[self]


@dataclass(frozen=True, slots=True)
class EstopCheckResult:
    """Outcome of one integrity check."""

    passed: bool
    kind: str
    message: str
    timestamp: float = 0.0
    detail: dict[str, float | str] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{'✓' if self.passed else '✗'} e-stop {self.kind}: {self.message}"


class _ProofPhase(str, Enum):
    """Stages of the hardware proof test."""

    IDLE = "idle"
    #: Stop commanded; waiting for telemetry to confirm the firmware honoured it.
    CONFIRMING = "confirming"
    #: Latch cleared; waiting for the drive to come back.
    REARMING = "rearming"


class EstopSelfCheck:
    """Periodically verifies that the emergency stop still works.

    Ticked from the diagnostics rate group. Cheap on almost every tick: the work
    is a pair of interval comparisons, and the proof test only runs when the hand
    has nothing better to do.
    """

    def __init__(
        self,
        estop: EmergencyStop,
        controller: HandController,
        clock: Clock,
        bus: EventBus | None = None,
        *,
        rehearsal_interval_s: float = 30.0,
        proof_interval_s: float = 6 * 3600.0,
        proof_enabled: bool = True,
        confirm_timeout_s: float = 0.5,
        rearm_timeout_s: float = 1.0,
    ) -> None:
        self._estop = estop
        self._controller = controller
        self._clock = clock
        self._bus = bus
        self._rehearsal_interval = rehearsal_interval_s
        self._proof_interval = proof_interval_s
        self._proof_enabled = proof_enabled
        self._confirm_timeout = confirm_timeout_s
        self._rearm_timeout = rearm_timeout_s

        #: ``None`` rather than ``0.0``: a simulated clock starts at zero, so
        #: zero is a real time and cannot double as "never".
        self._last_rehearsal_at: float | None = None
        self._last_proof_at: float | None = None
        self._phase = _ProofPhase.IDLE
        self._phase_started = 0.0
        self._status = IntegrityStatus.UNKNOWN
        self._last_result: EstopCheckResult | None = None
        self._last_failure: EstopCheckResult | None = None
        self.rehearsals = 0
        self.proofs = 0
        self.failures = 0

    # -- inspection -----------------------------------------------------------

    @property
    def status(self) -> IntegrityStatus:
        return self._status

    @property
    def last_result(self) -> EstopCheckResult | None:
        return self._last_result

    @property
    def last_failure(self) -> EstopCheckResult | None:
        """The most recent failure. Sticky — it is not cleared by a later pass.

        A stop that failed once and passed afterwards is not a stop anyone should
        rely on until someone has looked at why.
        """
        return self._last_failure

    @property
    def busy(self) -> bool:
        """True while a proof test is in progress."""
        return self._phase is not _ProofPhase.IDLE

    def describe(self) -> str:
        parts = [f"e-stop integrity: {self._status.label}"]
        if self._last_result is not None:
            parts.append(str(self._last_result))
        return " — ".join(parts)

    # -- the periodic check ---------------------------------------------------

    def tick(self) -> None:
        """Advance the checker. Safe to call at any rate; it paces itself."""
        now = self._clock.monotonic()

        if self._phase is not _ProofPhase.IDLE:
            self._advance_proof(now)
            return

        if self._due(self._last_rehearsal_at, now, self._rehearsal_interval):
            self._run_rehearsal(now)

        if self._proof_enabled and self._due(self._last_proof_at, now, self._proof_interval):
            self._try_start_proof(now)

    def _due(self, last: float | None, now: float, interval: float) -> bool:
        return last is None or (now - last) >= interval

    # -- the software chain ---------------------------------------------------

    def _run_rehearsal(self, now: float) -> None:
        """Verify the notification path reaches the controller."""
        self._last_rehearsal_at = now
        self.rehearsals += 1

        if self._estop.listener_count == 0:
            self._fail(
                "rehearsal",
                "nothing is listening to the emergency stop",
                now,
                {"listeners": 0},
            )
            return

        before = self._controller.last_estop_rehearsal
        self._estop.rehearse()
        after = self._controller.last_estop_rehearsal

        if after is None or after == before:
            self._fail(
                "rehearsal",
                "the emergency stop does not reach the motion controller",
                now,
                {"listeners": self._estop.listener_count},
            )
            return

        self._pass(
            "rehearsal",
            f"signalling path intact ({self._estop.listener_count} listener(s))",
            now,
            # The rehearsal alone never claims full verification: it says nothing
            # about whether the actuators would stop.
            promote_to=IntegrityStatus.VERIFIED
            if self._status is IntegrityStatus.VERIFIED
            else IntegrityStatus.CHAIN_OK,
        )

    # -- the hardware chain ---------------------------------------------------

    def _safe_to_prove(self) -> str:
        """Empty string when the proof test may run, or the reason it may not."""
        state = self._controller.state
        if state.estop:
            return "already stopped"
        if not state.enabled:
            return "drive is not enabled"
        if not state.comms_ok:
            return "no link to the motor controller"
        if state.moving:
            return "the hand is moving"
        if state.holding:
            return "the hand is holding something"
        if self._controller.queue.active is not None:
            return "a motion command is active"
        if not state.is_open:
            # Cutting drive with the fingers part-closed lets them relax, which
            # the user would see and feel. Open, they are already at rest.
            return "the hand is not open"
        return ""

    def _try_start_proof(self, now: float) -> None:
        blocked = self._safe_to_prove()
        if blocked:
            # Not a failure and not logged at each attempt — a hand in use is
            # simply not a hand to run diagnostics on. It will be idle later.
            return

        log.info("running e-stop proof test")
        self._phase = _ProofPhase.CONFIRMING
        self._phase_started = now
        self._controller.emergency_stop("periodic integrity check", diagnostic=True)

    def _advance_proof(self, now: float) -> None:
        elapsed = now - self._phase_started
        state = self._controller.state

        if self._phase is _ProofPhase.CONFIRMING:
            # The check is on *telemetry*, not on the command having been sent:
            # the question is whether the far end honoured it.
            if state.estop and not state.enabled:
                self._begin_rearm(now)
                return
            if elapsed >= self._confirm_timeout:
                self._fail(
                    "proof",
                    "drive was not de-energised by the emergency stop",
                    now,
                    {"estop": str(state.estop), "enabled": str(state.enabled)},
                )
                # Still attempt to re-arm: leaving the hand stopped because the
                # stop misbehaved would compound the problem.
                self._begin_rearm(now)
            return

        # REARMING
        if state.enabled and not state.estop:
            self._last_proof_at = now
            self.proofs += 1
            if self._last_result is not None and not self._last_result.passed:
                # The confirm step already failed; do not overwrite that verdict.
                return
            self._pass(
                "proof",
                "drive de-energised and re-armed",
                now,
                promote_to=IntegrityStatus.VERIFIED,
            )
            return

        if elapsed >= self._rearm_timeout:
            self._last_proof_at = now
            self._fail(
                "proof",
                "the hand did not re-arm after the check",
                now,
                {"estop": str(state.estop), "enabled": str(state.enabled)},
            )

    def _begin_rearm(self, now: float) -> None:
        self._phase = _ProofPhase.REARMING
        self._phase_started = now
        try:
            self._controller.clear_emergency_stop()
            self._controller.enable()
        except Exception as exc:
            log.error("could not re-arm after the e-stop proof test", error=str(exc))

    # -- results --------------------------------------------------------------

    def _pass(
        self, kind: str, message: str, now: float, *, promote_to: IntegrityStatus
    ) -> None:
        self._phase = _ProofPhase.IDLE
        result = EstopCheckResult(passed=True, kind=kind, message=message, timestamp=now)
        self._last_result = result
        # A previous failure is never cleared by a later pass. Someone has to
        # look at why it failed.
        if self._last_failure is None:
            self._status = promote_to
        log.debug("e-stop check passed", kind=kind, detail=message)

    def _fail(
        self, kind: str, message: str, now: float, detail: dict[str, float | str] | None = None
    ) -> None:
        self._phase = _ProofPhase.IDLE
        result = EstopCheckResult(
            passed=False, kind=kind, message=message, timestamp=now, detail=detail or {}
        )
        self._last_result = result
        self._last_failure = result
        self._status = IntegrityStatus.FAILED
        self.failures += 1
        log.critical("E-STOP INTEGRITY CHECK FAILED", kind=kind, detail=message)
        if self._bus is not None:
            self._bus.publish(Topics.SAFETY_FAULT_RAISED, result, source="estop-integrity")

    def reset(self, reason: str = "manually cleared") -> None:
        """Clear a sticky failure, after someone has investigated it."""
        self._last_failure = None
        self._status = IntegrityStatus.UNKNOWN
        self._last_rehearsal_at = None
        self._last_proof_at = None
        log.warning("e-stop integrity failure cleared", reason=reason)


class EstopIntegrityRule(_BaseRule):
    """Reports an emergency stop that has failed its integrity check.

    A rule rather than something the checker does itself, so the response goes
    through the same severity ladder as every other fault and appears in the same
    places: the diagnostics screen, the fault log, the black box.
    """

    rule_name = "estop_integrity"

    def __init__(self, check: EstopSelfCheck, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self._check = check

    def evaluate(self, context: SafetyContext) -> Fault | None:
        failure = self._check.last_failure
        if failure is None:
            return None
        return Fault(
            code="estop_integrity",
            severity=Severity.FALLBACK,
            message=f"emergency stop failed its self-check: {failure.message}",
            rule=self.name,
            detail={"kind": failure.kind, **failure.detail},
            remedy=(
                "AI assistance is disabled; direct control continues. "
                "Run `neurogrip test estop` and do not rely on the stop until it passes."
            ),
        )
