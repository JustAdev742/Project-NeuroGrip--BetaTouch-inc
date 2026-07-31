"""Motion command queue with priority and preemption.

Commands arrive from several places — the active mode, the training exercises,
the UI's manual controls, the calibration routine, the safety layer — and they
must not fight over the actuators. The queue arbitrates by priority:

===========================  ========================================
priority                     example
===========================  ========================================
``EMERGENCY``                e-stop, safe-hold on fault
``USER_OVERRIDE``            EMG cancel, on-screen stop
``USER_DIRECT``              proportional EMG control
``ASSISTED``                 an AI grasp plan
``BACKGROUND``               idle relax pose, homing
===========================  ========================================

Rules:

* A higher-priority command **preempts** whatever is running, immediately.
* An equal-priority command **replaces** it (the newest user command wins —
  a user pressing again means "I meant that", not "queue it up").
* A lower-priority command is dropped, not queued behind. Queueing stale motion
  behind a user action would produce a hand that moves on its own after the user
  has stopped asking, which is exactly the behaviour this project forbids.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

from ..core.types import HandPose

__all__ = ["CommandResult", "MotionCommand", "MotionQueue", "Priority"]


class Priority(IntEnum):
    """Command priority; higher wins."""

    BACKGROUND = 10
    ASSISTED = 20
    USER_DIRECT = 30
    USER_OVERRIDE = 40
    EMERGENCY = 50

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class MotionCommand:
    """A request to move the hand."""

    target: HandPose
    priority: Priority = Priority.USER_DIRECT
    #: Force ceiling for this motion, ``[0, 1]``.
    force: float = 0.5
    #: Speed multiplier.
    speed: float = 1.0
    #: Optional intermediate pose to pass through first.
    preshape: HandPose | None = None
    #: Who issued this — appears in logs and the black-box record.
    source: str = ""
    #: Human-readable purpose, shown in the UI.
    description: str = ""
    #: Abort if not completed within this many seconds. ``0`` means no limit.
    timeout_s: float = 8.0
    issued_at: float = 0.0
    #: Set for commands that must run to completion before yielding to an equal
    #: priority (used by the homing routine).
    atomic: bool = False
    #: Arbitrary payload the issuer can use to correlate completion events.
    tag: str = ""

    def with_target(self, target: HandPose) -> MotionCommand:
        from dataclasses import replace

        return replace(self, target=target)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Outcome of submitting a command."""

    accepted: bool
    preempted: MotionCommand | None = None
    reason: str = ""
    #: True when this replaced a command from the *same* continuous stream
    #: (same source and priority). Continuous control re-issues its command
    #: every cycle; the controller uses this to retarget the trajectory instead
    #: of restarting it, which would zero the velocity and reset the pre-shape
    #: leg on every tick.
    same_stream: bool = False


@dataclass(slots=True)
class _ActiveCommand:
    command: MotionCommand
    started_at: float
    #: True once the pre-shape leg has finished (or if there was none).
    preshape_done: bool = False


class MotionQueue:
    """Single-slot priority arbiter with a short history for diagnostics."""

    def __init__(self, *, history: int = 32) -> None:
        self._active: _ActiveCommand | None = None
        self._history: deque[tuple[float, MotionCommand, str]] = deque(maxlen=history)
        #: Diagnostics counters.
        self.accepted = 0
        self.rejected = 0
        self.preempted = 0
        self.completed = 0
        self.timed_out = 0

    # -- submission -----------------------------------------------------------

    def submit(self, command: MotionCommand, now: float) -> CommandResult:
        """Offer a command to the queue."""
        active = self._active

        if active is None:
            self._activate(command, now, "accepted")
            return CommandResult(accepted=True)

        if command.priority > active.command.priority:
            preempted = active.command
            self.preempted += 1
            self._record(now, preempted, f"preempted by {command.priority.label}")
            self._activate(command, now, "accepted (preempting)")
            return CommandResult(accepted=True, preempted=preempted, reason="preempted lower priority")

        if command.priority == active.command.priority:
            if active.command.atomic:
                self.rejected += 1
                return CommandResult(
                    accepted=False, reason="an atomic motion of equal priority is in progress"
                )
            replaced = active.command
            same_stream = (
                replaced.source == command.source and replaced.priority == command.priority
            )
            if same_stream:
                # Same continuous stream: keep the pre-shape progress so the
                # motion carries on rather than restarting from the beginning,
                # and refresh the timeout — a stream that is still being issued
                # is by definition alive. A genuinely jammed finger is caught by
                # ServoTimeoutRule, which watches tracking error, not wall time.
                self._activate(command, now, "updated", inherit_preshape=active.preshape_done)
                return CommandResult(accepted=True, reason="updated", same_stream=True)
            self._record(now, replaced, "replaced by newer command")
            self._activate(command, now, "accepted (replacing)")
            return CommandResult(accepted=True, preempted=replaced, reason="replaced")

        self.rejected += 1
        return CommandResult(
            accepted=False,
            reason=(
                f"{command.priority.label} is below the active "
                f"{active.command.priority.label} command"
            ),
        )

    def _activate(
        self,
        command: MotionCommand,
        now: float,
        note: str,
        *,
        inherit_preshape: bool = False,
    ) -> None:
        self._active = _ActiveCommand(
            command=command,
            started_at=now,
            preshape_done=inherit_preshape or command.preshape is None,
        )
        self.accepted += 1
        self._record(now, command, note)

    def _record(self, now: float, command: MotionCommand, note: str) -> None:
        self._history.append((now, command, note))

    # -- execution ------------------------------------------------------------

    @property
    def active(self) -> MotionCommand | None:
        return self._active.command if self._active else None

    @property
    def is_busy(self) -> bool:
        return self._active is not None

    @property
    def current_leg_target(self) -> HandPose | None:
        """The pose to move towards right now: pre-shape first, then the target."""
        if self._active is None:
            return None
        command = self._active.command
        if not self._active.preshape_done and command.preshape is not None:
            return command.preshape
        return command.target

    @property
    def preshape_pending(self) -> bool:
        """True while the active command still has a pre-shape leg to run."""
        return (
            self._active is not None
            and not self._active.preshape_done
            and self._active.command.preshape is not None
        )

    def mark_preshape_done(self) -> None:
        """Called by the controller once the pre-shape pose has been reached."""
        if self._active is not None:
            self._active.preshape_done = True

    def complete(self, now: float, note: str = "completed") -> MotionCommand | None:
        """Mark the active command finished; returns it."""
        if self._active is None:
            return None
        command = self._active.command
        self._active = None
        self.completed += 1
        self._record(now, command, note)
        return command

    def cancel(self, now: float, reason: str = "cancelled") -> MotionCommand | None:
        """Abort the active command."""
        if self._active is None:
            return None
        command = self._active.command
        self._active = None
        self._record(now, command, reason)
        return command

    def check_timeout(self, now: float) -> MotionCommand | None:
        """Cancel and return the active command if it has exceeded its timeout.

        A motion that never finishes usually means the hand is jammed or a servo
        has failed. Detecting it here means the fault surfaces as a diagnosable
        event instead of a hand that quietly stops responding.
        """
        if self._active is None:
            return None
        command = self._active.command
        if command.timeout_s <= 0:
            return None
        if now - self._active.started_at < command.timeout_s:
            return None
        self._active = None
        self.timed_out += 1
        self._record(now, command, "timed out")
        return command

    def elapsed(self, now: float) -> float:
        return now - self._active.started_at if self._active else 0.0

    # -- introspection --------------------------------------------------------

    def history(self, limit: int = 10) -> list[tuple[float, MotionCommand, str]]:
        return list(self._history)[-limit:]

    def stats(self) -> dict[str, int]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "preempted": self.preempted,
            "completed": self.completed,
            "timed_out": self.timed_out,
        }

    def clear(self, now: float) -> None:
        """Drop everything (e-stop, mode change)."""
        self.cancel(now, "queue cleared")
