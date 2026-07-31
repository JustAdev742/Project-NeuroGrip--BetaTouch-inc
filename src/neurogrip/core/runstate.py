"""Unclean-shutdown detection.

A prosthetic hand loses power without warning: the battery is disconnected, a
cell sags under a stall, the user takes the socket off. Most of the time that is
harmless. Sometimes it is the *symptom* — the process crashed, the watchdog
fired, the controller browned out mid-grasp — and the next startup is the only
opportunity anyone has to notice.

Nothing here tries to reconstruct what the hand was doing. Resuming a grasp
across a crash would be exactly wrong: the hand is a limb, the user's arm has
moved, and whatever was in front of the camera is gone. What this does is much
smaller and more useful:

* record that a run started, and what it was doing when it last checkpointed;
* notice at the next startup that the previous run never recorded an ending;
* make the next run **more conservative**, not less — AI assistance stays off
  until the user asks for it, so a boot loop cannot repeatedly re-enter the
  state that caused the crash.

The marker is written to the same ``var/`` directory as the other runtime state
and is deliberately tiny: a partially written marker must still parse, so it is
one JSON object rewritten atomically rather than an append log.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .logging import get_logger

__all__ = ["RunMarker", "RunRecord", "ShutdownReason"]

log = get_logger(__name__)


class ShutdownReason(str, Enum):
    """How the previous run ended."""

    #: Stopped through the normal shutdown path.
    CLEAN = "clean"
    #: A marker was found, so the previous run never recorded an ending.
    UNCLEAN = "unclean"
    #: No marker at all — first run, or the state directory was cleared.
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class RunRecord:
    """What the previous run was doing when it last checkpointed."""

    reason: ShutdownReason = ShutdownReason.UNKNOWN
    version: str = ""
    pid: int = 0
    started_at: float = 0.0
    #: Wall-clock time of the last checkpoint, not of the crash. The gap between
    #: the two is bounded by the checkpoint interval.
    last_seen_at: float = 0.0
    state: str = ""
    mode: str = ""
    #: True if the hand was executing a motion at the last checkpoint. The one
    #: fact worth surfacing: a crash mid-grasp means the drive was live.
    moving: bool = False
    estop: bool = False
    notes: str = ""

    @property
    def crashed(self) -> bool:
        return self.reason is ShutdownReason.UNCLEAN

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.last_seen_at) if self.last_seen_at else 0.0

    def describe(self) -> str:
        if self.reason is ShutdownReason.UNKNOWN:
            return "no record of a previous run"
        if self.reason is ShutdownReason.CLEAN:
            return "previous run shut down cleanly"
        detail = f"previous run (pid {self.pid}) ended without shutting down"
        if self.state:
            detail += f" while {self.state}"
        if self.moving:
            detail += ", with the hand in motion"
        if self.estop:
            detail += ", after an emergency stop"
        return detail


@dataclass(slots=True)
class RunMarker:
    """Writes and reads the run marker.

    Usage is three calls: :meth:`begin` at startup (which returns what the
    *previous* run left behind), :meth:`checkpoint` periodically, and
    :meth:`finish` on a clean shutdown.
    """

    path: Path = field(default_factory=lambda: Path("var/run-state.json"))
    version: str = ""
    _started_at: float = 0.0
    _previous: RunRecord = field(default_factory=RunRecord)

    def __post_init__(self) -> None:
        self.path = Path(self.path)

    @property
    def previous(self) -> RunRecord:
        """What the last run left behind. Meaningful only after :meth:`begin`."""
        return self._previous

    def begin(self) -> RunRecord:
        """Claim the marker and report how the previous run ended."""
        self._previous = self._read()
        self._started_at = time.time()
        self.checkpoint()
        if self._previous.crashed:
            log.warning(
                "unclean shutdown detected",
                detail=self._previous.describe(),
                previous_pid=self._previous.pid,
                seconds_ago=round(self._previous.age_s),
            )
        return self._previous

    def checkpoint(
        self,
        *,
        state: str = "",
        mode: str = "",
        moving: bool = False,
        estop: bool = False,
    ) -> None:
        """Record that this run is still alive, with a little context.

        Called from the diagnostics group, so the marker is at most one
        diagnostics period behind reality. Never raises: an unwritable state
        directory must not stop the hand from working.
        """
        record = {
            "reason": ShutdownReason.UNCLEAN.value,
            "version": self.version,
            "pid": os.getpid(),
            "started_at": self._started_at,
            "last_seen_at": time.time(),
            "state": state,
            "mode": mode,
            "moving": moving,
            "estop": estop,
        }
        self._write(record)

    def finish(self, notes: str = "") -> None:
        """Record a clean shutdown."""
        self._write(
            {
                "reason": ShutdownReason.CLEAN.value,
                "version": self.version,
                "pid": os.getpid(),
                "started_at": self._started_at,
                "last_seen_at": time.time(),
                "notes": notes,
            }
        )

    # -- storage --------------------------------------------------------------

    def _write(self, record: dict) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(record), encoding="utf-8")
            temporary.replace(self.path)
        except OSError as exc:
            log.throttled(
                "runstate-write",
                "warning",
                "could not write the run marker",
                now=time.monotonic(),
                error=str(exc),
            )

    def _read(self) -> RunRecord:
        if not self.path.exists():
            return RunRecord(reason=ShutdownReason.UNKNOWN)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt marker is itself evidence of an interrupted write, which
            # only happens if the previous run died at an awkward moment.
            log.warning("run marker is unreadable; treating as unclean", error=str(exc))
            return RunRecord(reason=ShutdownReason.UNCLEAN, notes=f"unreadable marker: {exc}")
        try:
            reason = ShutdownReason(data.get("reason", "unknown"))
        except ValueError:
            reason = ShutdownReason.UNCLEAN
        return RunRecord(
            reason=reason,
            version=str(data.get("version", "")),
            pid=int(data.get("pid", 0)),
            started_at=float(data.get("started_at", 0.0)),
            last_seen_at=float(data.get("last_seen_at", 0.0)),
            state=str(data.get("state", "")),
            mode=str(data.get("mode", "")),
            moving=bool(data.get("moving", False)),
            estop=bool(data.get("estop", False)),
            notes=str(data.get("notes", "")),
        )
