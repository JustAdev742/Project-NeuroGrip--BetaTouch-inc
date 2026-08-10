"""EMG session recording and automatic re-calibration.

Recording
---------
:class:`EmgRecorder` writes raw samples in the format understood by
:class:`~neurogrip.hal.emg.replay.ReplayEmgSource`: a JSON header line followed by
CSV rows. Deliberately boring — a recording made today must still open in five
years, with ``head`` if necessary.

Labels can be attached while recording (the training exercises do this
automatically), producing the supervised dataset needed to fit a per-user gesture
model.

Auto re-calibration
-------------------
Surface EMG drifts within a single session: electrodes warm up, gel dries, the
skin sweats, the muscle fatigues. :class:`AutoRecalibrator` watches for sustained
rest periods and gently updates the rest baseline, so the onset threshold tracks
the drift instead of the user having to keep re-calibrating.

It only ever adjusts the *rest baseline*, never the MVC. Lowering the effort
needed to trigger a grasp based on unlabelled data would be a safety change made
without the user's knowledge; raising the noise floor when the environment gets
noisier is the conservative direction and is safe.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from ..core.clock import Clock
from ..core.logging import get_logger
from ..core.ringbuffer import RunningStats
from ..hal.emg.base import EmgChannelSpec, EmgSample
from .calibration import EmgCalibration
from .pipeline import EmgFrame

__all__ = ["AutoRecalibrator", "EmgRecorder", "RecalibrationEvent", "RecordingInfo"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RecordingInfo:
    """Summary of a finished recording."""

    path: Path
    samples: int
    duration_s: float
    channels: int
    labels: tuple[str, ...] = ()


class EmgRecorder:
    """Streams raw EMG samples to disk.

    Writes incrementally rather than buffering the session in RAM: a 10-minute
    two-channel recording at 1 kHz is 1.2 M samples, which is not something an
    embedded target should hold in memory to be tidy.
    """

    def __init__(
        self,
        path: Path | str,
        channels: Sequence[EmgChannelSpec],
        *,
        sample_rate_hz: float = 1000.0,
        subject: str = "default",
        notes: str = "",
    ) -> None:
        self._path = Path(path)
        self._channels = tuple(channels)
        self._rate = sample_rate_hz
        self._subject = subject
        self._notes = notes
        self._handle: TextIO | None = None
        self._count = 0
        self._first_timestamp: float | None = None
        self._last_timestamp = 0.0
        self._label = ""
        self._labels: set[str] = set()

    def start(self) -> None:
        """Open the file and write the header."""
        if self._handle is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", encoding="utf-8")
        header = {
            "format": "neurogrip-emg-1",
            "subject": self._subject,
            "recorded_at": time.time(),
            "sample_rate_hz": self._rate,
            "notes": self._notes,
            "channels": [
                {"index": c.index, "name": c.name, "role": c.role, "site": c.site}
                for c in self._channels
            ],
        }
        self._handle.write(json.dumps(header) + "\n")
        self._handle.write("# timestamp," + ",".join(c.name for c in self._channels) + ",label\n")
        log.info("recording started", path=str(self._path))

    def set_label(self, label: str) -> None:
        """Tag subsequent samples (used to build supervised training sets)."""
        self._label = label
        if label:
            self._labels.add(label)

    def write(self, samples: Sequence[EmgSample]) -> None:
        """Append a batch of samples."""
        if self._handle is None or not samples:
            return
        lines = []
        for sample in samples:
            if self._first_timestamp is None:
                self._first_timestamp = sample.timestamp
            self._last_timestamp = sample.timestamp
            values = ",".join(f"{v:.9g}" for v in sample.values)
            lines.append(f"{sample.timestamp:.6f},{values},{self._label}")
        self._handle.write("\n".join(lines) + "\n")
        self._count += len(samples)

    def stop(self) -> RecordingInfo:
        """Close the file and return a summary."""
        handle, self._handle = self._handle, None
        if handle is not None:
            handle.flush()
            handle.close()
        duration = (
            self._last_timestamp - self._first_timestamp if self._first_timestamp is not None else 0.0
        )
        info = RecordingInfo(
            path=self._path,
            samples=self._count,
            duration_s=duration,
            channels=len(self._channels),
            labels=tuple(sorted(self._labels)),
        )
        log.info("recording stopped", path=str(self._path), samples=self._count)
        return info

    @property
    def is_recording(self) -> bool:
        return self._handle is not None

    @property
    def sample_count(self) -> int:
        return self._count

    def __enter__(self) -> EmgRecorder:
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


@dataclass(frozen=True, slots=True)
class RecalibrationEvent:
    """Emitted when a channel's rest baseline is updated."""

    channel: int
    old_rest: float
    new_rest: float
    samples: int
    timestamp: float

    @property
    def drift_ratio(self) -> float:
        return self.new_rest / self.old_rest if self.old_rest > 1e-12 else 1.0


class AutoRecalibrator:
    """Tracks baseline drift during quiet periods and updates calibration."""

    def __init__(
        self,
        calibration: EmgCalibration,
        clock: Clock,
        *,
        rest_window_s: float = 3.0,
        min_interval_s: float = 60.0,
        max_adjust_ratio: float = 2.5,
        blend: float = 0.3,
    ) -> None:
        self._calibration = calibration
        self._clock = clock
        #: Continuous rest required before an update is considered.
        self._rest_window = rest_window_s
        #: Minimum gap between updates, so the baseline cannot chase itself.
        self._min_interval = min_interval_s
        #: Refuse adjustments larger than this ratio — that is a fault, not drift.
        self._max_adjust = max_adjust_ratio
        #: Fraction of the new measurement blended in (exponential smoothing).
        self._blend = blend

        self._rest_since: float | None = None
        self._stats: dict[int, RunningStats] = {}
        self._last_update = -1e18

    @property
    def calibration(self) -> EmgCalibration:
        return self._calibration

    def set_calibration(self, calibration: EmgCalibration) -> None:
        self._calibration = calibration
        self._reset_window()

    def update(self, frame: EmgFrame) -> list[RecalibrationEvent]:
        """Feed a processed frame; returns any recalibration events produced."""
        now = frame.timestamp

        if not frame.is_resting:
            self._reset_window()
            return []

        if self._rest_since is None:
            self._rest_since = now
            self._stats = {c.index: RunningStats() for c in frame.channels}

        for channel in frame.channels:
            self._stats[channel.index].add(channel.envelope)

        if now - self._rest_since < self._rest_window:
            return []
        if now - self._last_update < self._min_interval:
            return []

        events: list[RecalibrationEvent] = []
        for channel in frame.channels:
            stats = self._stats[channel.index]
            if stats.count < 20:
                continue
            current = self._calibration.get(channel.index)
            measured = stats.mean
            if current.rest_mean > 1e-12:
                ratio = measured / current.rest_mean
                if ratio > self._max_adjust or ratio < 1.0 / self._max_adjust:
                    # A tenfold change is a detached electrode or a hardware
                    # fault. Recalibrating around it would hide the problem.
                    log.warning(
                        "rest baseline changed implausibly; not auto-recalibrating",
                        channel=channel.index,
                        ratio=round(ratio, 2),
                    )
                    continue

            blended = current.rest_mean * (1 - self._blend) + measured * self._blend
            blended_std = current.rest_std * (1 - self._blend) + max(stats.std, 1e-7) * self._blend
            self._calibration.set(current.with_rest(blended, blended_std, stats.count, time.time()))
            events.append(
                RecalibrationEvent(
                    channel=channel.index,
                    old_rest=current.rest_mean,
                    new_rest=blended,
                    samples=stats.count,
                    timestamp=now,
                )
            )

        if events:
            self._calibration.updated_at = time.time()
            self._last_update = now
            log.info(
                "auto-recalibrated rest baseline",
                channels=[e.channel for e in events],
                drift={e.channel: round(e.drift_ratio, 3) for e in events},
            )
        self._reset_window()
        return events

    def _reset_window(self) -> None:
        self._rest_since = None
        self._stats = {}
