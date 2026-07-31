"""Replay a recorded EMG session as if it were a live source.

Recorded sessions are the backbone of EMG development: a user's real signals can
be captured once and then replayed through the *entire* pipeline — filters,
calibration, classifier, intent engine, fusion — every time the code changes.
That turns "does the classifier still recognise Alex's grasp?" into a test.

Recordings use the format written by :class:`neurogrip.emg.recorder.EmgRecorder`
(see ``docs/emg.md``): a JSON header line followed by newline-delimited CSV rows
of ``timestamp,ch0,ch1,...``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from ...core.clock import Clock, RealClock
from ...core.errors import DeviceNotAvailableError
from ..base import DeviceCapability, DeviceInfo, DeviceKind
from .base import EmgChannelSpec, EmgSample

__all__ = ["ReplayEmgSource", "load_recording"]


def load_recording(path: Path | str) -> tuple[dict, list[EmgSample], tuple[EmgChannelSpec, ...]]:
    """Load a recording file into memory.

    Returns ``(header, samples, channels)``. Raises
    :class:`~neurogrip.core.errors.DeviceNotAvailableError` when the file is
    missing or malformed, so a bad recording degrades like a missing device
    rather than crashing the application.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise DeviceNotAvailableError(f"recording not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            header_line = handle.readline()
            header = json.loads(header_line)
            channels = tuple(
                EmgChannelSpec(
                    index=spec.get("index", i),
                    name=spec.get("name", f"ch{i}"),
                    role=spec.get("role", "auxiliary"),
                    site=spec.get("site", ""),
                )
                for i, spec in enumerate(header.get("channels", []))
            )
            samples: list[EmgSample] = []
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                # Rows are "timestamp,ch0,…,chN[,label]". The label is optional
                # and is not part of the sample, so take exactly the channels
                # the header declared and ignore anything after them.
                count = len(channels) if channels else len(parts) - 1
                samples.append(
                    EmgSample(
                        timestamp=float(parts[0]),
                        values=tuple(float(v) for v in parts[1 : 1 + count]),
                    )
                )
    except (json.JSONDecodeError, ValueError, IndexError) as exc:
        raise DeviceNotAvailableError(
            f"malformed EMG recording: {exc}", context={"path": str(file_path)}
        ) from exc

    if not channels and samples:
        channels = tuple(
            EmgChannelSpec(index=i, name=f"ch{i}") for i in range(len(samples[0].values))
        )
    return header, samples, channels


class ReplayEmgSource:
    """EMG source that plays back a recorded session in real time."""

    def __init__(
        self,
        path: Path | str,
        clock: Clock | None = None,
        *,
        loop: bool = False,
        speed: float = 1.0,
    ) -> None:
        self._path = Path(path)
        self._clock = clock or RealClock()
        self._loop = loop
        self._speed = max(0.01, speed)
        self._header: dict = {}
        self._samples: list[EmgSample] = []
        self._channels: tuple[EmgChannelSpec, ...] = ()
        self._cursor = 0
        self._start_wall = 0.0
        self._start_sample = 0.0
        self._open = False
        self._passes = 0

    def open(self) -> None:
        self._header, self._samples, self._channels = load_recording(self._path)
        if not self._samples:
            raise DeviceNotAvailableError(f"recording contains no samples: {self._path}")
        self._cursor = 0
        self._passes = 0
        self._start_wall = self._clock.monotonic()
        self._start_sample = self._samples[0].timestamp
        self._open = True

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="emg",
            kind=DeviceKind.EMG,
            driver="replay",
            connection=str(self._path),
            capabilities=frozenset({DeviceCapability.SIMULATED}),
            extra={
                "samples": len(self._samples),
                "duration_s": round(self.duration, 3),
                "subject": self._header.get("subject", ""),
                "recorded_at": self._header.get("recorded_at", ""),
            },
        )

    @property
    def sample_rate_hz(self) -> float:
        return float(self._header.get("sample_rate_hz", 1000.0))

    @property
    def channels(self) -> Sequence[EmgChannelSpec]:
        return self._channels

    @property
    def duration(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1].timestamp - self._samples[0].timestamp

    @property
    def progress(self) -> float:
        """Fraction of the recording played, in ``[0, 1]``."""
        return self._cursor / len(self._samples) if self._samples else 1.0

    @property
    def finished(self) -> bool:
        return not self._loop and self._cursor >= len(self._samples)

    def dropped_samples(self) -> int:
        return 0

    def read(self) -> list[EmgSample]:
        """Return every sample whose recorded time has now elapsed."""
        if not self._open or not self._samples:
            return []

        elapsed = (self._clock.monotonic() - self._start_wall) * self._speed
        horizon = self._start_sample + elapsed
        out: list[EmgSample] = []

        while self._cursor < len(self._samples):
            sample = self._samples[self._cursor]
            if sample.timestamp > horizon:
                break
            out.append(sample)
            self._cursor += 1

        if self._cursor >= len(self._samples) and self._loop:
            self._cursor = 0
            self._passes += 1
            self._start_wall = self._clock.monotonic()
        return out

    def seek(self, fraction: float) -> None:
        """Jump to a position in the recording (used by the replay UI scrubber)."""
        if not self._samples:
            return
        self._cursor = max(0, min(len(self._samples) - 1, int(fraction * len(self._samples))))
        self._start_wall = self._clock.monotonic()
        self._start_sample = self._samples[self._cursor].timestamp

    def __iter__(self) -> Iterator[EmgSample]:
        """Iterate the whole recording ignoring timing (offline analysis)."""
        return iter(self._samples)
