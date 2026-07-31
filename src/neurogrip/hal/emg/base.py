"""EMG source abstraction.

An :class:`EmgSource` delivers *raw* samples. Every bit of signal conditioning —
filtering, rectification, envelope extraction, normalisation — happens in
:mod:`neurogrip.emg`, above the HAL. That separation is what allows a MyoWare
board (which supplies a hardware envelope) and a research-grade differential
amplifier (which supplies raw ±2 mV signals) to feed the exact same pipeline.

Sources are pull-based: the EMG service asks for whatever samples have
accumulated since the last call. Push-based drivers wrap their callback in an
internal queue and drain it here, so the control loop never runs inside a
driver's interrupt or thread context.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..base import DeviceInfo

__all__ = ["EmgChannelSpec", "EmgSample", "EmgSource"]


@dataclass(frozen=True, slots=True)
class EmgSample:
    """One multi-channel sample.

    ``values`` are in volts at the electrode (post-amplifier, pre-filter). Using
    physical units rather than raw ADC counts means calibration data stays valid
    when the ADC or its gain changes.
    """

    timestamp: float
    values: tuple[float, ...]

    def __getitem__(self, channel: int) -> float:
        return self.values[channel]

    def __len__(self) -> int:
        return len(self.values)


@dataclass(frozen=True, slots=True)
class EmgChannelSpec:
    """Describes what a physical channel is measuring.

    The ``role`` is what the gesture classifier keys off: it needs to know which
    electrode sits over the flexor group and which over the extensor group. Roles
    are configuration, not code, so a different electrode placement is a config
    change.
    """

    index: int
    name: str
    #: One of ``flexor``, ``extensor``, ``auxiliary``.
    role: str = "auxiliary"
    #: Full-scale range of the front end, used for saturation detection.
    full_scale_v: float = 3.3
    #: Anatomical site, shown in the calibration wizard and muscle visualisation.
    site: str = ""

    @property
    def is_flexor(self) -> bool:
        return self.role == "flexor"

    @property
    def is_extensor(self) -> bool:
        return self.role == "extensor"


@runtime_checkable
class EmgSource(Protocol):
    """Raw multi-channel EMG acquisition device."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    @property
    def is_open(self) -> bool: ...

    def info(self) -> DeviceInfo: ...

    @property
    def sample_rate_hz(self) -> float:
        """Nominal sampling rate. Must be at least twice the highest EMG band edge."""
        ...

    @property
    def channels(self) -> Sequence[EmgChannelSpec]:
        """Channel descriptors in acquisition order."""
        ...

    def read(self) -> list[EmgSample]:
        """Return samples accumulated since the previous call.

        Must never block. An empty list means "no new data", which is normal when
        polled faster than the sample rate.
        """
        ...

    def dropped_samples(self) -> int:
        """Total samples lost to buffer overrun — a hard quality signal."""
        ...


@dataclass(slots=True)
class EmgSourceStats:
    """Bookkeeping shared by concrete sources (composition, not inheritance)."""

    samples_read: int = 0
    dropped: int = 0
    last_timestamp: float = 0.0
    overruns: int = 0
    errors: list[str] = field(default_factory=list)

    def note_error(self, message: str, limit: int = 20) -> None:
        self.errors.append(message)
        if len(self.errors) > limit:
            del self.errors[0 : len(self.errors) - limit]
