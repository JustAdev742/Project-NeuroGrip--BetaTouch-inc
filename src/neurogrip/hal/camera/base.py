"""Camera abstraction and frame container.

Frames carry raw bytes plus a pixel-format tag rather than a NumPy array, so the
HAL has no hard dependency on NumPy and the same :class:`Frame` can come from a
V4L2 device, a Pi camera, a synthetic scene generator or a video file. Backends
that want an array call :meth:`Frame.as_array`, which converts lazily and only if
NumPy is installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..base import DeviceInfo

__all__ = ["CameraSettings", "CameraSource", "Frame", "PixelFormat"]


class PixelFormat(str, Enum):
    """Supported buffer layouts."""

    GRAY8 = "gray8"
    RGB888 = "rgb888"
    BGR888 = "bgr888"
    #: Compressed; ``Frame.data`` holds a complete JPEG file.
    JPEG = "jpeg"

    @property
    def channels(self) -> int:
        if self is PixelFormat.GRAY8:
            return 1
        if self is PixelFormat.JPEG:
            return 0  # variable / not applicable
        return 3


@dataclass(frozen=True, slots=True)
class Frame:
    """A single captured image."""

    width: int
    height: int
    pixel_format: PixelFormat
    data: bytes
    timestamp: float
    index: int = 0
    #: Optional per-frame metadata: exposure, focus distance, and — for the
    #: simulated camera — the ground-truth scene description (see
    #: ``neurogrip.hal.camera.simulated``).
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def pixel(self, x: int, y: int) -> tuple[int, int, int]:
        """Read one pixel as RGB. Raises for compressed formats."""
        if self.pixel_format is PixelFormat.JPEG:
            raise ValueError("cannot index into a compressed frame")
        channels = self.pixel_format.channels
        offset = (y * self.width + x) * channels
        if self.pixel_format is PixelFormat.GRAY8:
            value = self.data[offset]
            return (value, value, value)
        r, g, b = self.data[offset], self.data[offset + 1], self.data[offset + 2]
        if self.pixel_format is PixelFormat.BGR888:
            return (b, g, r)
        return (r, g, b)

    def as_array(self) -> Any:
        """Return a ``numpy`` view of the frame; raises if NumPy is unavailable.

        Only vision backends call this, and only after checking that their
        runtime dependency is installed.
        """
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - exercised only without numpy
            raise RuntimeError("numpy is required for array access; install the 'vision' extra") from exc
        if self.pixel_format is PixelFormat.JPEG:
            raise ValueError("decode the JPEG before requesting an array")
        channels = self.pixel_format.channels
        array = np.frombuffer(self.data, dtype=np.uint8)
        return array.reshape((self.height, self.width, channels)) if channels > 1 else array.reshape(
            (self.height, self.width)
        )


@dataclass(frozen=True, slots=True)
class CameraSettings:
    """Requested capture configuration."""

    width: int = 640
    height: int = 480
    fps: float = 30.0
    pixel_format: PixelFormat = PixelFormat.RGB888
    #: ``None`` means "let the driver decide" (auto exposure / auto white balance).
    exposure_us: int | None = None
    gain: float | None = None
    autofocus: bool = True


@runtime_checkable
class CameraSource(Protocol):
    """Frame source."""

    def open(self) -> None: ...

    def close(self) -> None: ...

    @property
    def is_open(self) -> bool: ...

    def info(self) -> DeviceInfo: ...

    @property
    def settings(self) -> CameraSettings: ...

    def read(self) -> Frame | None:
        """Return the newest frame, or ``None`` if none is ready.

        Must not block: the vision service polls this from its own rate group and
        a blocking grab would couple camera latency into the scheduler.
        """
        ...

    def dropped_frames(self) -> int:
        """Frames produced by the sensor but never delivered — a health signal."""
        ...
