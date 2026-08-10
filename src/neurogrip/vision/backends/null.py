"""Null vision backend — "no vision available".

Used when the camera is absent, vision is disabled in configuration, or a
requested backend could not be constructed. It always returns an empty result.

This is not a placeholder to be filled in later; it is a functional part of the
safety design. Because the decision-fusion layer treats "no vision" as a legal
state (assistance degrades, direct control continues), having a real object that
represents it removes every ``if backend is None`` branch from the rest of the
codebase.
"""

from __future__ import annotations

from ...core.config import Config
from ...hal.camera.base import Frame
from ..backend import BackendInfo, register_backend
from ..types import VisionCapability, VisionResult

__all__ = ["NullVisionBackend"]


class NullVisionBackend:
    """Backend that produces nothing, successfully."""

    def __init__(self, reason: str = "vision disabled") -> None:
        self._reason = reason

    def initialize(self) -> None:
        """Nothing to load."""

    def shutdown(self) -> None:
        """Nothing to release."""

    @property
    def capabilities(self) -> VisionCapability:
        return VisionCapability.NONE

    def info(self) -> BackendInfo:
        return BackendInfo(
            name="null",
            version="1.0",
            capabilities=VisionCapability.NONE,
            runtime="none",
            degraded_reason=self._reason,
        )

    def process(self, frame: Frame) -> VisionResult:
        # Empty but *not* an error: "I looked and saw nothing to report" is a
        # valid answer, and marking it as an error would trip the safety layer.
        return VisionResult(
            timestamp=frame.timestamp,
            frame_index=frame.index,
            backend="null",
            capabilities=VisionCapability.NONE,
        )


def _factory(config: Config | None = None, **kwargs: object) -> NullVisionBackend:
    reason = str(kwargs.get("reason", "vision disabled"))
    return NullVisionBackend(reason)


register_backend("null", _factory)
