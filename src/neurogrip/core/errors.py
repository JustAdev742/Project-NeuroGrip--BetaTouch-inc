"""Exception hierarchy.

A flat ``Exception`` soup makes it impossible for the safety monitor to decide
whether a failure is recoverable. Every error raised inside NeuroGrip derives from
:class:`NeuroGripError` and declares a :attr:`~NeuroGripError.severity`, which the
safety layer maps onto an action (log / degrade / fall back to manual / stop).
"""

from __future__ import annotations

from enum import IntEnum

__all__ = [
    "CalibrationError",
    "CommunicationError",
    "ConfigurationError",
    "DeviceError",
    "DeviceNotAvailableError",
    "EmergencyStopActive",
    "ModeTransitionError",
    "ModelLoadError",
    "NeuroGripError",
    "PlanningError",
    "ProtocolError",
    "SafetyViolation",
    "ServiceError",
    "Severity",
    "VisionError",
]


class Severity(IntEnum):
    """How bad a failure is, from the safety layer's point of view."""

    #: Informational; the operation can be retried transparently.
    MINOR = 10
    #: A feature is unavailable; degrade gracefully (e.g. vision offline).
    DEGRADED = 20
    #: Assistive features must be disabled; fall back to direct manual control.
    FALLBACK = 30
    #: Motion must stop immediately and stay stopped until acknowledged.
    CRITICAL = 40


class NeuroGripError(Exception):
    """Base class for every error raised by this stack."""

    severity: Severity = Severity.MINOR

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        #: Structured detail attached to log records and the black-box recorder.
        self.context: dict[str, object] = dict(context or {})

    def __str__(self) -> str:
        if not self.context:
            return self.message
        detail = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} ({detail})"


class ConfigurationError(NeuroGripError):
    """Invalid, missing or contradictory configuration.

    Raised at startup only. A misconfigured device must refuse to start rather
    than run with guessed limits.
    """

    severity = Severity.CRITICAL


class ServiceError(NeuroGripError):
    """A runtime service failed to start, stop, or tick."""

    severity = Severity.FALLBACK


class DeviceError(NeuroGripError):
    """A hardware device reported or caused a failure."""

    severity = Severity.FALLBACK


class DeviceNotAvailableError(DeviceError):
    """A device could not be opened (missing, busy, or no driver installed).

    Degraded rather than fallback: the composition root substitutes a simulated
    or null implementation and continues, which is what allows the hand to keep
    working manually when the camera is unplugged.
    """

    severity = Severity.DEGRADED


class CommunicationError(DeviceError):
    """Link-level failure talking to the motor controller (timeout, CRC, reset)."""

    severity = Severity.FALLBACK


class ProtocolError(CommunicationError):
    """A well-formed frame carried an unexpected or unsupported payload."""

    severity = Severity.DEGRADED


class CalibrationError(NeuroGripError):
    """Calibration data is missing, stale or implausible."""

    severity = Severity.DEGRADED


class VisionError(NeuroGripError):
    """The vision pipeline failed for a frame or a sequence of frames."""

    severity = Severity.DEGRADED


class ModelLoadError(VisionError):
    """A model file is missing, corrupt, or its runtime is not installed."""

    severity = Severity.DEGRADED


class PlanningError(NeuroGripError):
    """The grasp planner could not produce a usable plan."""

    severity = Severity.DEGRADED


class ModeTransitionError(NeuroGripError):
    """A mode change was requested that the mode manager refuses to perform."""

    severity = Severity.MINOR


class SafetyViolation(NeuroGripError):
    """A command would breach a hard safety limit and was rejected."""

    severity = Severity.CRITICAL


class EmergencyStopActive(SafetyViolation):
    """Motion was requested while the emergency stop latch is engaged."""

    severity = Severity.CRITICAL
