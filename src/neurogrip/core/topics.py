"""Canonical event topic names.

Topics are hierarchical, dot-separated strings. Subscribers may use a trailing
``*`` wildcard (``"emg.*"``). Declaring them centrally means a typo is an
``AttributeError`` at import time rather than a subscription that silently never
fires — a failure mode that is genuinely dangerous in a device where the UI's
"connection lost" banner is driven by an event.
"""

from __future__ import annotations

__all__ = ["Topics"]


class Topics:
    """Namespace of every topic published on the event bus."""

    # -- lifecycle ------------------------------------------------------------
    SYSTEM_STARTING = "system.starting"
    SYSTEM_READY = "system.ready"
    SYSTEM_STOPPING = "system.stopping"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"

    # -- service lifecycle -----------------------------------------------------
    # Published by the service manager. Separate from SYSTEM_* because these
    # describe one subsystem's lifecycle, not the whole runtime's: a service
    # restarting is routine, the system stopping is not.
    SERVICE_STARTED = "service.started"
    SERVICE_STOPPED = "service.stopped"
    SERVICE_HEALTH = "service.health"
    SERVICE_ERROR = "service.error"
    SERVICE_RESTARTED = "service.restarted"

    # -- hardware availability -------------------------------------------------
    # The tri-state (detected / missing / disabled) that gates operational mode.
    HARDWARE_SCANNED = "hardware.scanned"
    HARDWARE_MISSING = "hardware.missing"
    HARDWARE_READY = "hardware.ready"

    # -- EMG ------------------------------------------------------------------
    EMG_FRAME = "emg.frame"
    EMG_QUALITY = "emg.quality"
    EMG_CALIBRATION_STARTED = "emg.calibration.started"
    EMG_CALIBRATION_STEP = "emg.calibration.step"
    EMG_CALIBRATION_COMPLETE = "emg.calibration.complete"
    EMG_RECALIBRATED = "emg.recalibrated"

    # -- intent ---------------------------------------------------------------
    INTENT_UPDATED = "intent.updated"
    INTENT_GESTURE = "intent.gesture"
    INTENT_CANCEL = "intent.cancel"

    # -- vision ---------------------------------------------------------------
    VISION_RESULT = "vision.result"
    VISION_BACKEND_CHANGED = "vision.backend.changed"
    VISION_ERROR = "vision.error"
    CAMERA_FRAME = "camera.frame"

    # -- AI / fusion ----------------------------------------------------------
    GRASP_PLANNED = "ai.grasp.planned"
    DECISION_MADE = "fusion.decision"
    DECISION_REJECTED = "fusion.rejected"

    # -- control --------------------------------------------------------------
    HAND_STATE = "control.hand_state"
    MOTION_STARTED = "control.motion.started"
    MOTION_COMPLETED = "control.motion.completed"
    MOTION_CANCELLED = "control.motion.cancelled"
    GRIP_CONTACT = "control.grip.contact"
    GRIP_SLIP = "control.grip.slip"

    # -- safety ---------------------------------------------------------------
    SAFETY_STATE = "safety.state"
    SAFETY_FAULT_RAISED = "safety.fault.raised"
    SAFETY_FAULT_CLEARED = "safety.fault.cleared"
    ESTOP_ENGAGED = "safety.estop.engaged"
    ESTOP_RELEASED = "safety.estop.released"
    WATCHDOG_EXPIRED = "safety.watchdog.expired"

    # -- modes ----------------------------------------------------------------
    MODE_CHANGED = "mode.changed"
    MODE_REJECTED = "mode.rejected"

    # -- training -------------------------------------------------------------
    TRAINING_SESSION_STARTED = "training.session.started"
    TRAINING_SESSION_ENDED = "training.session.ended"
    TRAINING_TRIAL = "training.trial"
    TRAINING_ACHIEVEMENT = "training.achievement"

    # -- diagnostics / telemetry ----------------------------------------------
    DIAGNOSTICS_REPORT = "diagnostics.report"
    SELFTEST_RESULT = "diagnostics.selftest"
    METRIC_SAMPLE = "diagnostics.metric"
    LOG_RECORD = "log.record"

    # -- UI -------------------------------------------------------------------
    UI_NAVIGATE = "ui.navigate"
    UI_NOTIFICATION = "ui.notification"
    UI_ACTION = "ui.action"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """Every declared topic — used by the debug console's ``topics`` command."""
        return tuple(
            sorted(
                value
                for name, value in vars(cls).items()
                if not name.startswith("_") and isinstance(value, str)
            )
        )
