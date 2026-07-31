"""Sensor services: the periodic tasks that feed the decision layer.

Two small services live here because they are pure glue between the HAL and the
processing layers, and giving them their own modules would add navigation cost
without adding structure:

* :class:`EmgService` — drains the EMG source, runs the pipeline, updates the
  intent engine, handles calibration and auto-recalibration, publishes frames.
* :class:`VisionService` — drives the vision pipeline at its own rate.

Both kick their watchdogs on every successful cycle. That is what turns "the
sensor thread stopped" into a detected fault rather than a hand that mysteriously
stops responding.
"""

from __future__ import annotations

from ..core.clock import Clock
from ..core.events import EventBus
from ..core.lifecycle import HealthReport, ServiceBase
from ..core.logging import get_logger
from ..core.topics import Topics
from ..emg.calibration import CalibrationWizard
from ..emg.intent import IntentEngine, IntentEstimate
from ..emg.pipeline import EmgFrame, EmgPipeline
from ..emg.quality import SignalQuality
from ..emg.recorder import AutoRecalibrator, EmgRecorder
from ..hal.emg.base import EmgSource
from ..safety.watchdog import WatchdogGroup
from ..vision.pipeline import VisionPipeline
from ..vision.types import VisionResult

__all__ = ["EmgService", "VisionService"]

log = get_logger(__name__)


class EmgService(ServiceBase):
    """Acquisition → pipeline → intent, once per cycle."""

    service_name = "emg"

    def __init__(
        self,
        source: EmgSource,
        pipeline: EmgPipeline,
        intent_engine: IntentEngine,
        clock: Clock,
        bus: EventBus,
        watchdogs: WatchdogGroup,
        *,
        wizard: CalibrationWizard | None = None,
        recalibrator: AutoRecalibrator | None = None,
        calibration_path: str | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._pipeline = pipeline
        self._intent = intent_engine
        self._clock = clock
        self._bus = bus
        self._watchdogs = watchdogs
        self._wizard = wizard
        self._recalibrator = recalibrator
        self._calibration_path = calibration_path
        self._recorder: EmgRecorder | None = None

        self._latest_frame: EmgFrame | None = None
        self._latest_intent: IntentEstimate | None = None
        self._last_dropped = 0
        self.cycles = 0
        self.empty_cycles = 0

    # -- lifecycle ------------------------------------------------------------

    def on_start(self) -> None:
        self._source.open()
        log.info("EMG service started", device=str(self._source.info()))

    def on_stop(self) -> None:
        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.stop()
        self._source.close()

    # -- accessors ------------------------------------------------------------

    @property
    def frame(self) -> EmgFrame | None:
        return self._latest_frame

    @property
    def intent(self) -> IntentEstimate | None:
        return self._latest_intent

    @property
    def wizard(self) -> CalibrationWizard | None:
        return self._wizard

    @property
    def pipeline(self) -> EmgPipeline:
        return self._pipeline

    @property
    def source(self) -> EmgSource:
        return self._source

    # -- recording ------------------------------------------------------------

    def start_recording(self, path: str, *, subject: str = "default", notes: str = "") -> None:
        """Begin capturing raw EMG to disk."""
        self._recorder = EmgRecorder(
            path,
            self._source.channels,
            sample_rate_hz=self._source.sample_rate_hz,
            subject=subject,
            notes=notes,
        )
        self._recorder.start()

    def stop_recording(self):
        if self._recorder is None:
            return None
        info = self._recorder.stop()
        self._recorder = None
        return info

    def label_recording(self, label: str) -> None:
        """Tag subsequent samples — used to build supervised training sets."""
        if self._recorder is not None:
            self._recorder.set_label(label)

    # -- cycle ----------------------------------------------------------------

    def tick(self) -> EmgFrame | None:
        """Drain the source and update intent."""
        self.cycles += 1
        samples = self._source.read()
        if not samples:
            self.empty_cycles += 1
            return None

        if self._recorder is not None:
            self._recorder.write(samples)

        dropped_total = self._source.dropped_samples()
        dropped = max(0, dropped_total - self._last_dropped)
        self._last_dropped = dropped_total

        frame = self._pipeline.process(samples, dropped=dropped)
        if frame is None:
            return None

        self._latest_frame = frame
        self._bus.publish(Topics.EMG_FRAME, frame, source=self.name)
        self._watchdogs.kick("emg")

        # Calibration takes priority: while the wizard is running the user is
        # following instructions, and interpreting their signals as control
        # intent would fight them.
        if self._wizard is not None and self._wizard.active:
            progress = self._wizard.update([c.envelope for c in frame.channels])
            self._bus.publish(Topics.EMG_CALIBRATION_STEP, progress, source=self.name)
            if progress.finished:
                self._finish_calibration()
            return frame

        estimate = self._intent.update(frame)
        self._latest_intent = estimate
        self._bus.publish(Topics.INTENT_UPDATED, estimate, source=self.name)
        if estimate.is_cancel:
            self._bus.publish(Topics.INTENT_CANCEL, estimate, source=self.name)

        if self._recalibrator is not None:
            for event in self._recalibrator.update(frame):
                self._bus.publish(Topics.EMG_RECALIBRATED, event, source=self.name)

        return frame

    def start_calibration(self) -> None:
        """Begin the guided calibration wizard."""
        if self._wizard is None:
            log.warning("calibration requested but no wizard is configured")
            return
        self._wizard.start()
        self._bus.publish(Topics.EMG_CALIBRATION_STARTED, {}, source=self.name)

    def _finish_calibration(self) -> None:
        assert self._wizard is not None
        result = self._wizard.result
        if result is None:
            log.warning("calibration did not produce a usable result")
            return
        self._pipeline.set_calibration(result)
        self._intent.reset()
        if self._recalibrator is not None:
            self._recalibrator.set_calibration(result)
        if self._calibration_path:
            try:
                result.save(self._calibration_path)
            except OSError as exc:
                log.error("could not save calibration", error=str(exc))
        self._bus.publish(Topics.EMG_CALIBRATION_COMPLETE, result, source=self.name)

    # -- reporting ------------------------------------------------------------

    def health(self) -> HealthReport:
        if not self.running:
            return HealthReport.offline(self.name)
        frame = self._latest_frame
        if frame is None:
            return HealthReport.degraded(self.name, "no EMG frames yet")
        if frame.quality <= SignalQuality.UNUSABLE:
            return HealthReport.failed(
                self.name, "signal unusable — check the electrodes", quality=frame.quality.label
            )
        if frame.quality < SignalQuality.FAIR:
            return HealthReport.degraded(
                self.name,
                frame.reasons[0] if frame.reasons else "poor signal quality",
                quality=frame.quality.label,
            )
        return HealthReport.ok(
            self.name,
            quality=frame.quality.label,
            dropped=frame.dropped_samples,
            cycles=self.cycles,
        )


class VisionService(ServiceBase):
    """Drives the vision pipeline at its own rate."""

    service_name = "vision"

    def __init__(
        self,
        pipeline: VisionPipeline,
        bus: EventBus,
        watchdogs: WatchdogGroup,
    ) -> None:
        super().__init__()
        self._pipeline = pipeline
        self._bus = bus
        self._watchdogs = watchdogs
        self.cycles = 0

    def on_start(self) -> None:
        self._pipeline.start()
        # With no camera, the vision watchdog would expire immediately and
        # generate a permanent (and useless) fault.
        self._watchdogs.enable("vision", self._pipeline.has_camera)

    def on_stop(self) -> None:
        self._pipeline.stop()

    @property
    def pipeline(self) -> VisionPipeline:
        return self._pipeline

    @property
    def latest(self) -> VisionResult:
        return self._pipeline.latest

    def tick(self) -> VisionResult | None:
        self.cycles += 1
        result = self._pipeline.tick()
        if result is not None:
            self._bus.publish(Topics.VISION_RESULT, result, source=self.name)
            self._watchdogs.kick("vision")
            if not result.ok:
                self._bus.publish(
                    Topics.VISION_ERROR, {"error": result.error}, source=self.name
                )
        return result

    def health(self) -> HealthReport:
        return self._pipeline.health()
