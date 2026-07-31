"""Vision pipeline: camera → backend → tracking → depth → cached result.

Runs in its own rate group (typically 15–30 Hz), decoupled from the 200 Hz
control loop. Control never waits for vision; it reads the latest cached result
and checks its age. That decoupling is the reason a slow model degrades the
*quality* of assistance rather than the *responsiveness* of the hand.

Responsibilities:

* pull frames without blocking, skipping when the backend is behind;
* run the backend and never let it raise into the caller;
* stabilise detections through the tracker;
* attach a depth estimate when the backend does not provide one;
* keep statistics (FPS, latency, error rate) for diagnostics;
* let the backend be swapped at runtime, from the Settings screen.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..core.clock import Clock
from ..core.lifecycle import HealthReport, HealthStatus
from ..core.logging import get_logger
from ..core.ringbuffer import RingBuffer
from ..hal.camera.base import CameraSource
from .backend import VisionBackend
from .backends.null import NullVisionBackend
from .depth import MonocularDepthEstimator
from .tracking import ObjectTracker
from .types import VisionCapability, VisionResult

__all__ = ["VisionPipeline", "VisionStats"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VisionStats:
    """Rolling performance figures for the diagnostics screen."""

    frames_processed: int = 0
    frames_skipped: int = 0
    errors: int = 0
    fps: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    active_tracks: int = 0
    backend: str = ""
    last_error: str = ""

    @property
    def error_rate(self) -> float:
        total = self.frames_processed + self.errors
        return self.errors / total if total else 0.0


class VisionPipeline:
    """Owns the camera, the backend and the temporal post-processing."""

    def __init__(
        self,
        camera: CameraSource | None,
        backend: VisionBackend,
        clock: Clock,
        *,
        tracker: ObjectTracker | None = None,
        depth_estimator: MonocularDepthEstimator | None = None,
        max_result_age: float = 0.5,
    ) -> None:
        self._camera = camera
        self._backend = backend
        self._clock = clock
        self._tracker = tracker or ObjectTracker()
        self._depth = depth_estimator or MonocularDepthEstimator()
        #: Results older than this are reported as stale to the fusion layer.
        self._max_age = max_result_age

        #: Optional :class:`~neurogrip.vision.backends.replay.VisionRecorder`.
        #: Set to capture a replayable recording of what vision reported.
        self.recorder = None
        self._latest = VisionResult.empty(0.0, backend="none")
        self._latencies = RingBuffer(120)
        self._frame_times = RingBuffer(120)
        self._processed = 0
        self._skipped = 0
        self._errors = 0
        self._last_error = ""
        self._last_frame_at = 0.0
        self._started = False

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        if self._camera is not None:
            try:
                self._camera.open()
                settings = self._camera.settings
                self._depth.configure(62.0, settings.width, settings.height)
            except Exception as exc:
                log.warning("camera failed to open; continuing without vision", error=str(exc))
                self._camera = None
        try:
            self._backend.initialize()
        except Exception as exc:
            log.error("vision backend failed to initialise", error=str(exc))
            self._backend = NullVisionBackend(f"initialisation failed: {exc}")
            self._backend.initialize()
        self._started = True
        log.info("vision pipeline started", backend=str(self._backend.info()))

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        try:
            self._backend.shutdown()
        finally:
            if self._camera is not None:
                self._camera.close()

    # -- runtime --------------------------------------------------------------

    def set_backend(self, backend: VisionBackend) -> None:
        """Hot-swap the backend (Settings ▸ Vision).

        The tracker is reset because track identities are not comparable across
        models — carrying them over would let a stale label from the old backend
        outvote fresh evidence from the new one.
        """
        old = self._backend
        try:
            backend.initialize()
        except Exception as exc:
            log.error("new vision backend failed to initialise; keeping current", error=str(exc))
            return
        self._backend = backend
        self._tracker.reset()
        old.shutdown()
        log.info("vision backend switched", backend=str(backend.info()))

    def tick(self) -> VisionResult | None:
        """Process at most one frame. Returns the new result, or ``None``."""
        if not self._started or self._camera is None:
            return None

        frame = self._camera.read()
        if frame is None:
            return None

        now = self._clock.monotonic()
        if self._last_frame_at:
            self._frame_times.append(now - self._last_frame_at)
        self._last_frame_at = now

        try:
            result = self._backend.process(frame)
        except Exception as exc:
            self._errors += 1
            self._last_error = str(exc)
            log.throttled(
                "vision-backend", "error", "vision backend raised", now=now, error=str(exc)
            )
            self._latest = VisionResult.empty(frame.timestamp, self._backend.info().name, str(exc))
            return self._latest

        if not result.ok:
            self._errors += 1
            self._last_error = result.error
        else:
            self._processed += 1
            self._latencies.append(result.latency_ms)

        result = self._post_process(result)
        self._latest = result
        if self.recorder is not None:
            # Recorded after post-processing, so a replay reproduces what the
            # rest of the stack actually saw — including tracking and the depth
            # the pipeline filled in — rather than the raw backend output.
            try:
                self.recorder.write(result)
            except Exception as exc:
                log.throttled(
                    "vision-record", "warning", "could not write recording",
                    now=now, error=str(exc),
                )
        return result

    def _post_process(self, result: VisionResult) -> VisionResult:
        """Apply tracking and fill in depth when the backend did not supply it."""
        tracked = self._tracker.update(result.detections, result.timestamp)

        depth = result.depth
        if depth is None and tracked:
            primary = max(tracked, key=lambda d: d.confidence)
            depth = self._depth.estimate(primary)

        grasps = result.grasps
        if depth is not None and grasps:
            # Backends that work purely in image space get metric annotations
            # here, keeping the pinhole maths in exactly one place:
            #   width_m = width_normalised × image_width_px × distance / focal_px
            pixels_per_metre = max(1e-6, self._depth.focal_length_px) / depth.distance_m
            grasps = tuple(
                g
                if g.depth_m is not None
                else replace(
                    g,
                    depth_m=depth.distance_m,
                    width_m=g.width * self._camera_width() / pixels_per_metre,
                )
                for g in grasps
            )

        return VisionResult(
            timestamp=result.timestamp,
            frame_index=result.frame_index,
            detections=tracked,
            grasps=grasps,
            depth=depth,
            latency_ms=result.latency_ms,
            backend=result.backend,
            capabilities=(
                result.capabilities
                | (VisionCapability.TRACKING if tracked else VisionCapability.NONE)
            ),
            error=result.error,
        )

    def _camera_width(self) -> float:
        return float(self._camera.settings.width) if self._camera else 640.0

    # -- accessors ------------------------------------------------------------

    @property
    def latest(self) -> VisionResult:
        """Most recent result. Always safe to read; check :meth:`is_fresh`."""
        return self._latest

    def is_fresh(self, now: float | None = None) -> bool:
        moment = self._clock.monotonic() if now is None else now
        return self._latest.is_fresh(moment, self._max_age)

    @property
    def has_camera(self) -> bool:
        return self._camera is not None

    @property
    def capabilities(self) -> VisionCapability:
        return self._backend.capabilities

    def stats(self) -> VisionStats:
        mean_period = self._frame_times.mean()
        return VisionStats(
            frames_processed=self._processed,
            frames_skipped=self._skipped + (self._camera.dropped_frames() if self._camera else 0),
            errors=self._errors,
            fps=(1.0 / mean_period) if mean_period > 1e-6 else 0.0,
            mean_latency_ms=self._latencies.mean(),
            p95_latency_ms=self._latencies.percentile(0.95),
            active_tracks=self._tracker.active_tracks,
            backend=self._backend.info().name,
            last_error=self._last_error,
        )

    def health(self) -> HealthReport:
        """Health for the diagnostics aggregator."""
        info = self._backend.info()
        stats = self.stats()
        if self._camera is None:
            return HealthReport(
                name="vision",
                status=HealthStatus.OFFLINE,
                detail="no camera; AI assistance limited to EMG-only control",
                metrics={"backend": info.name},
            )
        if not self._started:
            return HealthReport.offline("vision")
        if stats.error_rate > 0.25:
            return HealthReport.failed(
                "vision",
                f"{stats.error_rate * 100:.0f}% of frames failed: {stats.last_error}",
                **stats.__dict__,
            )
        if info.is_degraded:
            return HealthReport.degraded("vision", info.degraded_reason, backend=info.name, fps=stats.fps)
        if stats.fps < 5.0 and stats.frames_processed > 30:
            return HealthReport.degraded("vision", f"low frame rate ({stats.fps:.1f} fps)", fps=stats.fps)
        return HealthReport.ok("vision", fps=round(stats.fps, 1), latency_ms=round(stats.mean_latency_ms, 1))
