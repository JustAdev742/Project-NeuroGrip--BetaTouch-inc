"""V4L2 / OpenCV camera driver.

Capture runs on its own thread with a depth-1 buffer. For a prosthetic hand,
*latency* matters far more than *throughput*: acting on a 200 ms-old frame is
worse than skipping frames, so the grabber always discards stale images and keeps
only the newest one.

``cv2`` is imported inside :meth:`open` so the module remains importable — and
``neurogrip diagnose`` remains runnable — on systems without OpenCV.
"""

from __future__ import annotations

import threading
from typing import Any

from ...core.clock import Clock, RealClock
from ...core.errors import DeviceNotAvailableError
from ...core.logging import get_logger
from ..base import DeviceInfo, DeviceKind
from .base import CameraSettings, Frame, PixelFormat

__all__ = ["OpenCvCamera"]

log = get_logger(__name__)


class OpenCvCamera:
    """Threaded camera grabber backed by ``cv2.VideoCapture``."""

    def __init__(
        self,
        device: int | str = 0,
        clock: Clock | None = None,
        *,
        settings: CameraSettings | None = None,
        backend: str = "auto",
    ) -> None:
        self._device = device
        self._clock = clock or RealClock()
        self._settings = settings or CameraSettings()
        self._backend = backend
        self._capture: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._consumed_index = 0
        self._index = 0
        self._dropped = 0
        self._error = ""

    def open(self) -> None:
        if self._capture is not None:
            return
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DeviceNotAvailableError(
                "opencv-python is not installed; install the 'vision' extra",
                context={"device": self._device},
            ) from exc

        api = getattr(cv2, "CAP_V4L2", 0) if self._backend == "v4l2" else getattr(cv2, "CAP_ANY", 0)
        capture = cv2.VideoCapture(self._device, api)
        if not capture.isOpened():
            capture.release()
            raise DeviceNotAvailableError(
                "camera could not be opened", context={"device": self._device}
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.height)
        capture.set(cv2.CAP_PROP_FPS, self._settings.fps)
        # Depth-1 driver buffer: with a deeper queue the newest frame would sit
        # behind stale ones, adding latency we cannot remove downstream.
        capture.set(getattr(cv2, "CAP_PROP_BUFFERSIZE", 38), 1)
        if self._settings.exposure_us is not None:
            capture.set(getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", 21), 1)
            capture.set(getattr(cv2, "CAP_PROP_EXPOSURE", 15), self._settings.exposure_us / 1e6)

        self._capture = capture
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, name="camera-grab", daemon=True)
        self._thread.start()
        log.info("camera opened", device=str(self._device), width=self._settings.width)

    def close(self) -> None:
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        capture, self._capture = self._capture, None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._running

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="camera",
            kind=DeviceKind.CAMERA,
            driver="opencv",
            connection=str(self._device),
            extra={
                "width": self._settings.width,
                "height": self._settings.height,
                "fps": self._settings.fps,
                "dropped": self._dropped,
                "error": self._error,
            },
        )

    @property
    def settings(self) -> CameraSettings:
        return self._settings

    def dropped_frames(self) -> int:
        return self._dropped

    def read(self) -> Frame | None:
        """Return the newest unconsumed frame, or ``None``."""
        with self._lock:
            frame = self._latest
            if frame is None or frame.index == self._consumed_index:
                return None
            self._consumed_index = frame.index
            return frame

    def _grab_loop(self) -> None:
        """Continuously grab frames, keeping only the most recent one."""
        while self._running and self._capture is not None:
            try:
                ok, image = self._capture.read()
            except Exception as exc:
                self._error = str(exc)
                continue
            if not ok or image is None:
                self._error = "capture returned no frame"
                continue

            self._index += 1
            height, width = image.shape[:2]
            frame = Frame(
                width=width,
                height=height,
                pixel_format=PixelFormat.BGR888,
                data=image.tobytes(),
                timestamp=self._clock.monotonic(),
                index=self._index,
                metadata={"driver": "opencv"},
            )
            with self._lock:
                if self._latest is not None and self._latest.index != self._consumed_index:
                    # Previous frame was never read: it is being dropped now.
                    self._dropped += 1
                self._latest = frame
                self._error = ""
