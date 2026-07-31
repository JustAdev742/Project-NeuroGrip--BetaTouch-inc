"""Synthetic camera that renders a simple scene.

The generated image is deliberately crude — a coloured blob on a gradient
background — because its job is not to fool a neural network. Its job is to give
the whole pipeline *real bytes* to move around at a realistic rate and size, so
that frame timing, buffer handling, letterboxing and preprocessing are all
genuinely exercised.

Ground truth is attached to :attr:`Frame.metadata` under ``"scene"``. That is
consumed **only** by :class:`~neurogrip.vision.backends.mock.MockVisionBackend`,
which uses it to emit plausible detections with configurable noise and false
negatives. Any other consumer reading ground truth would be simulation cheating,
and ``tests/unit/test_vision_backends.py`` asserts that the real backends do not.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ...core.clock import Clock, RealClock
from ...core.types import clamp
from ..base import DeviceCapability, DeviceInfo, DeviceKind
from .base import CameraSettings, Frame, PixelFormat

__all__ = ["SceneObject", "SimulatedCamera"]


@dataclass(slots=True)
class SceneObject:
    """Ground-truth description of the object in view."""

    #: Semantic label; must match a key in the affordance database.
    label: str = "bottle"
    #: Normalised centre in image coordinates.
    center_x: float = 0.5
    center_y: float = 0.5
    #: Normalised extents.
    width: float = 0.25
    height: float = 0.45
    #: Metres from the camera.
    distance_m: float = 0.35
    #: Radians; used by grasp planners that reason about approach angle.
    orientation: float = 0.0
    colour: tuple[int, int, int] = (60, 140, 200)
    #: Roughly cylindrical objects render as vertical bars, spherical as circles.
    shape: str = "cylinder"
    #: Set false to render an empty scene (no object in view).
    visible: bool = True
    #: Extra truth passed through to the mock backend, e.g. graspable width.
    attributes: dict[str, float] = field(default_factory=dict)

    def as_metadata(self) -> dict[str, object]:
        return {
            "label": self.label,
            "bbox": (
                self.center_x - self.width / 2,
                self.center_y - self.height / 2,
                self.center_x + self.width / 2,
                self.center_y + self.height / 2,
            ),
            "distance_m": self.distance_m,
            "orientation": self.orientation,
            "shape": self.shape,
            "visible": self.visible,
            "attributes": dict(self.attributes),
        }


class SimulatedCamera:
    """Renders :class:`SceneObject` instances into RGB frames at a fixed rate."""

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        settings: CameraSettings | None = None,
        scene: SceneObject | None = None,
        noise: float = 6.0,
        seed: int = 4242,
        motion_blur: bool = False,
    ) -> None:
        self._clock = clock or RealClock()
        # Small by default: 160×120 keeps synthetic rendering cheap while
        # remaining representative for preprocessing and letterbox maths.
        self._settings = settings or CameraSettings(width=160, height=120, fps=30.0)
        self._scene = scene or SceneObject()
        self._noise = noise
        self._random = random.Random(seed)
        self._motion_blur = motion_blur
        self._open = False
        self._index = 0
        self._next_frame_at = 0.0
        self._dropped = 0
        self._shake_phase = 0.0

    # -- lifecycle ------------------------------------------------------------

    def open(self) -> None:
        self._open = True
        self._next_frame_at = self._clock.monotonic()

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="camera",
            kind=DeviceKind.CAMERA,
            driver="simulated",
            connection="sim",
            capabilities=frozenset({DeviceCapability.SIMULATED}),
            extra={
                "width": self._settings.width,
                "height": self._settings.height,
                "fps": self._settings.fps,
            },
        )

    @property
    def settings(self) -> CameraSettings:
        return self._settings

    def dropped_frames(self) -> int:
        return self._dropped

    # -- scene control --------------------------------------------------------

    @property
    def scene(self) -> SceneObject:
        return self._scene

    def set_scene(self, scene: SceneObject) -> None:
        """Install a new ground-truth scene (called by the simulated world)."""
        self._scene = scene

    # -- capture --------------------------------------------------------------

    def read(self) -> Frame | None:
        if not self._open:
            return None
        now = self._clock.monotonic()
        if now < self._next_frame_at:
            return None

        period = 1.0 / max(1e-3, self._settings.fps)
        behind = now - self._next_frame_at
        if behind > period * 3:
            # Consumer is slower than the sensor: count the skipped frames.
            skipped = int(behind / period)
            self._dropped += skipped
            self._next_frame_at = now + period
        else:
            self._next_frame_at += period

        self._index += 1
        self._shake_phase += 0.17
        return Frame(
            width=self._settings.width,
            height=self._settings.height,
            pixel_format=PixelFormat.RGB888,
            data=self._render(),
            timestamp=now,
            index=self._index,
            metadata={
                "scene": self._scene.as_metadata(),
                "simulated": True,
                "exposure_us": self._settings.exposure_us or 8000,
            },
        )

    def _render(self) -> bytes:
        """Rasterise the scene into an RGB888 buffer."""
        width, height = self._settings.width, self._settings.height
        buffer = bytearray(width * height * 3)
        scene = self._scene

        # Hand-held cameras shake; a couple of pixels of jitter keeps trackers honest.
        shake_x = math.sin(self._shake_phase) * 0.004
        shake_y = math.cos(self._shake_phase * 0.7) * 0.003

        cx = clamp(scene.center_x + shake_x)
        cy = clamp(scene.center_y + shake_y)
        half_w = scene.width / 2
        half_h = scene.height / 2

        for y in range(height):
            v = y / height
            # Background: vertical gradient, darker towards the bottom (a table).
            bg_r = int(120 - 60 * v)
            bg_g = int(125 - 55 * v)
            bg_b = int(130 - 50 * v)
            row = y * width * 3
            for x in range(width):
                u = x / width
                r, g, b = bg_r, bg_g, bg_b

                if scene.visible and self._inside(u, v, cx, cy, half_w, half_h, scene.shape):
                    # Simple diffuse shading so the blob is not flat.
                    shade = 0.75 + 0.25 * (1.0 - abs(u - cx) / max(1e-6, half_w))
                    r = int(scene.colour[0] * shade)
                    g = int(scene.colour[1] * shade)
                    b = int(scene.colour[2] * shade)

                if self._noise > 0:
                    n = self._random.gauss(0.0, self._noise)
                    r, g, b = int(r + n), int(g + n), int(b + n)

                offset = row + x * 3
                buffer[offset] = max(0, min(255, r))
                buffer[offset + 1] = max(0, min(255, g))
                buffer[offset + 2] = max(0, min(255, b))
        return bytes(buffer)

    @staticmethod
    def _inside(
        u: float, v: float, cx: float, cy: float, half_w: float, half_h: float, shape: str
    ) -> bool:
        dx = (u - cx) / max(1e-6, half_w)
        dy = (v - cy) / max(1e-6, half_h)
        if shape == "sphere":
            return dx * dx + dy * dy <= 1.0
        if shape == "flat":
            return abs(dx) <= 1.0 and abs(dy) <= 1.0 and abs(dy) > 0.55
        return abs(dx) <= 1.0 and abs(dy) <= 1.0
