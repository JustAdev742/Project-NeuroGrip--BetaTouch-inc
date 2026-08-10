"""Camera calibration for monocular distance estimation.

:class:`~neurogrip.vision.depth.MonocularDepthEstimator` recovers distance from
apparent size using a pinhole model, so every distance it reports is only as good
as one number: the horizontal field of view. That number is normally copied from
a datasheet, and datasheet FOV is quoted for the sensor's full area — a camera
configured at a cropped resolution has a materially narrower one. A 66° figure
used on a sensor delivering 58° biases every distance by about 12%, consistently,
in the direction that makes objects look further away than they are.

The procedure is the simplest one that produces a real measurement: show the
camera an object of known width at a known distance, and solve the pinhole
relation for focal length.

    f = (apparent_width_px × distance) / real_width

    FOV = 2 · atan((image_width / 2) / f)

Several samples at different distances are averaged, and their disagreement is
reported: a consistent set means the pinhole model fits, and a scattered one
means either the measurements were sloppy or the lens has enough distortion that
size-based depth will not be trustworthy. Saying so is more useful than emitting
a confident average of bad data.

This does *not* estimate lens distortion coefficients. Doing that properly needs
a checkerboard and an optimiser, which would mean a hard dependency on OpenCV for
a correction the size-prior uncertainty already dominates. If a future backend
needs true intrinsics, this is where they would go.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..core.errors import CalibrationError
from ..core.logging import get_logger
from ..core.ringbuffer import RunningStats

__all__ = [
    "REFERENCE_TARGETS",
    "CalibrationSample",
    "CameraCalibration",
    "CameraCalibrationWizard",
]

log = get_logger(__name__)

#: Everyday objects with a tightly controlled width, so a bring-up session does
#: not need a printed target. Widths in metres.
REFERENCE_TARGETS: dict[str, float] = {
    # ISO/IEC 7810 ID-1 — bank cards, most ID cards. 85.60 mm, ±0.12 mm.
    "card": 0.0856,
    # ISO 216 A4 short edge.
    "a4": 0.210,
    # A standard 330 ml drinks can.
    "can": 0.066,
    # A CD/DVD.
    "disc": 0.120,
}

#: Below this the measurement is dominated by pixel quantisation; above it the
#: object is too small in frame for its edges to be located reliably.
MIN_APPARENT_WIDTH_PX = 40.0

#: Spread across samples above which the pinhole fit is not trustworthy.
MAX_ACCEPTABLE_SPREAD = 0.08


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    """One measurement of a known target."""

    #: True width of the target, metres.
    target_width_m: float
    #: Distance from the lens to the target, metres.
    distance_m: float
    #: Apparent width in the image, pixels.
    apparent_width_px: float
    label: str = ""

    @property
    def focal_length_px(self) -> float:
        """Focal length implied by this sample alone."""
        if self.target_width_m <= 0:
            raise CalibrationError("target width must be positive")
        return self.apparent_width_px * self.distance_m / self.target_width_m

    @property
    def usable(self) -> bool:
        return (
            self.apparent_width_px >= MIN_APPARENT_WIDTH_PX
            and self.distance_m > 0
            and self.target_width_m > 0
        )


@dataclass(slots=True)
class CameraCalibration:
    """Measured camera intrinsics, as far as monocular depth needs them."""

    horizontal_fov_deg: float = 62.0
    focal_length_px: float = 0.0
    image_width: int = 640
    image_height: int = 480
    #: Relative standard deviation of the focal length across samples.
    spread: float = 0.0
    samples: int = 0
    measured_at: float = 0.0
    notes: str = ""
    version: int = 1

    @property
    def vertical_fov_deg(self) -> float:
        """Derived from the focal length; assumes square pixels."""
        if self.focal_length_px <= 0:
            return 0.0
        return math.degrees(2.0 * math.atan((self.image_height / 2.0) / self.focal_length_px))

    @property
    def is_trustworthy(self) -> bool:
        """False when the samples disagree enough to distrust the pinhole fit."""
        return self.samples >= 3 and self.spread <= MAX_ACCEPTABLE_SPREAD

    def describe(self) -> str:
        return (
            f"FOV {self.horizontal_fov_deg:.1f}° horizontal "
            f"({self.vertical_fov_deg:.1f}° vertical), "
            f"f = {self.focal_length_px:.0f} px at {self.image_width}×{self.image_height}, "
            f"spread {self.spread * 100:.1f}% over {self.samples} sample(s)"
        )

    def save(self, path: Path | str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        temporary.replace(target)
        log.info("camera calibration saved", path=str(target), fov=round(self.horizontal_fov_deg, 2))

    @classmethod
    def load(cls, path: Path | str) -> CameraCalibration:
        file_path = Path(path)
        if not file_path.exists():
            raise CalibrationError(f"camera calibration file not found: {file_path}")
        try:
            return cls(**json.loads(file_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid camera calibration file: {exc}") from exc


class CameraCalibrationWizard:
    """Collects samples and solves for the field of view.

    Kept free of any camera dependency: it takes measurements, not frames. That
    lets the same implementation serve an operator typing pixel widths measured
    by hand, a detector-driven flow that reads the width off a bounding box, and
    the tests — none of which need a camera attached.
    """

    def __init__(self, image_width: int = 640, image_height: int = 480) -> None:
        self._image_width = image_width
        self._image_height = image_height
        self._samples: list[CalibrationSample] = []

    def add_sample(self, sample: CalibrationSample) -> None:
        """Record one measurement, rejecting ones too poor to contribute."""
        if not sample.usable:
            raise CalibrationError(
                f"unusable sample: {sample.apparent_width_px:.0f} px wide at "
                f"{sample.distance_m:.2f} m (need at least "
                f"{MIN_APPARENT_WIDTH_PX:.0f} px — move the target closer)"
            )
        self._samples.append(sample)

    def add_measurement(
        self, target: str, distance_m: float, apparent_width_px: float
    ) -> None:
        """Add a sample using one of the :data:`REFERENCE_TARGETS`."""
        if target not in REFERENCE_TARGETS:
            raise CalibrationError(
                f"unknown target {target!r}; known targets: {', '.join(REFERENCE_TARGETS)}"
            )
        self.add_sample(
            CalibrationSample(
                target_width_m=REFERENCE_TARGETS[target],
                distance_m=distance_m,
                apparent_width_px=apparent_width_px,
                label=target,
            )
        )

    @property
    def samples(self) -> tuple[CalibrationSample, ...]:
        return tuple(self._samples)

    def clear(self) -> None:
        self._samples.clear()

    def solve(self) -> CameraCalibration:
        """Compute the calibration from the collected samples."""
        if not self._samples:
            raise CalibrationError("no samples collected")

        stats = RunningStats()
        for sample in self._samples:
            stats.add(sample.focal_length_px)

        focal = stats.mean
        if focal <= 0:
            raise CalibrationError("samples produced a non-physical focal length")

        spread = stats.std / focal if focal > 0 else 0.0
        fov = math.degrees(2.0 * math.atan((self._image_width / 2.0) / focal))

        calibration = CameraCalibration(
            horizontal_fov_deg=fov,
            focal_length_px=focal,
            image_width=self._image_width,
            image_height=self._image_height,
            spread=spread,
            samples=len(self._samples),
            measured_at=time.time(),
        )
        if not calibration.is_trustworthy:
            calibration.notes = (
                f"samples disagree by {spread * 100:.1f}% — re-measure the distances, "
                "or expect size-based depth to be unreliable"
                if len(self._samples) >= 3
                else "fewer than three samples; take more at different distances"
            )
            log.warning("camera calibration is not trustworthy", detail=calibration.notes)
        log.info("camera calibration solved", detail=calibration.describe())
        return calibration

    def residuals(self, calibration: CameraCalibration) -> tuple[tuple[str, float], ...]:
        """Per-sample distance error under ``calibration``, for the report.

        This is what tells an operator *which* measurement was wrong, rather than
        only that one of them was.
        """
        rows: list[tuple[str, float]] = []
        for sample in self._samples:
            if calibration.focal_length_px <= 0:
                continue
            implied = (
                sample.target_width_m * calibration.focal_length_px / sample.apparent_width_px
            )
            error = implied - sample.distance_m
            rows.append(
                (
                    f"{sample.label or 'target'} at {sample.distance_m:.2f} m",
                    error,
                )
            )
        return tuple(rows)
