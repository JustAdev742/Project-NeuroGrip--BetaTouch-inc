"""Monocular depth estimation from object size priors.

The hand carries a single camera, so metric depth has to be inferred. For known
object classes this is tractable with the pinhole relation::

    distance = (real_height × focal_length_px) / apparent_height_px

The catch is that the prior is a *class* prior: bottles range from 15 cm to 30 cm
tall. The estimator therefore reports a distance **and an honest error bar**, and
its confidence collapses for unknown classes. Downstream, depth only ever
modulates approach speed and grip aperture — never whether a grasp happens — so
a wrong estimate degrades comfort, not safety.

When a depth sensor is fitted the backend supplies a
:class:`~neurogrip.vision.types.DepthEstimate` with ``method="sensor"`` and this
module steps aside.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import clamp
from .types import BoundingBox, DepthEstimate, Detection

__all__ = ["OBJECT_SIZE_PRIORS", "MonocularDepthEstimator", "SizePrior"]


@dataclass(frozen=True, slots=True)
class SizePrior:
    """Typical physical size of an object class, in metres."""

    height_m: float
    width_m: float
    #: Relative spread of the class, e.g. 0.25 = ±25 %. Drives the error bar.
    variability: float = 0.25

    @property
    def max_dimension(self) -> float:
        return max(self.height_m, self.width_m)


#: Priors for the classes the vision backends can report. Values are median
#: dimensions of common household examples; the variability column matters as
#: much as the size, because it is what stops a wide class from being trusted.
OBJECT_SIZE_PRIORS: dict[str, SizePrior] = {
    "bottle": SizePrior(0.24, 0.07, 0.30),
    "cup": SizePrior(0.10, 0.08, 0.25),
    "can": SizePrior(0.12, 0.065, 0.12),
    "box": SizePrior(0.15, 0.20, 0.55),
    "ball": SizePrior(0.07, 0.07, 0.40),
    "pen": SizePrior(0.14, 0.01, 0.20),
    "key": SizePrior(0.05, 0.02, 0.30),
    "card": SizePrior(0.054, 0.086, 0.05),
    "phone": SizePrior(0.15, 0.07, 0.15),
    "book": SizePrior(0.22, 0.15, 0.35),
    "tool": SizePrior(0.20, 0.05, 0.50),
    "fruit": SizePrior(0.08, 0.08, 0.35),
    "plate": SizePrior(0.02, 0.24, 0.25),
    "handle": SizePrior(0.12, 0.03, 0.40),
}

#: Used when the class is unknown: a mid-sized object with a very wide spread,
#: which yields a low-confidence estimate rather than a confident wrong one.
_UNKNOWN_PRIOR = SizePrior(0.14, 0.10, 0.9)


class MonocularDepthEstimator:
    """Distance from apparent size, with calibrated uncertainty."""

    def __init__(
        self,
        *,
        horizontal_fov_deg: float = 62.0,
        image_width: int = 640,
        image_height: int = 480,
        min_distance_m: float = 0.06,
        max_distance_m: float = 1.20,
    ) -> None:
        self._min = min_distance_m
        self._max = max_distance_m
        self.configure(horizontal_fov_deg, image_width, image_height)

    def configure(self, horizontal_fov_deg: float, image_width: int, image_height: int) -> None:
        """Recompute focal length from the camera's field of view.

        Called whenever the camera changes resolution, so the estimator stays
        correct without anyone having to remember to update a constant.
        """
        import math

        self._image_width = image_width
        self._image_height = image_height
        half_fov = math.radians(max(1.0, horizontal_fov_deg)) / 2.0
        # Pinhole: f = (W/2) / tan(FOV/2)
        self._focal_px = (image_width / 2.0) / math.tan(half_fov)

    @property
    def focal_length_px(self) -> float:
        return self._focal_px

    def estimate(self, detection: Detection) -> DepthEstimate:
        """Estimate distance to a detected object."""
        prior = OBJECT_SIZE_PRIORS.get(detection.label, _UNKNOWN_PRIOR)
        return self.estimate_from_box(
            detection.bbox, prior, known_class=detection.label in OBJECT_SIZE_PRIORS
        )

    def estimate_from_box(
        self, bbox: BoundingBox, prior: SizePrior, *, known_class: bool = True
    ) -> DepthEstimate:
        """Estimate distance from a box and a size prior."""
        apparent_h_px = max(1.0, bbox.height * self._image_height)
        apparent_w_px = max(1.0, bbox.width * self._image_width)

        # Use whichever axis is better conditioned: the larger apparent extent
        # has proportionally less quantisation error.
        if apparent_h_px >= apparent_w_px:
            distance = prior.height_m * self._focal_px / apparent_h_px
        else:
            distance = prior.width_m * self._focal_px / apparent_w_px

        clamped = min(max(distance, self._min), self._max)
        out_of_range = abs(clamped - distance) > 1e-6

        # Confidence falls with class variability, with clipping to the working
        # range, and with very small boxes (few pixels, large relative error).
        confidence = 1.0 - clamp(prior.variability)
        if not known_class:
            confidence *= 0.35
        if out_of_range:
            confidence *= 0.4
        pixel_extent = max(apparent_h_px, apparent_w_px)
        confidence *= clamp(pixel_extent / 40.0, 0.2, 1.0)

        return DepthEstimate(
            distance_m=clamped,
            confidence=clamp(confidence),
            method="size_prior" if known_class else "default",
            relative_error=clamp(prior.variability + (0.3 if not known_class else 0.0), 0.05, 1.0),
        )

    def aperture_for(self, bbox: BoundingBox, distance_m: float) -> float:
        """Metric width of a box at a given distance, in metres.

        Feeds :mod:`neurogrip.ai.grasp`, which converts an object's real width
        into the finger closure that just contains it.
        """
        apparent_w_px = bbox.width * self._image_width
        return apparent_w_px * distance_m / max(1e-6, self._focal_px)
