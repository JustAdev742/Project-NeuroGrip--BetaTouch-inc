"""Time-domain EMG features (the Hudgins set).

These five features — mean absolute value, waveform length, zero crossings, slope
sign changes, plus RMS — remain the standard input to myoelectric pattern
recognition because they are cheap, robust and separable. They are what a trained
classifier consumes; the threshold classifier uses only the amplitude features.

All extractors work on a window of *band-pass filtered* samples (post
:class:`~neurogrip.emg.filters.FilterChain`), never on raw ADC values.

Reference: Hudgins, Parker & Scott, "A new strategy for multifunction myoelectric
control", IEEE Trans. Biomed. Eng. 40(1), 1993.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "FeatureVector",
    "extract_features",
    "mav",
    "rms",
    "slope_sign_changes",
    "waveform_length",
    "zero_crossings",
]

#: Deadzone applied to zero-crossing and slope-sign counts, in volts. Without it,
#: baseline noise produces hundreds of spurious crossings per window and the
#: feature becomes a noise meter rather than a frequency proxy.
DEFAULT_THRESHOLD_V = 1.5e-5


def mav(samples: Sequence[float]) -> float:
    """Mean absolute value — the primary amplitude feature."""
    return sum(abs(s) for s in samples) / len(samples) if samples else 0.0


def rms(samples: Sequence[float]) -> float:
    """Root mean square — amplitude, weighted towards larger excursions."""
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


def waveform_length(samples: Sequence[float]) -> float:
    """Cumulative length of the waveform: sensitive to both amplitude and frequency."""
    if len(samples) < 2:
        return 0.0
    return sum(abs(samples[i] - samples[i - 1]) for i in range(1, len(samples)))


def zero_crossings(samples: Sequence[float], threshold: float = DEFAULT_THRESHOLD_V) -> int:
    """Count sign changes exceeding ``threshold`` — a crude frequency estimate."""
    count = 0
    for i in range(1, len(samples)):
        a, b = samples[i - 1], samples[i]
        if a * b < 0 and abs(a - b) >= threshold:
            count += 1
    return count


def slope_sign_changes(samples: Sequence[float], threshold: float = DEFAULT_THRESHOLD_V) -> int:
    """Count changes in the sign of the first difference — a spectral shape proxy."""
    count = 0
    for i in range(1, len(samples) - 1):
        prev_delta = samples[i] - samples[i - 1]
        next_delta = samples[i + 1] - samples[i]
        if prev_delta * next_delta < 0 and (
            abs(prev_delta) >= threshold or abs(next_delta) >= threshold
        ):
            count += 1
    return count


def variance(samples: Sequence[float]) -> float:
    """Sample variance about zero (EMG is zero-mean after filtering)."""
    if len(samples) < 2:
        return 0.0
    return sum(s * s for s in samples) / (len(samples) - 1)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Feature set for one channel over one window."""

    channel: int
    mav: float = 0.0
    rms: float = 0.0
    waveform_length: float = 0.0
    zero_crossings: int = 0
    slope_sign_changes: int = 0
    variance: float = 0.0
    window_samples: int = 0

    def as_tuple(self) -> tuple[float, ...]:
        """Ordered vector for a linear/LDA classifier. Order is part of the model contract."""
        return (
            self.mav,
            self.rms,
            self.waveform_length,
            float(self.zero_crossings),
            float(self.slope_sign_changes),
            self.variance,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "mav": self.mav,
            "rms": self.rms,
            "wl": self.waveform_length,
            "zc": float(self.zero_crossings),
            "ssc": float(self.slope_sign_changes),
            "var": self.variance,
        }

    @property
    def frequency_proxy(self) -> float:
        """Zero crossings normalised by window length — fatigue shifts this down.

        Muscle fatigue lowers conduction velocity and hence the EMG median
        frequency. Tracking this cheaply lets the auto-recalibration layer notice
        that a user's "maximum" is drifting because they are tired, rather than
        because their intent changed.
        """
        return self.zero_crossings / self.window_samples if self.window_samples else 0.0


def extract_features(
    channel: int, samples: Sequence[float], *, threshold: float = DEFAULT_THRESHOLD_V
) -> FeatureVector:
    """Compute the full feature set for one channel window in a single pass where possible."""
    if not samples:
        return FeatureVector(channel=channel)
    return FeatureVector(
        channel=channel,
        mav=mav(samples),
        rms=rms(samples),
        waveform_length=waveform_length(samples),
        zero_crossings=zero_crossings(samples, threshold),
        slope_sign_changes=slope_sign_changes(samples, threshold),
        variance=variance(samples),
        window_samples=len(samples),
    )
