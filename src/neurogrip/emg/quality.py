"""Signal-quality assessment.

Confidence in an EMG-derived intent is only as good as the signal it came from. A
detached electrode produces a plausible-looking envelope; without an independent
quality estimate the fusion layer would happily act on it. This module computes
that estimate from four independent indicators, so a single failure mode cannot
masquerade as a healthy signal:

* **saturation** — samples pinned at the amplifier rails;
* **noise floor** — baseline RMS relative to the calibrated rest level;
* **mains contamination** — energy at 50/60 Hz relative to the EMG band;
* **dropouts** — samples lost or timestamps stalling.

The resulting :class:`SignalQuality` gates AI assistance: below ``FAIR`` the
system stays in direct manual control and says why on screen.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from ..core.ringbuffer import RingBuffer
from ..core.types import clamp

__all__ = ["ChannelQuality", "QualityEstimator", "SignalQuality"]


class SignalQuality(IntEnum):
    """Coarse quality bands, ordered worst-to-best for easy comparison."""

    UNUSABLE = 0
    POOR = 1
    FAIR = 2
    GOOD = 3
    EXCELLENT = 4

    @property
    def label(self) -> str:
        return self.name.title()

    @property
    def allows_ai(self) -> bool:
        """Whether the assistive pipeline may use this signal."""
        return self >= SignalQuality.FAIR

    @classmethod
    def from_score(cls, score: float) -> SignalQuality:
        if score >= 0.85:
            return cls.EXCELLENT
        if score >= 0.68:
            return cls.GOOD
        if score >= 0.45:
            return cls.FAIR
        if score >= 0.2:
            return cls.POOR
        return cls.UNUSABLE


@dataclass(frozen=True, slots=True)
class ChannelQuality:
    """Quality assessment for one channel."""

    channel: int
    quality: SignalQuality
    score: float
    saturation_ratio: float = 0.0
    noise_ratio: float = 1.0
    mains_ratio: float = 0.0
    dropout_ratio: float = 0.0
    contact_ok: bool = True
    reasons: tuple[str, ...] = ()

    @property
    def is_usable(self) -> bool:
        return self.quality >= SignalQuality.FAIR


class QualityEstimator:
    """Rolling quality estimate for one channel.

    Mains contamination is measured with a two-bin Goertzel evaluation rather
    than a full FFT: we only need the power at two known frequencies, and
    Goertzel gets it in O(N) with no buffers and no NumPy.
    """

    #: Fraction of full scale above which a sample counts as clipped.
    SATURATION_LEVEL = 0.95
    #: Window used for all ratios, in samples (1 s at 1 kHz).
    WINDOW = 1000

    def __init__(
        self,
        channel: int,
        *,
        sample_rate_hz: float = 1000.0,
        full_scale_v: float = 1.65e-3,
        mains_hz: float = 50.0,
        window: int | None = None,
    ) -> None:
        self._channel = channel
        self._rate = sample_rate_hz
        self._full_scale = full_scale_v
        self._mains_hz = mains_hz
        self._window = window or self.WINDOW
        self._samples = RingBuffer(self._window)
        #: Pre-notch samples, used only for the interference measurement.
        self._prenotch = RingBuffer(self._window)
        self._saturated = 0
        self._count = 0
        self._dropped = 0
        self._expected = 0

    def add_sample(self, value: float, prenotch: float | None = None) -> None:
        """Feed one filtered sample, and optionally its pre-notch counterpart."""
        self._samples.append(value)
        self._prenotch.append(value if prenotch is None else prenotch)
        self._count += 1
        if abs(value) >= self._full_scale * self.SATURATION_LEVEL:
            self._saturated += 1
        if self._count > self._window:
            # Age out the saturation counter alongside the sample window.
            self._count = self._window
            self._saturated = int(self._saturated * (1.0 - 1.0 / self._window))

    def note_dropouts(self, dropped: int, expected: int) -> None:
        """Report acquisition losses observed by the HAL."""
        self._dropped += max(0, dropped)
        self._expected += max(1, expected)

    def evaluate(self, *, rest_noise_floor: float, active: bool) -> ChannelQuality:
        """Compute the current quality assessment.

        ``rest_noise_floor`` comes from calibration. ``active`` tells the
        estimator whether the user is currently contracting — a high signal level
        is expected then and must not be mistaken for noise.
        """
        reasons: list[str] = []
        values = self._samples.to_list()
        if len(values) < 16:
            return ChannelQuality(
                channel=self._channel,
                quality=SignalQuality.FAIR,
                score=0.5,
                reasons=("warming up",),
            )

        rms = self._samples.rms()

        # 1. Saturation.
        saturation = self._saturated / max(1, min(self._count, self._window))
        if saturation > 0.01:
            reasons.append(f"clipping {saturation * 100:.0f}% of samples")

        # 2. Noise floor. When the user is at rest, the RMS *is* the noise; a
        #    level far above the calibrated baseline means interference or a
        #    loose electrode.
        floor = max(rest_noise_floor, 1e-9)
        noise_ratio = rms / floor
        if not active and noise_ratio > 4.0:
            reasons.append(f"baseline {noise_ratio:.1f}× above calibrated noise floor")

        # 3. Mains contamination, measured before the notch filter removes it.
        reference = self._prenotch.to_list() or values
        mains_power = _goertzel_power(reference, self._mains_hz, self._rate)
        mains_power += _goertzel_power(reference, self._mains_hz * 3, self._rate)
        total_power = sum(v * v for v in reference) / len(reference)
        mains_ratio = clamp(mains_power / total_power) if total_power > 1e-18 else 0.0
        if mains_ratio > 0.25:
            reasons.append(f"{mains_ratio * 100:.0f}% mains interference")

        # 4. Dropouts.
        dropout_ratio = self._dropped / self._expected if self._expected else 0.0
        if dropout_ratio > 0.01:
            reasons.append(f"{dropout_ratio * 100:.1f}% samples dropped")

        # 5. Contact. Two signatures, because electrodes fail in two ways:
        #    a dead lead goes silent, while a lead that has lifted off the skin
        #    becomes a floating high-impedance input dominated by mains pickup.
        silent = rms <= floor * 0.25
        # A well-bonded differential electrode rejects mains down to around the
        # noise floor; a floating one is dominated by it. The threshold sits
        # well above any plausible bonded value.
        floating = mains_ratio > 0.75
        contact_ok = not (silent or floating)
        if silent:
            reasons.append("electrode contact lost (no signal)")
        elif floating:
            reasons.append("electrode contact lost (input floating, mains dominant)")

        score = 1.0
        score -= clamp(saturation * 5.0) * 0.35
        score -= clamp(max(0.0, noise_ratio - 3.0) / 10.0) * (0.0 if active else 0.3)
        score -= clamp(mains_ratio * 1.5) * 0.25
        score -= clamp(dropout_ratio * 10.0) * 0.25
        if not contact_ok:
            score -= 0.6
        score = clamp(score)

        return ChannelQuality(
            channel=self._channel,
            quality=SignalQuality.from_score(score),
            score=score,
            saturation_ratio=saturation,
            noise_ratio=noise_ratio,
            mains_ratio=mains_ratio,
            dropout_ratio=dropout_ratio,
            contact_ok=contact_ok,
            reasons=tuple(reasons),
        )

    def reset(self) -> None:
        self._samples.clear()
        self._prenotch.clear()
        self._saturated = 0
        self._count = 0
        self._dropped = 0
        self._expected = 0


def _goertzel_power(samples: Sequence[float], frequency_hz: float, sample_rate_hz: float) -> float:
    """Power at a single frequency via the Goertzel algorithm.

    O(N) with two state variables — the right tool when you need two bins rather
    than a whole spectrum.
    """
    n = len(samples)
    if n == 0 or frequency_hz >= sample_rate_hz / 2:
        return 0.0
    k = int(0.5 + n * frequency_hz / sample_rate_hz)
    omega = math.tau * k / n
    coefficient = 2.0 * math.cos(omega)
    s_prev = s_prev2 = 0.0
    for sample in samples:
        s = sample + coefficient * s_prev - s_prev2
        s_prev2 = s_prev
        s_prev = s
    power = s_prev2 * s_prev2 + s_prev * s_prev - coefficient * s_prev * s_prev2
    return power / (n * n) * 2.0
