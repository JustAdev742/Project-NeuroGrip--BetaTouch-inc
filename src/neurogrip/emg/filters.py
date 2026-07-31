"""Digital filters for surface EMG.

Pure-Python, sample-at-a-time, O(1) per sample and allocation-free in the hot
path. At 1 kHz × 2 channels this costs well under 1 % of a Cortex-A72 core, and
avoiding NumPy keeps the core runtime dependency-free and its latency predictable
(no hidden buffering, no vectorisation cliff).

The standard EMG conditioning chain, in order:

1. **DC block** — removes electrode half-cell potential (a large, slowly varying
   offset that would otherwise dominate everything downstream).
2. **Mains notch** at 50/60 Hz plus the third harmonic — the dominant artefact in
   any real recording environment.
3. **Band-pass 20–400 Hz** — below 20 Hz is motion artefact, above ~400 Hz is
   mostly noise; this is the conventional surface-EMG band (De Luca, 1997).
4. **Rectify + envelope** — full-wave rectification followed by an asymmetric
   low-pass, giving the smooth activation signal control uses.

Coefficients follow the Audio EQ Cookbook (Robert Bristow-Johnson) biquad forms,
which are numerically well behaved at the low centre frequencies used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..core.ringbuffer import SlidingWindow

__all__ = [
    "Biquad",
    "BiquadCascade",
    "DCBlocker",
    "EnvelopeFollower",
    "FilterChain",
    "MedianFilter",
    "MovingRms",
    "build_emg_chain",
]


@dataclass(slots=True)
class Biquad:
    """Second-order IIR section in transposed direct form II.

    Transposed DF-II is used because it has the best numerical behaviour of the
    direct forms for fixed coefficients and minimises state (two registers).
    """

    b0: float = 1.0
    b1: float = 0.0
    b2: float = 0.0
    a1: float = 0.0
    a2: float = 0.0
    z1: float = 0.0
    z2: float = 0.0

    def process(self, x: float) -> float:
        """Filter one sample."""
        y = self.b0 * x + self.z1
        self.z1 = self.b1 * x - self.a1 * y + self.z2
        self.z2 = self.b2 * x - self.a2 * y
        return y

    def reset(self) -> None:
        """Zero the delay line (called on calibration or electrode reconnect)."""
        self.z1 = 0.0
        self.z2 = 0.0

    # -- design helpers -------------------------------------------------------

    @staticmethod
    def _normalise(b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> Biquad:
        return Biquad(b0=b0 / a0, b1=b1 / a0, b2=b2 / a0, a1=a1 / a0, a2=a2 / a0)

    @classmethod
    def lowpass(cls, cutoff_hz: float, sample_rate_hz: float, q: float = 0.7071) -> Biquad:
        _w0, cos_w0, alpha = _omega(cutoff_hz, sample_rate_hz, q)
        b0 = (1 - cos_w0) / 2
        b1 = 1 - cos_w0
        b2 = b0
        return cls._normalise(b0, b1, b2, 1 + alpha, -2 * cos_w0, 1 - alpha)

    @classmethod
    def highpass(cls, cutoff_hz: float, sample_rate_hz: float, q: float = 0.7071) -> Biquad:
        _w0, cos_w0, alpha = _omega(cutoff_hz, sample_rate_hz, q)
        b0 = (1 + cos_w0) / 2
        b1 = -(1 + cos_w0)
        b2 = b0
        return cls._normalise(b0, b1, b2, 1 + alpha, -2 * cos_w0, 1 - alpha)

    @classmethod
    def bandpass(cls, centre_hz: float, sample_rate_hz: float, q: float = 1.0) -> Biquad:
        """Constant-peak-gain band-pass."""
        _w0, cos_w0, alpha = _omega(centre_hz, sample_rate_hz, q)
        return cls._normalise(alpha, 0.0, -alpha, 1 + alpha, -2 * cos_w0, 1 - alpha)

    @classmethod
    def notch(cls, centre_hz: float, sample_rate_hz: float, q: float = 30.0) -> Biquad:
        """Narrow band-stop, used for mains interference.

        A high ``Q`` (30 by default) keeps the notch tight enough that it removes
        the 50/60 Hz tone without gutting the EMG energy that sits right beside it.
        """
        _w0, cos_w0, alpha = _omega(centre_hz, sample_rate_hz, q)
        return cls._normalise(1.0, -2 * cos_w0, 1.0, 1 + alpha, -2 * cos_w0, 1 - alpha)

    def magnitude_at(self, frequency_hz: float, sample_rate_hz: float) -> float:
        """Magnitude response at ``frequency_hz`` — used by the filter self-test."""
        w = math.tau * frequency_hz / sample_rate_hz
        cos_w, sin_w = math.cos(w), math.sin(w)
        cos_2w, sin_2w = math.cos(2 * w), math.sin(2 * w)
        num_real = self.b0 + self.b1 * cos_w + self.b2 * cos_2w
        num_imag = -(self.b1 * sin_w + self.b2 * sin_2w)
        den_real = 1.0 + self.a1 * cos_w + self.a2 * cos_2w
        den_imag = -(self.a1 * sin_w + self.a2 * sin_2w)
        num = math.hypot(num_real, num_imag)
        den = math.hypot(den_real, den_imag)
        return num / den if den > 1e-18 else 0.0


def _omega(frequency_hz: float, sample_rate_hz: float, q: float) -> tuple[float, float, float]:
    """Shared biquad design intermediates, with frequency clamped below Nyquist."""
    nyquist = sample_rate_hz / 2.0
    frequency_hz = min(max(frequency_hz, 0.1), nyquist * 0.98)
    q = max(0.05, q)
    w0 = math.tau * frequency_hz / sample_rate_hz
    return w0, math.cos(w0), math.sin(w0) / (2.0 * q)


class BiquadCascade:
    """Series chain of :class:`Biquad` sections (higher-order responses)."""

    __slots__ = ("_sections",)

    def __init__(self, sections: list[Biquad]) -> None:
        self._sections = sections

    def process(self, x: float) -> float:
        for section in self._sections:
            x = section.process(x)
        return x

    def reset(self) -> None:
        for section in self._sections:
            section.reset()

    def magnitude_at(self, frequency_hz: float, sample_rate_hz: float) -> float:
        gain = 1.0
        for section in self._sections:
            gain *= section.magnitude_at(frequency_hz, sample_rate_hz)
        return gain

    @classmethod
    def bandpass(
        cls, low_hz: float, high_hz: float, sample_rate_hz: float, order: int = 2
    ) -> BiquadCascade:
        """Band-pass built from cascaded high-pass and low-pass sections.

        Cascading HP+LP rather than using a single band-pass biquad gives
        independent control of the two edges, which matters because the low edge
        (motion artefact rejection) wants a steeper roll-off than the high edge.
        """
        sections: list[Biquad] = []
        for _ in range(max(1, order)):
            sections.append(Biquad.highpass(low_hz, sample_rate_hz))
            sections.append(Biquad.lowpass(high_hz, sample_rate_hz))
        return cls(sections)


class DCBlocker:
    """One-pole high-pass removing electrode DC offset.

    ``y[n] = x[n] - x[n-1] + r * y[n-1]``; ``r`` is derived from the corner
    frequency. Cheaper and better conditioned at very low corners (~0.5 Hz) than
    a biquad, which is why it is a separate stage.
    """

    __slots__ = ("_r", "_x1", "_y1")

    def __init__(self, cutoff_hz: float = 0.5, sample_rate_hz: float = 1000.0) -> None:
        self._r = math.exp(-math.tau * cutoff_hz / sample_rate_hz)
        self._x1 = 0.0
        self._y1 = 0.0

    def process(self, x: float) -> float:
        y = x - self._x1 + self._r * self._y1
        self._x1 = x
        self._y1 = y
        return y

    def reset(self) -> None:
        self._x1 = 0.0
        self._y1 = 0.0


class EnvelopeFollower:
    """Rectify-and-smooth with independent attack and release time constants.

    Asymmetry is a control decision, not a cosmetic one:

    * a **short attack** (~30 ms) keeps the latency between a user's contraction
      and the hand responding within the ~100 ms that feels immediate;
    * a **longer release** (~150 ms) prevents the grip from fluttering open
      during the natural amplitude dips of a sustained contraction.
    """

    __slots__ = ("_attack", "_dt", "_release", "_value")

    def __init__(
        self, attack_s: float = 0.03, release_s: float = 0.15, sample_rate_hz: float = 1000.0
    ) -> None:
        self._dt = 1.0 / sample_rate_hz
        self._attack = self._coefficient(attack_s)
        self._release = self._coefficient(release_s)
        self._value = 0.0

    def _coefficient(self, tau: float) -> float:
        return 1.0 - math.exp(-self._dt / max(1e-6, tau)) if tau > 0 else 1.0

    def process(self, x: float) -> float:
        rectified = abs(x)
        alpha = self._attack if rectified > self._value else self._release
        self._value += (rectified - self._value) * alpha
        return self._value

    @property
    def value(self) -> float:
        return self._value

    def reset(self, value: float = 0.0) -> None:
        self._value = value


class MovingRms:
    """Windowed RMS over a fixed number of samples.

    Maintains a running sum of squares, so cost is O(1) per sample regardless of
    window length.
    """

    __slots__ = ("_buffer", "_count", "_index", "_sum", "_window")

    def __init__(self, window_samples: int) -> None:
        self._window = max(1, window_samples)
        self._buffer = [0.0] * self._window
        self._sum = 0.0
        self._index = 0
        self._count = 0

    def process(self, x: float) -> float:
        square = x * x
        self._sum -= self._buffer[self._index]
        self._buffer[self._index] = square
        self._sum += square
        self._index = (self._index + 1) % self._window
        self._count = min(self._count + 1, self._window)
        # Guard against negative sums from floating-point cancellation.
        return math.sqrt(max(0.0, self._sum) / self._count)

    def reset(self) -> None:
        self._buffer = [0.0] * self._window
        self._sum = 0.0
        self._index = 0
        self._count = 0


class MedianFilter:
    """Sliding median — removes isolated spikes without smearing edges.

    Applied to the *envelope*, not the raw signal: a single ADC glitch would
    otherwise produce a spurious activation spike and, in the worst case, an
    unintended grasp.
    """

    __slots__ = ("_size", "_values")

    def __init__(self, size: int = 5) -> None:
        self._size = max(1, size | 1)  # force odd
        self._values: list[float] = []

    def process(self, x: float) -> float:
        self._values.append(x)
        if len(self._values) > self._size:
            self._values.pop(0)
        ordered = sorted(self._values)
        return ordered[len(ordered) // 2]

    def reset(self) -> None:
        self._values.clear()


class FilterChain:
    """Complete per-channel conditioning chain.

    One instance per EMG channel. Holds all filter state, so resetting a channel
    (electrode reseated, calibration restarted) is a single call.
    """

    __slots__ = ("_sample_rate", "band", "dc", "envelope", "median", "notches", "prenotch", "rms")

    def __init__(
        self,
        sample_rate_hz: float,
        *,
        mains_hz: float = 50.0,
        band_low_hz: float = 20.0,
        band_high_hz: float = 400.0,
        attack_s: float = 0.03,
        release_s: float = 0.15,
        median_size: int = 5,
        rms_window_s: float = 0.2,
    ) -> None:
        self._sample_rate = sample_rate_hz
        self.dc = DCBlocker(0.5, sample_rate_hz)
        # Fundamental plus third harmonic; the second is largely absent from
        # mains hum but the third is consistently present.
        self.notches = [
            Biquad.notch(mains_hz, sample_rate_hz, q=30.0),
            Biquad.notch(mains_hz * 3.0, sample_rate_hz, q=30.0),
        ]
        self.band = BiquadCascade.bandpass(band_low_hz, band_high_hz, sample_rate_hz, order=1)
        self.envelope = EnvelopeFollower(attack_s, release_s, sample_rate_hz)
        self.median = MedianFilter(median_size)
        self.rms = MovingRms(int(rms_window_s * sample_rate_hz))
        #: Last sample after DC blocking but *before* the mains notch. The
        #: quality estimator needs this: measuring interference downstream of the
        #: filter that removes it would always report a clean signal.
        self.prenotch = 0.0

    def process(self, sample: float) -> tuple[float, float, float]:
        """Filter one raw sample.

        Returns ``(filtered, envelope, rms)`` where ``filtered`` is the
        band-limited signal (used for feature extraction and the raw waveform
        display) and ``envelope`` is the smoothed activation.
        """
        x = self.dc.process(sample)
        self.prenotch = x
        for notch in self.notches:
            x = notch.process(x)
        filtered = self.band.process(x)
        envelope = self.median.process(self.envelope.process(filtered))
        rms = self.rms.process(filtered)
        return filtered, envelope, rms

    def reset(self) -> None:
        self.dc.reset()
        for notch in self.notches:
            notch.reset()
        self.band.reset()
        self.envelope.reset()
        self.median.reset()
        self.rms.reset()

    @property
    def sample_rate_hz(self) -> float:
        return self._sample_rate


def build_emg_chain(sample_rate_hz: float, **kwargs: float) -> FilterChain:
    """Convenience factory used by the pipeline and by tests."""
    return FilterChain(sample_rate_hz, **kwargs)  # type: ignore[arg-type]


def sliding_window(duration_s: float) -> SlidingWindow:
    """Re-exported for feature extractors that window by time, not sample count."""
    return SlidingWindow(duration_s)
