"""Synthetic EMG source.

Generates signals with the properties the processing chain must actually handle,
so that filters, calibration, quality metrics and the intent engine are exercised
realistically without a human in the loop:

* **Baseline noise** — Gaussian, ~10 µV RMS, present even at rest.
* **Contraction bursts** — amplitude-modulated noise whose RMS scales with the
  drive level, which is the first-order model of surface EMG.
* **Mains interference** — 50/60 Hz plus its third harmonic, the single most
  important artefact the notch filter has to remove. With good electrode contact
  a differential front end with a driven-reference leg rejects most of it, so the
  residual sits near the noise floor; when an electrode lifts, the input floats
  and mains pickup rises by more than an order of magnitude. That contrast is
  what makes electrode-off detection possible without impedance hardware.
* **Motion artefact** — occasional low-frequency excursions from electrode
  movement, which is what the high-pass stage exists for.
* **DC offset and drift** — slow baseline wander per channel.
* **Crosstalk** — a fraction of the flexor drive appears on the extensor channel
  and vice versa, which is why co-contraction detection needs a margin.

Drive levels are set by the caller: programmatically (tests), by a scripted
scenario (:mod:`neurogrip.simulation`), or by the Training Mode UI when a user is
practising without hardware.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from ...core.clock import Clock, RealClock
from ...core.types import clamp
from ..base import DeviceCapability, DeviceInfo, DeviceKind
from .base import EmgChannelSpec, EmgSample, EmgSourceStats

__all__ = ["DEFAULT_CHANNELS", "SimulatedEmgSource"]

#: Two-site placement: one electrode over the flexor group, one over the extensor
#: group. This is the minimum viable configuration for open/close/co-contraction
#: and matches the hardware described in ``docs/hardware.md``.
DEFAULT_CHANNELS: tuple[EmgChannelSpec, ...] = (
    EmgChannelSpec(index=0, name="Flexor", role="flexor", site="flexor digitorum superficialis"),
    EmgChannelSpec(index=1, name="Extensor", role="extensor", site="extensor digitorum communis"),
)


class SimulatedEmgSource:
    """Deterministic, seedable synthetic EMG generator."""

    def __init__(
        self,
        clock: Clock | None = None,
        *,
        sample_rate_hz: float = 1000.0,
        channels: Sequence[EmgChannelSpec] = DEFAULT_CHANNELS,
        seed: int = 20260730,
        mains_hz: float = 50.0,
        mains_amplitude_v: float = 1.2e-5,
        noise_floor_v: float = 1.0e-5,
        max_amplitude_v: float = 1.2e-3,
        crosstalk: float = 0.12,
        buffer_limit: int = 8192,
    ) -> None:
        self._clock = clock or RealClock()
        self._rate = sample_rate_hz
        self._channels = tuple(channels)
        self._random = random.Random(seed)
        self._mains_hz = mains_hz
        self._mains_amplitude = mains_amplitude_v
        self._noise_floor = noise_floor_v
        self._max_amplitude = max_amplitude_v
        self._crosstalk = crosstalk
        self._buffer_limit = buffer_limit

        self._open = False
        self._next_sample_time = 0.0
        self._drive = [0.0] * len(self._channels)
        self._offset = [self._random.uniform(-2e-4, 2e-4) for _ in self._channels]
        self._drift_phase = [self._random.uniform(0, math.tau) for _ in self._channels]
        self._artefact_until = -1.0
        self._artefact_channel = 0
        self._artefact_amplitude = 0.0
        self._contact_quality = [1.0] * len(self._channels)
        self._stats = EmgSourceStats()

    # -- device lifecycle -----------------------------------------------------

    def open(self) -> None:
        self._open = True
        self._next_sample_time = self._clock.monotonic()

    def close(self) -> None:
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="emg",
            kind=DeviceKind.EMG,
            driver="simulated",
            connection="sim",
            capabilities=frozenset({DeviceCapability.SIMULATED}),
            extra={"channels": len(self._channels), "rate_hz": self._rate},
        )

    @property
    def sample_rate_hz(self) -> float:
        return self._rate

    @property
    def channels(self) -> Sequence[EmgChannelSpec]:
        return self._channels

    def dropped_samples(self) -> int:
        return self._stats.dropped

    # -- stimulus control -----------------------------------------------------

    def set_drive(self, channel: int, level: float) -> None:
        """Set the normalised muscle activation for one channel (0..1)."""
        self._drive[channel] = clamp(level)

    def set_drives(self, levels: Sequence[float]) -> None:
        """Set every channel at once."""
        for index, level in enumerate(levels[: len(self._drive)]):
            self._drive[index] = clamp(level)

    def set_flexor(self, level: float) -> None:
        """Convenience for the common two-channel layout."""
        for spec in self._channels:
            if spec.is_flexor:
                self._drive[spec.index] = clamp(level)

    def set_extensor(self, level: float) -> None:
        for spec in self._channels:
            if spec.is_extensor:
                self._drive[spec.index] = clamp(level)

    def inject_motion_artefact(
        self, channel: int = 0, amplitude: float = 1e-3, duration: float = 0.25
    ) -> None:
        """Simulate an electrode bump — used by fault-injection tests."""
        self._artefact_channel = channel
        self._artefact_amplitude = amplitude
        self._artefact_until = self._clock.monotonic() + duration

    def set_contact_quality(self, channel: int, quality: float) -> None:
        """Degrade a channel's electrode contact (1.0 = perfect, 0.0 = detached).

        Poor contact raises the noise floor and attenuates signal, which is
        exactly what the quality estimator must detect.
        """
        self._contact_quality[channel] = clamp(quality)

    # -- acquisition ----------------------------------------------------------

    def read(self) -> list[EmgSample]:
        if not self._open:
            return []
        now = self._clock.monotonic()
        period = 1.0 / self._rate
        samples: list[EmgSample] = []

        pending = int((now - self._next_sample_time) / period) + 1 if now >= self._next_sample_time else 0
        if pending > self._buffer_limit:
            # Consumer fell behind: model a hardware FIFO overrun.
            self._stats.dropped += pending - self._buffer_limit
            self._stats.overruns += 1
            self._next_sample_time = now - self._buffer_limit * period
            pending = self._buffer_limit

        for _ in range(pending):
            timestamp = self._next_sample_time
            samples.append(EmgSample(timestamp=timestamp, values=self._generate(timestamp)))
            self._next_sample_time += period

        self._stats.samples_read += len(samples)
        if samples:
            self._stats.last_timestamp = samples[-1].timestamp
        return samples

    def _generate(self, timestamp: float) -> tuple[float, ...]:
        """Synthesise one multi-channel sample at ``timestamp``."""
        values: list[float] = []
        for spec in self._channels:
            index = spec.index
            contact = self._contact_quality[index]

            # Crosstalk: neighbouring muscle groups are never perfectly isolated.
            drive = self._drive[index]
            for other, other_drive in enumerate(self._drive):
                if other != index:
                    drive += other_drive * self._crosstalk
            drive = clamp(drive)

            # Surface EMG ~ amplitude-modulated Gaussian noise.
            burst_rms = self._max_amplitude * (drive**1.3)
            signal = self._random.gauss(0.0, burst_rms) if burst_rms > 0 else 0.0

            # A lifted electrode transmits no muscle signal at all.
            signal *= contact

            # Baseline noise rises sharply as electrode contact degrades.
            noise_scale = self._noise_floor * (1.0 + 6.0 * (1.0 - contact))
            signal += self._random.gauss(0.0, noise_scale)

            # Mains interference and its third harmonic. A floating,
            # high-impedance input picks up far more of it than a bonded one —
            # which is the signature the quality estimator keys off.
            phase = math.tau * self._mains_hz * timestamp
            mains = self._mains_amplitude * (1.0 + 25.0 * (1.0 - contact))
            signal += mains * math.sin(phase) + mains * 0.3 * math.sin(3 * phase)

            # Slow baseline drift plus a static per-channel DC offset.
            signal += self._offset[index]
            signal += 3e-5 * math.sin(0.05 * math.tau * timestamp + self._drift_phase[index])

            # Transient motion artefact.
            if timestamp < self._artefact_until and index == self._artefact_channel:
                remaining = (self._artefact_until - timestamp) / 0.25
                signal += self._artefact_amplitude * remaining * math.sin(math.tau * 4.0 * timestamp)

            # Front-end saturation.
            limit = spec.full_scale_v / 2000.0  # amplifier gain of 1000 assumed
            values.append(max(-limit, min(limit, signal)))
        return tuple(values)
