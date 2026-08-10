"""EMG calibration: data model, persistence and the guided wizard.

Every user's signals differ — muscle mass, electrode placement, skin impedance,
amputation level — and the same user's signals differ between sessions. Nothing
downstream may use raw microvolts; the pipeline works in *normalised activation*,
and this module is what defines that mapping.

Per channel we learn:

* ``rest_mean`` / ``rest_std`` — the baseline with the muscle relaxed. The onset
  threshold is placed at ``rest_mean + k·rest_std``, which adapts automatically to
  a noisy electrode rather than using a fixed microvolt number.
* ``mvc`` — maximum voluntary contraction, the top of the activation scale.
  Deliberately stored as a *usable* maximum (95th percentile of the hold) rather
  than the peak, because peak values are spiky and would make full activation
  unreachable in normal use.

The wizard is a state machine driven by the UI, one step at a time, so the same
implementation backs the touchscreen flow, the CLI (``neurogrip calibrate``) and
the tests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from ..core.clock import Clock
from ..core.errors import CalibrationError
from ..core.logging import get_logger
from ..core.ringbuffer import RunningStats, percentile
from ..core.types import clamp
from ..hal.emg.base import EmgChannelSpec

__all__ = [
    "CalibrationPhase",
    "CalibrationProgress",
    "CalibrationStep",
    "CalibrationWizard",
    "ChannelCalibration",
    "EmgCalibration",
]

log = get_logger(__name__)

#: Onset threshold = rest_mean + ONSET_SIGMA × rest_std. Three sigma keeps false
#: activations from baseline noise at roughly one in a thousand samples, which the
#: dwell time in the intent engine then removes entirely.
ONSET_SIGMA = 3.0

#: Minimum plausible ratio between MVC and rest. Below this the electrode is
#: almost certainly not making contact and the calibration must be rejected.
MIN_SNR_RATIO = 4.0


@dataclass(frozen=True, slots=True)
class ChannelCalibration:
    """Learned per-channel normalisation constants."""

    channel: int
    name: str = ""
    role: str = "auxiliary"
    rest_mean: float = 0.0
    rest_std: float = 1e-6
    mvc: float = 1e-4
    #: Envelope value at which activation reads 1.0 (a fraction of MVC, so the
    #: user does not have to maximally contract for full grip).
    full_scale: float = 1e-4
    #: Fraction of MVC used as ``full_scale``; 0.6 is comfortable for sustained use.
    effort_fraction: float = 0.6
    calibrated_at: float = 0.0
    samples: int = 0

    @property
    def onset_threshold(self) -> float:
        """Envelope level above which the channel counts as active."""
        return self.rest_mean + ONSET_SIGMA * self.rest_std

    @property
    def snr_ratio(self) -> float:
        """MVC relative to the rest baseline — the headline quality number."""
        return self.mvc / self.rest_mean if self.rest_mean > 1e-12 else 0.0

    @property
    def is_valid(self) -> bool:
        return (
            self.samples > 0
            and self.mvc > self.rest_mean
            and self.snr_ratio >= MIN_SNR_RATIO
            and self.full_scale > self.onset_threshold
        )

    def activation(self, envelope: float) -> float:
        """Map a raw envelope value to normalised activation in ``[0, 1]``.

        Below the onset threshold the result is exactly ``0.0``. That hard floor
        is deliberate: a resting user must produce *zero* activation, not a small
        number that could accumulate into motion.
        """
        threshold = self.onset_threshold
        if envelope <= threshold:
            return 0.0
        span = self.full_scale - threshold
        if span <= 1e-12:
            return 0.0
        return clamp((envelope - threshold) / span)

    def with_rest(self, mean: float, std: float, samples: int, now: float) -> ChannelCalibration:
        """Return a copy with an updated rest baseline (used by auto-recalibration)."""
        return ChannelCalibration(
            channel=self.channel,
            name=self.name,
            role=self.role,
            rest_mean=mean,
            rest_std=max(std, 1e-7),
            mvc=self.mvc,
            full_scale=self.full_scale,
            effort_fraction=self.effort_fraction,
            calibrated_at=now,
            samples=samples,
        )


@dataclass(slots=True)
class EmgCalibration:
    """Calibration for the whole array, plus provenance."""

    channels: dict[int, ChannelCalibration] = field(default_factory=dict)
    subject: str = "default"
    created_at: float = 0.0
    updated_at: float = 0.0
    #: Free-form note, e.g. "electrodes 3 cm distal, dry skin".
    notes: str = ""
    version: int = 1

    def get(self, channel: int) -> ChannelCalibration:
        """Calibration for ``channel``; falls back to a conservative default.

        The fallback is intentionally *insensitive* — a missing calibration must
        make the hand harder to trigger, never easier.
        """
        found = self.channels.get(channel)
        if found is not None:
            return found
        return ChannelCalibration(channel=channel, rest_mean=2e-5, rest_std=5e-6, mvc=5e-4, full_scale=3e-4)

    def set(self, calibration: ChannelCalibration) -> None:
        self.channels[calibration.channel] = calibration

    @property
    def is_valid(self) -> bool:
        return bool(self.channels) and all(c.is_valid for c in self.channels.values())

    @property
    def age_days(self) -> float:
        """Days since calibration; the UI nags after a week."""
        if not self.updated_at:
            return float("inf")
        return (time.time() - self.updated_at) / 86400.0

    def channels_by_role(self, role: str) -> list[ChannelCalibration]:
        return [c for c in self.channels.values() if c.role == role]

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "subject": self.subject,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
            "channels": [asdict(c) for c in sorted(self.channels.values(), key=lambda c: c.channel)],
        }

    @classmethod
    def from_dict(cls, data: dict) -> EmgCalibration:
        calibration = cls(
            subject=data.get("subject", "default"),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            notes=data.get("notes", ""),
            version=int(data.get("version", 1)),
        )
        for entry in data.get("channels", []):
            calibration.set(ChannelCalibration(**entry))
        return calibration

    def save(self, path: Path | str) -> None:
        """Write atomically — a half-written calibration file must never load."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(target)
        log.info("calibration saved", path=str(target), channels=len(self.channels))

    @classmethod
    def load(cls, path: Path | str) -> EmgCalibration:
        file_path = Path(path)
        if not file_path.exists():
            raise CalibrationError(f"calibration file not found: {file_path}")
        try:
            return cls.from_dict(json.loads(file_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid calibration file: {exc}") from exc


class CalibrationPhase(str, Enum):
    """Wizard phases, in execution order."""

    IDLE = "idle"
    #: "Relax completely." Establishes the noise floor.
    REST = "rest"
    #: "Close your hand as hard as is comfortable."
    FLEXOR_MAX = "flexor_max"
    #: "Open your hand as wide as is comfortable."
    EXTENSOR_MAX = "extensor_max"
    #: "Tense both together." Calibrates the cancel gesture margin.
    CO_CONTRACTION = "co_contraction"
    #: Validation of the collected data.
    VERIFY = "verify"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class CalibrationStep:
    """One instruction shown to the user."""

    phase: CalibrationPhase
    title: str
    instruction: str
    duration_s: float
    #: Channels whose data this step contributes to; empty means all.
    channels: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class CalibrationProgress:
    """Live wizard state for the UI to render."""

    phase: CalibrationPhase
    step_index: int
    step_count: int
    elapsed: float
    duration: float
    instruction: str
    title: str
    #: Live per-channel envelope, so the user sees their muscles responding.
    levels: tuple[float, ...] = ()
    message: str = ""

    @property
    def fraction(self) -> float:
        return clamp(self.elapsed / self.duration) if self.duration > 0 else 0.0

    @property
    def finished(self) -> bool:
        return self.phase in (CalibrationPhase.COMPLETE, CalibrationPhase.FAILED)


class CalibrationWizard:
    """Guided calibration procedure.

    Usage: call :meth:`start`, then feed every incoming envelope sample to
    :meth:`update` and render the returned :class:`CalibrationProgress`. The
    wizard advances its own steps against the clock; the caller only pumps data.
    """

    def __init__(
        self,
        channels: Sequence[EmgChannelSpec],
        clock: Clock,
        *,
        rest_duration: float = 5.0,
        contraction_duration: float = 4.0,
        effort_fraction: float = 0.6,
        settle_time: float = 1.0,
    ) -> None:
        self._channels = tuple(channels)
        self._clock = clock
        self._effort_fraction = effort_fraction
        #: Data from the first ``settle_time`` seconds of each step is discarded,
        #: because users take a moment to reach and hold the requested effort.
        self._settle = settle_time
        self._steps = self._build_steps(rest_duration, contraction_duration)
        self._index = 0
        self._phase = CalibrationPhase.IDLE
        self._step_started = 0.0
        self._rest_stats: dict[int, RunningStats] = {}
        self._peaks: dict[CalibrationPhase, dict[int, list[float]]] = {}
        self._latest: tuple[float, ...] = tuple(0.0 for _ in self._channels)
        self._message = ""
        self._result: EmgCalibration | None = None

    def _build_steps(self, rest: float, contraction: float) -> tuple[CalibrationStep, ...]:
        flexors = tuple(c.index for c in self._channels if c.is_flexor)
        extensors = tuple(c.index for c in self._channels if c.is_extensor)
        return (
            CalibrationStep(
                CalibrationPhase.REST,
                "Relax",
                "Rest your arm comfortably and stay completely relaxed.",
                rest,
            ),
            CalibrationStep(
                CalibrationPhase.FLEXOR_MAX,
                "Close",
                "Slowly close your hand and hold a firm, comfortable squeeze.",
                contraction,
                flexors,
            ),
            CalibrationStep(
                CalibrationPhase.EXTENSOR_MAX,
                "Open",
                "Open your hand wide and hold.",
                contraction,
                extensors,
            ),
            CalibrationStep(
                CalibrationPhase.CO_CONTRACTION,
                "Both",
                "Tense both muscle groups at once — this is your 'cancel' signal.",
                contraction * 0.75,
            ),
        )

    # -- control --------------------------------------------------------------

    def start(self) -> CalibrationProgress:
        """Begin the procedure, discarding any previous partial run."""
        self._index = 0
        self._rest_stats = {c.index: RunningStats() for c in self._channels}
        self._peaks = {step.phase: {c.index: [] for c in self._channels} for step in self._steps}
        self._result = None
        self._message = ""
        self._phase = self._steps[0].phase
        self._step_started = self._clock.monotonic()
        log.info("calibration started", channels=len(self._channels))
        return self.progress()

    def cancel(self) -> None:
        """Abort without producing a calibration."""
        self._phase = CalibrationPhase.IDLE
        self._message = "cancelled"

    def update(self, envelopes: Sequence[float]) -> CalibrationProgress:
        """Feed one set of per-channel envelope values and advance the wizard."""
        self._latest = tuple(envelopes)
        if self._phase in (
            CalibrationPhase.IDLE,
            CalibrationPhase.COMPLETE,
            CalibrationPhase.FAILED,
        ):
            return self.progress()

        step = self._steps[self._index]
        elapsed = self._clock.monotonic() - self._step_started

        if elapsed >= self._settle:
            for spec in self._channels:
                value = envelopes[spec.index] if spec.index < len(envelopes) else 0.0
                if step.phase is CalibrationPhase.REST:
                    self._rest_stats[spec.index].add(value)
                self._peaks[step.phase][spec.index].append(value)

        if elapsed >= step.duration_s:
            self._advance()
        return self.progress()

    def _advance(self) -> None:
        self._index += 1
        if self._index >= len(self._steps):
            self._finish()
            return
        self._phase = self._steps[self._index].phase
        self._step_started = self._clock.monotonic()

    def _finish(self) -> None:
        """Turn the collected samples into a validated :class:`EmgCalibration`."""
        self._phase = CalibrationPhase.VERIFY
        now = self._clock.monotonic()
        wall = time.time()
        calibration = EmgCalibration(subject="default", created_at=wall, updated_at=wall)
        problems: list[str] = []

        for spec in self._channels:
            stats = self._rest_stats[spec.index]
            if stats.count < 10:
                problems.append(f"{spec.name}: not enough rest data")
                continue

            # Use the phase that targets this channel; fall back to the strongest.
            candidates: list[float] = []
            for phase in (
                CalibrationPhase.FLEXOR_MAX if spec.is_flexor else CalibrationPhase.EXTENSOR_MAX,
                CalibrationPhase.CO_CONTRACTION,
            ):
                candidates.extend(self._peaks.get(phase, {}).get(spec.index, []))
            if not candidates:
                problems.append(f"{spec.name}: no contraction data")
                continue

            # 95th percentile, not max: peaks are spiky and unrepresentative of a
            # level the user can actually hold.
            mvc = percentile(candidates, 0.95)
            channel_cal = ChannelCalibration(
                channel=spec.index,
                name=spec.name,
                role=spec.role,
                rest_mean=stats.mean,
                rest_std=max(stats.std, 1e-7),
                mvc=mvc,
                full_scale=max(
                    mvc * self._effort_fraction,
                    stats.mean + ONSET_SIGMA * max(stats.std, 1e-7) * 2.0,
                ),
                effort_fraction=self._effort_fraction,
                calibrated_at=wall,
                samples=stats.count,
            )
            if not channel_cal.is_valid:
                problems.append(
                    f"{spec.name}: signal-to-noise too low "
                    f"({channel_cal.snr_ratio:.1f}× < {MIN_SNR_RATIO}×) — check electrode contact"
                )
            calibration.set(channel_cal)

        if problems:
            self._phase = CalibrationPhase.FAILED
            self._message = "; ".join(problems)
            log.warning("calibration failed", problems=self._message)
            return

        self._result = calibration
        self._phase = CalibrationPhase.COMPLETE
        self._message = "Calibration complete"
        log.info(
            "calibration complete",
            channels=len(calibration.channels),
            snr={c.name: round(c.snr_ratio, 1) for c in calibration.channels.values()},
            elapsed=round(now - self._step_started, 2),
        )

    # -- inspection -----------------------------------------------------------

    def progress(self) -> CalibrationProgress:
        if self._phase in (
            CalibrationPhase.IDLE,
            CalibrationPhase.COMPLETE,
            CalibrationPhase.FAILED,
            CalibrationPhase.VERIFY,
        ):
            return CalibrationProgress(
                phase=self._phase,
                step_index=self._index,
                step_count=len(self._steps),
                elapsed=0.0,
                duration=0.0,
                instruction="",
                title=self._phase.value.replace("_", " ").title(),
                levels=self._latest,
                message=self._message,
            )
        step = self._steps[self._index]
        return CalibrationProgress(
            phase=self._phase,
            step_index=self._index,
            step_count=len(self._steps),
            elapsed=self._clock.monotonic() - self._step_started,
            duration=step.duration_s,
            instruction=step.instruction,
            title=step.title,
            levels=self._latest,
            message=self._message,
        )

    @property
    def result(self) -> EmgCalibration | None:
        """The finished calibration, or ``None`` if the run did not complete."""
        return self._result

    @property
    def phase(self) -> CalibrationPhase:
        return self._phase

    @property
    def active(self) -> bool:
        """True only while a calibration run is in progress.

        Distinct from ``not progress().finished``: an idle wizard that has never
        been started is neither active nor finished, and treating it as active
        would silently suppress intent estimation.
        """
        return self._phase not in (
            CalibrationPhase.IDLE,
            CalibrationPhase.COMPLETE,
            CalibrationPhase.FAILED,
        )
