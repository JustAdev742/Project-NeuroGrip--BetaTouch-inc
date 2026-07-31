"""Training exercises.

Learning myoelectric control is genuinely hard. New users typically need weeks to
reach reliable control, and most of that time is spent without feedback that tells
them what they are doing wrong. These exercises exist to compress that: each one
isolates a single skill and gives immediate, quantitative feedback.

======================  =========================================================
exercise                skill trained
======================  =========================================================
Reaction Trainer        latency — contract *when* prompted
Grip Accuracy           proportionality — contract to a *specific* level
Finger Isolation        selectivity — move one digit without the others
Strength Meter          range — reach and hold a target effort
Consistency Tracker     repeatability — produce the same signal every time
======================  =========================================================

All five implement :class:`Exercise`, so the session runner, the scoring, the
statistics and the UI treat them uniformly. Adding a sixth means implementing one
class.

Every exercise is driven by :class:`~neurogrip.emg.pipeline.EmgFrame` and
:class:`~neurogrip.control.controller.HandState` — the same inputs the real
control path uses — so the skills transfer directly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from ..control.controller import HandState
from ..core.clock import Clock
from ..core.ringbuffer import RunningStats
from ..core.types import Finger, HandPose, clamp
from ..emg.pipeline import EmgFrame

__all__ = [
    "EXERCISES",
    "ConsistencyTracker",
    "Difficulty",
    "Exercise",
    "ExerciseState",
    "FingerIsolation",
    "GripAccuracy",
    "ReactionTrainer",
    "StrengthMeter",
    "TrialResult",
    "create_exercise",
]


class Difficulty(str, Enum):
    """Difficulty levels. Each exercise interprets these in its own terms."""

    BEGINNER = "beginner"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"

    @property
    def index(self) -> int:
        return list(Difficulty).index(self)

    @property
    def label(self) -> str:
        return self.value.title()

    @property
    def scale(self) -> float:
        """0.0 (beginner) … 1.0 (expert), for interpolating parameters."""
        return self.index / (len(Difficulty) - 1)

    def next(self) -> Difficulty:
        levels = list(Difficulty)
        return levels[min(len(levels) - 1, self.index + 1)]

    def previous(self) -> Difficulty:
        levels = list(Difficulty)
        return levels[max(0, self.index - 1)]


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Outcome of one repetition."""

    index: int
    success: bool
    #: Normalised score in ``[0, 1]``.
    score: float
    #: Seconds from prompt to response, where the exercise measures it.
    latency_s: float = 0.0
    #: Absolute error in whatever unit the exercise measures.
    error: float = 0.0
    detail: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class ExerciseState:
    """What the UI renders for the current moment of an exercise."""

    #: One-line instruction, e.g. "Hold at 60 %".
    prompt: str
    #: Progress through the current trial, ``[0, 1]``.
    progress: float = 0.0
    #: The value the user is being asked to produce, ``[0, 1]``.
    target: float = 0.0
    #: What they are currently producing, ``[0, 1]``.
    actual: float = 0.0
    #: Optional per-finger targets for isolation-style exercises.
    finger_targets: HandPose | None = None
    trial: int = 0
    trials_total: int = 0
    score: float = 0.0
    #: Set for one update when a trial completes, so the UI can flash feedback.
    last_result: TrialResult | None = None
    #: ``waiting`` | ``prompt`` | ``measuring`` | ``feedback`` | ``done``
    phase: str = "waiting"
    #: Encouragement or correction, shown under the prompt.
    feedback: str = ""


@runtime_checkable
class Exercise(Protocol):
    """A training exercise."""

    @property
    def key(self) -> str:
        """Stable identifier used in statistics and achievements."""
        ...

    @property
    def title(self) -> str: ...

    @property
    def description(self) -> str: ...

    def start(self, difficulty: Difficulty, now: float) -> None: ...

    def update(self, emg: EmgFrame, hand: HandState, now: float) -> ExerciseState: ...

    @property
    def finished(self) -> bool: ...

    @property
    def results(self) -> tuple[TrialResult, ...]: ...


class _ExerciseBase:
    """Shared trial bookkeeping."""

    key = "exercise"
    title = "Exercise"
    description = ""
    #: Trials per session at each difficulty.
    trials = 10

    def __init__(self, clock: Clock, *, seed: int = 7) -> None:
        self._clock = clock
        self._random = random.Random(seed)
        self._difficulty = Difficulty.MEDIUM
        self._results: list[TrialResult] = []
        self._trial = 0
        self._phase = "waiting"
        self._phase_started = 0.0
        self._last_result: TrialResult | None = None

    def start(self, difficulty: Difficulty, now: float) -> None:
        self._difficulty = difficulty
        self._results.clear()
        self._trial = 0
        self._last_result = None
        self._enter("waiting", now)

    def _enter(self, phase: str, now: float) -> None:
        self._phase = phase
        self._phase_started = now

    def _elapsed(self, now: float) -> float:
        return now - self._phase_started

    def _record(self, result: TrialResult) -> None:
        self._results.append(result)
        self._last_result = result
        self._trial += 1

    @property
    def finished(self) -> bool:
        return self._trial >= self.trials

    @property
    def results(self) -> tuple[TrialResult, ...]:
        return tuple(self._results)

    @property
    def difficulty(self) -> Difficulty:
        return self._difficulty

    @property
    def mean_score(self) -> float:
        return (
            sum(r.score for r in self._results) / len(self._results) if self._results else 0.0
        )

    @property
    def success_rate(self) -> float:
        return (
            sum(1 for r in self._results if r.success) / len(self._results)
            if self._results
            else 0.0
        )


class ReactionTrainer(_ExerciseBase):
    """Contract as soon as the prompt appears.

    Measures the round trip from visual cue to detected intent — the user's
    reaction plus the system's own latency (filtering, dwell, classification).
    Because the same pipeline is used here as in normal operation, improvements
    shown here are improvements the user will actually feel.

    The wait before each prompt is randomised so it cannot be anticipated.
    """

    key = "reaction"
    title = "Reaction Trainer"
    description = "Contract as soon as the target appears. Trains response speed."
    trials = 12

    #: Reaction time counted as a perfect score, in seconds.
    PERFECT_S = 0.25
    #: Reaction time scoring zero.
    FAIL_S = 1.6

    def __init__(self, clock: Clock, *, seed: int = 7) -> None:
        super().__init__(clock, seed=seed)
        self._wait = 1.5
        self._prompt_at = 0.0
        self._false_starts = 0

    def start(self, difficulty: Difficulty, now: float) -> None:
        super().start(difficulty, now)
        self._false_starts = 0
        self._schedule(now)

    def _schedule(self, now: float) -> None:
        # Harder levels use a wider, less predictable window.
        spread = 1.0 + 2.5 * self._difficulty.scale
        self._wait = self._random.uniform(0.8, 0.8 + spread)
        self._enter("waiting", now)

    def update(self, emg: EmgFrame, hand: HandState, now: float) -> ExerciseState:
        activation = emg.flexor
        threshold = 0.30 - 0.10 * self._difficulty.scale

        if self._phase == "waiting":
            if activation > threshold:
                # Contracted before the prompt: a false start costs a trial.
                self._false_starts += 1
                self._record(
                    TrialResult(
                        index=self._trial,
                        success=False,
                        score=0.0,
                        detail="false start — wait for the prompt",
                        timestamp=now,
                    )
                )
                self._enter("feedback", now)
                return self._state("Too early!", feedback="Wait for the target to appear.")
            if self._elapsed(now) >= self._wait:
                self._prompt_at = now
                self._enter("prompt", now)
            return self._state("Get ready…", progress=clamp(self._elapsed(now) / self._wait))

        if self._phase == "prompt":
            if activation > threshold:
                latency = now - self._prompt_at
                score = clamp(
                    1.0 - (latency - self.PERFECT_S) / (self.FAIL_S - self.PERFECT_S)
                )
                self._record(
                    TrialResult(
                        index=self._trial,
                        success=latency < self.FAIL_S,
                        score=score,
                        latency_s=latency,
                        detail=f"{latency * 1000:.0f} ms",
                        timestamp=now,
                    )
                )
                self._enter("feedback", now)
                return self._state("GO!", actual=activation)
            if self._elapsed(now) > self.FAIL_S * 2:
                self._record(
                    TrialResult(
                        index=self._trial, success=False, score=0.0, detail="no response", timestamp=now
                    )
                )
                self._enter("feedback", now)
            return self._state("GO!", target=1.0, actual=activation, progress=1.0)

        # feedback
        if self._elapsed(now) > 1.0 and not self.finished:
            self._schedule(now)
        return self._state("", feedback=self._feedback_text())

    def _feedback_text(self) -> str:
        if self._last_result is None:
            return ""
        if not self._last_result.success:
            return self._last_result.detail
        latency = self._last_result.latency_s
        if latency < 0.35:
            return f"Excellent — {latency * 1000:.0f} ms"
        if latency < 0.6:
            return f"Good — {latency * 1000:.0f} ms"
        return f"{latency * 1000:.0f} ms — try to react sooner"

    def _state(self, prompt: str, **kwargs) -> ExerciseState:
        return ExerciseState(
            prompt=prompt,
            trial=self._trial,
            trials_total=self.trials,
            score=self.mean_score,
            last_result=self._last_result,
            phase=self._phase,
            **kwargs,
        )


class GripAccuracy(_ExerciseBase):
    """Hold a specific effort level.

    Proportional control is the difference between picking up an egg and picking
    up a hammer. This asks for a randomly chosen target level and scores how
    closely the user matches it, and how steadily they hold it.
    """

    key = "accuracy"
    title = "Grip Accuracy"
    description = "Match and hold the target grip level. Trains fine proportional control."
    trials = 8

    #: Seconds the target must be held.
    HOLD_S = 1.5

    def __init__(self, clock: Clock, *, seed: int = 11) -> None:
        super().__init__(clock, seed=seed)
        self._target = 0.5
        self._samples = RunningStats()
        self._in_band_since = 0.0

    def start(self, difficulty: Difficulty, now: float) -> None:
        super().start(difficulty, now)
        self._next_target(now)

    def _next_target(self, now: float) -> None:
        self._target = round(self._random.uniform(0.25, 0.9), 2)
        self._samples.reset()
        self._in_band_since = 0.0
        self._enter("measuring", now)

    @property
    def tolerance(self) -> float:
        """Accepted band around the target — narrows with difficulty."""
        return 0.18 - 0.13 * self._difficulty.scale

    def update(self, emg: EmgFrame, hand: HandState, now: float) -> ExerciseState:
        activation = emg.flexor

        if self._phase == "measuring":
            error = abs(activation - self._target)
            in_band = error <= self.tolerance

            if in_band:
                if self._in_band_since == 0.0:
                    self._in_band_since = now
                self._samples.add(activation)
            else:
                # Leaving the band restarts the hold: the skill is *staying*
                # there, not passing through.
                self._in_band_since = 0.0
                self._samples.reset()

            held = now - self._in_band_since if self._in_band_since else 0.0
            if held >= self.HOLD_S:
                mean_error = abs(self._samples.mean - self._target)
                steadiness = clamp(1.0 - self._samples.std / max(0.05, self.tolerance))
                accuracy = clamp(1.0 - mean_error / max(1e-6, self.tolerance))
                score = clamp(accuracy * 0.65 + steadiness * 0.35)
                self._record(
                    TrialResult(
                        index=self._trial,
                        success=True,
                        score=score,
                        error=mean_error,
                        detail=f"±{mean_error * 100:.0f}%, steadiness {steadiness * 100:.0f}%",
                        timestamp=now,
                    )
                )
                self._enter("feedback", now)
            elif self._elapsed(now) > 15.0:
                self._record(
                    TrialResult(
                        index=self._trial,
                        success=False,
                        score=0.0,
                        error=error,
                        detail="ran out of time",
                        timestamp=now,
                    )
                )
                self._enter("feedback", now)

            return ExerciseState(
                prompt=f"Hold at {self._target * 100:.0f}%",
                progress=clamp(held / self.HOLD_S),
                target=self._target,
                actual=activation,
                trial=self._trial,
                trials_total=self.trials,
                score=self.mean_score,
                last_result=self._last_result,
                phase="measuring",
                feedback="Steady…" if in_band else self._nudge(activation),
            )

        if self._elapsed(now) > 1.2 and not self.finished:
            self._next_target(now)
        return ExerciseState(
            prompt="",
            target=self._target,
            actual=activation,
            trial=self._trial,
            trials_total=self.trials,
            score=self.mean_score,
            last_result=self._last_result,
            phase=self._phase,
            feedback=self._last_result.detail if self._last_result else "",
        )

    def _nudge(self, activation: float) -> str:
        return "A little more" if activation < self._target else "Ease off slightly"


class FingerIsolation(_ExerciseBase):
    """Move one finger while keeping the others still.

    Selectivity is what unlocks the precision grips. With two EMG channels the
    hand cannot address fingers independently by muscle alone, so this exercise
    trains the *pattern* — a controlled, low-amplitude contraction that the
    system maps to a single-digit grip — and scores how much the other fingers
    moved.
    """

    key = "isolation"
    title = "Finger Isolation"
    description = "Move the highlighted finger while keeping the others still."
    trials = 8

    HOLD_S = 1.2

    def __init__(self, clock: Clock, *, seed: int = 13) -> None:
        super().__init__(clock, seed=seed)
        self._finger = Finger.INDEX
        self._target_closure = 0.7
        self._worst_leak = 0.0

    def start(self, difficulty: Difficulty, now: float) -> None:
        super().start(difficulty, now)
        self._next_finger(now)

    def _next_finger(self, now: float) -> None:
        # Beginners get the digits with the most independent control.
        pool = (
            [Finger.INDEX, Finger.THUMB]
            if self._difficulty.scale < 0.4
            else list(Finger)
        )
        self._finger = self._random.choice(pool)
        self._target_closure = 0.6 + 0.2 * self._difficulty.scale
        self._worst_leak = 0.0
        self._enter("measuring", now)

    @property
    def leak_tolerance(self) -> float:
        """How much the other fingers may move before it counts against you."""
        return 0.25 - 0.17 * self._difficulty.scale

    def update(self, emg: EmgFrame, hand: HandState, now: float) -> ExerciseState:
        pose = hand.pose
        target_value = pose[self._finger]
        leak = max(
            (pose[f] for f in Finger if f is not self._finger),
            default=0.0,
        )
        self._worst_leak = max(self._worst_leak, leak)

        reached = target_value >= self._target_closure * 0.85
        if reached and self._elapsed(now) >= self.HOLD_S:
            isolation = clamp(1.0 - self._worst_leak / max(1e-6, self.leak_tolerance))
            score = clamp(isolation * 0.75 + clamp(target_value) * 0.25)
            self._record(
                TrialResult(
                    index=self._trial,
                    success=self._worst_leak <= self.leak_tolerance,
                    score=score,
                    error=self._worst_leak,
                    detail=f"other fingers moved {self._worst_leak * 100:.0f}%",
                    timestamp=now,
                )
            )
            self._enter("feedback", now)
        elif self._elapsed(now) > 12.0 and self._phase == "measuring":
            self._record(
                TrialResult(
                    index=self._trial, success=False, score=0.0, detail="ran out of time", timestamp=now
                )
            )
            self._enter("feedback", now)

        if self._phase == "feedback" and self._elapsed(now) > 1.2 and not self.finished:
            self._next_finger(now)

        targets = HandPose.open_hand().with_finger(self._finger, self._target_closure)
        return ExerciseState(
            prompt=f"Move your {self._finger.label.lower()}",
            progress=clamp(target_value / max(1e-6, self._target_closure)),
            target=self._target_closure,
            actual=target_value,
            finger_targets=targets,
            trial=self._trial,
            trials_total=self.trials,
            score=self.mean_score,
            last_result=self._last_result,
            phase=self._phase,
            feedback=(
                "Keep the others relaxed" if leak > self.leak_tolerance * 0.6 else "Good isolation"
            ),
        )


class StrengthMeter(_ExerciseBase):
    """Reach and hold a high effort level.

    Two purposes. It builds the endurance a user needs for sustained grips, and
    it tracks the *decline* in achievable effort across a session, which is a
    direct measure of muscle fatigue — the thing that makes control degrade
    after twenty minutes of use.
    """

    key = "strength"
    title = "Strength Meter"
    description = "Contract as strongly as you comfortably can, and hold."
    trials = 6

    HOLD_S = 2.5

    def __init__(self, clock: Clock, *, seed: int = 17) -> None:
        super().__init__(clock, seed=seed)
        self._peak = 0.0
        self._hold_start = 0.0
        self._samples = RunningStats()

    def start(self, difficulty: Difficulty, now: float) -> None:
        super().start(difficulty, now)
        self._begin_trial(now)

    def _begin_trial(self, now: float) -> None:
        self._peak = 0.0
        self._hold_start = 0.0
        self._samples.reset()
        self._enter("measuring", now)

    @property
    def target_level(self) -> float:
        return 0.55 + 0.3 * self._difficulty.scale

    def update(self, emg: EmgFrame, hand: HandState, now: float) -> ExerciseState:
        activation = emg.flexor
        self._peak = max(self._peak, activation)

        if self._phase == "measuring":
            above = activation >= self.target_level
            if above:
                if self._hold_start == 0.0:
                    self._hold_start = now
                self._samples.add(activation)
            else:
                self._hold_start = 0.0

            held = now - self._hold_start if self._hold_start else 0.0
            if held >= self.HOLD_S:
                consistency = clamp(1.0 - self._samples.coefficient_of_variation * 3.0)
                score = clamp(clamp(self._samples.mean) * 0.6 + consistency * 0.4)
                self._record(
                    TrialResult(
                        index=self._trial,
                        success=True,
                        score=score,
                        detail=f"held {self._samples.mean * 100:.0f}% for {held:.1f} s",
                        timestamp=now,
                    )
                )
                self._enter("feedback", now)
            elif self._elapsed(now) > 20.0:
                self._record(
                    TrialResult(
                        index=self._trial,
                        success=False,
                        score=clamp(self._peak * 0.5),
                        detail=f"peaked at {self._peak * 100:.0f}%",
                        timestamp=now,
                    )
                )
                self._enter("feedback", now)
        elif self._elapsed(now) > 2.5 and not self.finished:
            # Rest between repetitions matters: without it this measures
            # fatigue rather than strength.
            self._begin_trial(now)

        return ExerciseState(
            prompt=f"Hold above {self.target_level * 100:.0f}%",
            progress=clamp((now - self._hold_start) / self.HOLD_S) if self._hold_start else 0.0,
            target=self.target_level,
            actual=activation,
            trial=self._trial,
            trials_total=self.trials,
            score=self.mean_score,
            last_result=self._last_result,
            phase=self._phase,
            feedback="Rest" if self._phase == "feedback" else f"Peak {self._peak * 100:.0f}%",
        )

    @property
    def fatigue_index(self) -> float:
        """Drop from the first to the last repetition, in ``[0, 1]``.

        A large value means the user's achievable effort fell during the session,
        which the statistics screen surfaces as a recommendation to rest.
        """
        if len(self._results) < 3:
            return 0.0
        first = self._results[0].score
        last = self._results[-1].score
        return clamp((first - last) / max(1e-6, first))


class ConsistencyTracker(_ExerciseBase):
    """Repeat the same contraction over and over.

    Repeatability is what a classifier needs. A user whose "close" signal varies
    from trial to trial will get inconsistent behaviour no matter how good the
    algorithm is. This scores the *variance* between repetitions rather than the
    accuracy of any one of them.
    """

    key = "consistency"
    title = "Consistency"
    description = "Repeat the same contraction. Scores how alike your repetitions are."
    trials = 10

    HOLD_S = 1.0

    def __init__(self, clock: Clock, *, seed: int = 19) -> None:
        super().__init__(clock, seed=seed)
        self._peaks: list[float] = []
        self._current = RunningStats()
        self._hold_start = 0.0
        self._reference = 0.55

    def start(self, difficulty: Difficulty, now: float) -> None:
        super().start(difficulty, now)
        self._peaks.clear()
        self._reference = 0.45 + 0.2 * self._difficulty.scale
        self._begin_trial(now)

    def _begin_trial(self, now: float) -> None:
        self._current.reset()
        self._hold_start = 0.0
        self._enter("measuring", now)

    def update(self, emg: EmgFrame, hand: HandState, now: float) -> ExerciseState:
        activation = emg.flexor
        active = activation > 0.15

        if self._phase == "measuring":
            if active:
                if self._hold_start == 0.0:
                    self._hold_start = now
                self._current.add(activation)
            elif self._hold_start:
                # Contraction released before the hold completed.
                self._hold_start = 0.0
                self._current.reset()

            held = now - self._hold_start if self._hold_start else 0.0
            if held >= self.HOLD_S:
                level = self._current.mean
                self._peaks.append(level)
                deviation = (
                    abs(level - sum(self._peaks[:-1]) / len(self._peaks[:-1]))
                    if len(self._peaks) > 1
                    else 0.0
                )
                score = clamp(1.0 - deviation / max(0.05, 0.25 - 0.15 * self._difficulty.scale))
                self._record(
                    TrialResult(
                        index=self._trial,
                        success=deviation < 0.15,
                        score=score if len(self._peaks) > 1 else 1.0,
                        error=deviation,
                        detail=f"level {level * 100:.0f}%, deviation {deviation * 100:.0f}%",
                        timestamp=now,
                    )
                )
                self._enter("feedback", now)
            elif self._elapsed(now) > 15.0:
                self._record(
                    TrialResult(
                        index=self._trial, success=False, score=0.0, detail="no contraction", timestamp=now
                    )
                )
                self._enter("feedback", now)
        elif self._elapsed(now) > 1.2 and not self.finished:
            self._begin_trial(now)

        return ExerciseState(
            prompt="Contract, hold, release — the same way each time",
            progress=clamp((now - self._hold_start) / self.HOLD_S) if self._hold_start else 0.0,
            target=self._average_peak,
            actual=activation,
            trial=self._trial,
            trials_total=self.trials,
            score=self.mean_score,
            last_result=self._last_result,
            phase=self._phase,
            feedback=self._last_result.detail if self._last_result else "",
        )

    @property
    def _average_peak(self) -> float:
        return sum(self._peaks) / len(self._peaks) if self._peaks else self._reference

    @property
    def coefficient_of_variation(self) -> float:
        """Std/mean across repetitions — the headline consistency number.

        Below 0.10 is clinically good repeatability; above 0.25 usually means
        the electrodes or the calibration need attention rather than the user.
        """
        if len(self._peaks) < 2:
            return 0.0
        mean = sum(self._peaks) / len(self._peaks)
        if mean < 1e-6:
            return 0.0
        variance = sum((p - mean) ** 2 for p in self._peaks) / (len(self._peaks) - 1)
        return math.sqrt(variance) / mean


#: Registry of exercise classes, keyed by their stable identifier.
EXERCISES: dict[str, type[_ExerciseBase]] = {
    ReactionTrainer.key: ReactionTrainer,
    GripAccuracy.key: GripAccuracy,
    FingerIsolation.key: FingerIsolation,
    StrengthMeter.key: StrengthMeter,
    ConsistencyTracker.key: ConsistencyTracker,
}


def create_exercise(key: str, clock: Clock, *, seed: int = 7) -> _ExerciseBase:
    """Instantiate an exercise by key. Raises ``KeyError`` for unknown keys."""
    return EXERCISES[key](clock, seed=seed)
