"""Training session: running exercises, scoring, difficulty adaptation.

Owns one exercise at a time, feeds it data, records results, and adapts the
difficulty between sessions.

The adaptation rule is deliberately asymmetric: **promote slowly, demote
quickly**. A user who is struggling should be moved back to a level they can
succeed at immediately, because repeated failure is what makes people abandon
rehabilitation exercises. A user who is doing well can afford to spend an extra
session at their current level before being pushed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..control.controller import HandState
from ..core.clock import Clock
from ..core.events import EventBus
from ..core.logging import get_logger
from ..core.topics import Topics
from ..emg.pipeline import EmgFrame
from .achievements import AchievementTracker
from .exercises import Difficulty, Exercise, ExerciseState, TrialResult, create_exercise
from .stats import SessionRecord, TrainingStats

__all__ = ["SessionSummary", "TrainingSession"]

log = get_logger(__name__)

#: Mean score at or above which the user is promoted (after ``PROMOTE_STREAK``).
PROMOTE_THRESHOLD = 0.80
#: Consecutive qualifying sessions required before promotion.
PROMOTE_STREAK = 2
#: Mean score below which the user is demoted immediately.
DEMOTE_THRESHOLD = 0.40


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Result of a completed session."""

    exercise: str
    difficulty: Difficulty
    trials: int
    mean_score: float
    best_score: float
    success_rate: float
    duration_s: float
    mean_latency_s: float = 0.0
    #: Difficulty for the next session, after adaptation.
    next_difficulty: Difficulty = Difficulty.MEDIUM
    promoted: bool = False
    demoted: bool = False
    achievements: tuple[str, ...] = field(default_factory=tuple)
    #: Plain-language coaching, shown on the results screen.
    advice: str = ""

    @property
    def stars(self) -> int:
        """0–3 stars, the shorthand the results screen leads with."""
        if self.mean_score >= 0.85:
            return 3
        if self.mean_score >= 0.65:
            return 2
        if self.mean_score >= 0.40:
            return 1
        return 0


class TrainingSession:
    """Runs one exercise at a time and records the outcome."""

    def __init__(
        self,
        clock: Clock,
        bus: EventBus,
        stats: TrainingStats | None = None,
        achievements: AchievementTracker | None = None,
    ) -> None:
        self._clock = clock
        self._bus = bus
        self._stats = stats or TrainingStats()
        self._achievements = achievements or AchievementTracker()
        self._exercise: Exercise | None = None
        self._difficulty = Difficulty.MEDIUM
        self._started_at = 0.0
        self._active = False
        self._state: ExerciseState | None = None
        self._summary: SessionSummary | None = None
        self._seen_results = 0

    # -- state ----------------------------------------------------------------

    @property
    def active(self) -> bool:
        return self._active

    @property
    def exercise(self) -> Exercise | None:
        return self._exercise

    @property
    def state(self) -> ExerciseState | None:
        """Latest exercise state, for the UI to render."""
        return self._state

    @property
    def summary(self) -> SessionSummary | None:
        """Summary of the most recently completed session."""
        return self._summary

    @property
    def stats(self) -> TrainingStats:
        return self._stats

    @property
    def achievements(self) -> AchievementTracker:
        return self._achievements

    @property
    def difficulty(self) -> Difficulty:
        return self._difficulty

    # -- control --------------------------------------------------------------

    def start(self, exercise_key: str, difficulty: Difficulty | None = None) -> bool:
        """Begin a session. Difficulty defaults to the user's adapted level."""
        try:
            exercise = create_exercise(exercise_key, self._clock)
        except KeyError:
            log.warning("unknown exercise", exercise=exercise_key)
            return False

        self._difficulty = difficulty or self._stats.difficulty_for(exercise_key)
        now = self._clock.monotonic()
        exercise.start(self._difficulty, now)

        self._exercise = exercise
        self._started_at = now
        self._active = True
        self._summary = None
        self._seen_results = 0
        self._state = None

        log.info("training session started", exercise=exercise_key, difficulty=self._difficulty.value)
        self._bus.publish(
            Topics.TRAINING_SESSION_STARTED,
            {"exercise": exercise_key, "difficulty": self._difficulty.value},
            source="training",
        )
        return True

    def stop(self, reason: str = "user stopped") -> SessionSummary | None:
        """End the session early; results so far are still recorded."""
        if not self._active or self._exercise is None:
            return None
        summary = self._finish(reason)
        return summary

    def update(self, emg: EmgFrame, hand: HandState, now: float) -> ExerciseState | None:
        """Feed one cycle of data to the active exercise."""
        if not self._active or self._exercise is None:
            return None

        self._state = self._exercise.update(emg, hand, now)

        # Publish each newly completed trial exactly once.
        results = self._exercise.results
        while self._seen_results < len(results):
            self._publish_trial(results[self._seen_results])
            self._seen_results += 1

        if self._exercise.finished:
            self._finish("completed")
        return self._state

    def _publish_trial(self, result: TrialResult) -> None:
        self._bus.publish(
            Topics.TRAINING_TRIAL,
            {
                "exercise": self._exercise.key if self._exercise else "",
                "index": result.index,
                "success": result.success,
                "score": round(result.score, 3),
                "latency_ms": round(result.latency_s * 1000, 1),
                "detail": result.detail,
            },
            source="training",
        )

    # -- completion -----------------------------------------------------------

    def _finish(self, reason: str) -> SessionSummary:
        exercise = self._exercise
        assert exercise is not None  # guarded by callers
        now = self._clock.monotonic()
        results = exercise.results

        mean_score = sum(r.score for r in results) / len(results) if results else 0.0
        best = max((r.score for r in results), default=0.0)
        successes = sum(1 for r in results if r.success)
        latencies = [r.latency_s for r in results if r.latency_s > 0]

        record = SessionRecord(
            exercise=exercise.key,
            difficulty=self._difficulty,
            trials=len(results),
            mean_score=mean_score,
            best_score=best,
            success_rate=successes / len(results) if results else 0.0,
            duration_s=now - self._started_at,
            mean_latency_s=sum(latencies) / len(latencies) if latencies else 0.0,
            timestamp=self._clock.wall(),
        )
        self._stats.record(record)

        next_difficulty, promoted, demoted = self._adapt(record)
        self._stats.set_difficulty(exercise.key, next_difficulty)

        unlocked = self._achievements.evaluate(record, self._stats)
        for achievement in unlocked:
            self._bus.publish(
                Topics.TRAINING_ACHIEVEMENT,
                {"key": achievement.key, "title": achievement.title, "detail": achievement.description},
                source="training",
            )

        summary = SessionSummary(
            exercise=exercise.key,
            difficulty=self._difficulty,
            trials=len(results),
            mean_score=mean_score,
            best_score=best,
            success_rate=record.success_rate,
            duration_s=record.duration_s,
            mean_latency_s=record.mean_latency_s,
            next_difficulty=next_difficulty,
            promoted=promoted,
            demoted=demoted,
            achievements=tuple(a.title for a in unlocked),
            advice=self._advise(exercise.key, record),
        )

        self._active = False
        self._summary = summary
        log.info(
            "training session finished",
            exercise=exercise.key,
            reason=reason,
            score=round(mean_score, 3),
            next_difficulty=next_difficulty.value,
        )
        self._bus.publish(Topics.TRAINING_SESSION_ENDED, summary, source="training")
        return summary

    def _adapt(self, record: SessionRecord) -> tuple[Difficulty, bool, bool]:
        """Choose the next difficulty. Promote slowly, demote quickly."""
        current = self._difficulty

        if record.trials < 3:
            # Too short to judge; leave the level alone.
            return current, False, False

        if record.mean_score < DEMOTE_THRESHOLD and current is not Difficulty.BEGINNER:
            return current.previous(), False, True

        if record.mean_score >= PROMOTE_THRESHOLD and current is not Difficulty.EXPERT:
            streak = self._stats.qualifying_streak(
                record.exercise, current, PROMOTE_THRESHOLD
            )
            if streak >= PROMOTE_STREAK:
                return current.next(), True, False

        return current, False, False

    def _advise(self, exercise_key: str, record: SessionRecord) -> str:
        """Plain-language coaching based on what the numbers show."""
        if record.trials == 0:
            return "No trials completed — try again when you are ready."
        if record.mean_score >= 0.85:
            return "Excellent control. Ready for the next difficulty."
        if record.success_rate < 0.4:
            return (
                "Most attempts were missed. Check your electrode placement and "
                "consider re-running the calibration wizard."
            )
        if exercise_key == "reaction" and record.mean_latency_s > 0.7:
            return (
                "Your reactions are slow but consistent. Try to contract sharply "
                "rather than gradually — the system responds to the onset."
            )
        if exercise_key == "accuracy":
            return "Focus on holding steady rather than hitting the target quickly."
        if exercise_key == "isolation":
            return "Relax the muscles you are not using; less effort often isolates better."
        if exercise_key == "consistency":
            return "Aim for the same effort every time, even if it is not your strongest."
        return "Good progress. Keep practising to build consistency."
