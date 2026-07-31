"""Achievements.

Motivation is a real engineering concern here, not decoration. Myoelectric
training is repetitive and the gains are slow and invisible; drop-out is the
normal outcome. Achievements make progress legible.

They are chosen to reward the behaviour that actually produces control:

* **turning up** — streaks and totals, because frequency beats intensity;
* **consistency** — rewarded more than peak performance, because a repeatable
  signal is what the classifier needs;
* **breadth** — trying every exercise, since the skills are complementary.

Nothing here gates functionality. An achievement never unlocks a feature; the
hand is fully capable from the first minute.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.logging import get_logger
from .exercises import Difficulty
from .stats import SessionRecord, TrainingStats

__all__ = ["ACHIEVEMENTS", "Achievement", "AchievementTracker"]

log = get_logger(__name__)

#: Signature of an achievement condition.
Condition = Callable[[SessionRecord, TrainingStats], bool]


@dataclass(frozen=True, slots=True)
class Achievement:
    """A single unlockable."""

    key: str
    title: str
    description: str
    condition: Condition
    #: ``bronze`` | ``silver`` | ``gold``
    tier: str = "bronze"
    icon: str = "★"
    #: Hidden achievements are not listed until unlocked.
    hidden: bool = False

    def check(self, record: SessionRecord, stats: TrainingStats) -> bool:
        try:
            return self.condition(record, stats)
        except Exception as exc:
            log.warning("achievement condition raised", achievement=self.key, error=str(exc))
            return False


ACHIEVEMENTS: tuple[Achievement, ...] = (
    Achievement(
        key="first_session",
        title="First Steps",
        description="Complete your first training session.",
        condition=lambda r, s: s.total_sessions >= 1,
        tier="bronze",
        icon="🌱",
    ),
    Achievement(
        key="ten_sessions",
        title="Getting the Hang of It",
        description="Complete ten training sessions.",
        condition=lambda r, s: s.total_sessions >= 10,
        tier="silver",
        icon="💪",
    ),
    Achievement(
        key="hundred_sessions",
        title="Dedicated",
        description="Complete one hundred training sessions.",
        condition=lambda r, s: s.total_sessions >= 100,
        tier="gold",
        icon="🏅",
    ),
    Achievement(
        key="streak_3",
        title="Three in a Row",
        description="Train on three consecutive days.",
        condition=lambda r, s: s.streak_days >= 3,
        tier="bronze",
        icon="🔥",
    ),
    Achievement(
        key="streak_7",
        title="Week Strong",
        description="Train every day for a week.",
        condition=lambda r, s: s.streak_days >= 7,
        tier="silver",
        icon="🔥",
    ),
    Achievement(
        key="streak_30",
        title="Habit Formed",
        description="Train every day for a month.",
        condition=lambda r, s: s.streak_days >= 30,
        tier="gold",
        icon="🔥",
    ),
    Achievement(
        key="perfect_session",
        title="Flawless",
        description="Score above 95 % across a whole session.",
        condition=lambda r, s: r.mean_score >= 0.95 and r.trials >= 5,
        tier="gold",
        icon="✨",
    ),
    Achievement(
        key="quick_draw",
        title="Quick Draw",
        description="Average under 300 ms in the Reaction Trainer.",
        condition=lambda r, s: r.exercise == "reaction"
        and 0 < r.mean_latency_s < 0.30
        and r.trials >= 8,
        tier="gold",
        icon="⚡",
    ),
    Achievement(
        key="steady_hand",
        title="Steady Hand",
        description="Score above 90 % in Grip Accuracy at Hard or above.",
        condition=lambda r, s: r.exercise == "accuracy"
        and r.mean_score >= 0.90
        and r.difficulty.index >= Difficulty.HARD.index,
        tier="gold",
        icon="🎯",
    ),
    Achievement(
        key="isolationist",
        title="Independent Thinker",
        description="Complete Finger Isolation at Expert difficulty.",
        condition=lambda r, s: r.exercise == "isolation"
        and r.difficulty is Difficulty.EXPERT
        and r.success_rate >= 0.6,
        tier="gold",
        icon="🖐",
    ),
    Achievement(
        key="consistent",
        title="Metronome",
        description="Score above 85 % in the Consistency exercise.",
        condition=lambda r, s: r.exercise == "consistency" and r.mean_score >= 0.85,
        tier="silver",
        icon="📏",
    ),
    Achievement(
        key="all_rounder",
        title="All-Rounder",
        description="Complete at least one session of every exercise.",
        condition=lambda r, s: len(s.all_progress()) >= 5,
        tier="silver",
        icon="🎖",
    ),
    Achievement(
        key="promoted",
        title="Levelling Up",
        description="Reach Hard difficulty in any exercise.",
        condition=lambda r, s: any(
            p.difficulty.index >= Difficulty.HARD.index for p in s.all_progress()
        ),
        tier="silver",
        icon="⬆",
    ),
    Achievement(
        key="expert",
        title="Expert Control",
        description="Reach Expert difficulty in any exercise.",
        condition=lambda r, s: any(
            p.difficulty is Difficulty.EXPERT for p in s.all_progress()
        ),
        tier="gold",
        icon="👑",
    ),
    Achievement(
        key="hour_of_practice",
        title="An Hour In",
        description="Accumulate one hour of training.",
        condition=lambda r, s: s.total_time_s >= 3600,
        tier="silver",
        icon="⏱",
    ),
    Achievement(
        key="comeback",
        title="Comeback",
        description="Improve on a previous session by more than 30 %.",
        condition=lambda r, s: (
            len(s.progress(r.exercise).recent_scores) >= 2
            and r.mean_score - min(s.progress(r.exercise).recent_scores[:-1]) > 0.30
        ),
        tier="bronze",
        icon="📈",
        hidden=True,
    ),
)


class AchievementTracker:
    """Evaluates achievements and remembers which are unlocked."""

    def __init__(self, achievements: tuple[Achievement, ...] = ACHIEVEMENTS) -> None:
        self._achievements = achievements
        self._unlocked: dict[str, float] = {}

    def evaluate(self, record: SessionRecord, stats: TrainingStats) -> list[Achievement]:
        """Check every locked achievement; returns those newly unlocked."""
        unlocked: list[Achievement] = []
        for achievement in self._achievements:
            if achievement.key in self._unlocked:
                continue
            if achievement.check(record, stats):
                self._unlocked[achievement.key] = record.timestamp
                unlocked.append(achievement)
                log.info("achievement unlocked", key=achievement.key, title=achievement.title)
        return unlocked

    # -- queries --------------------------------------------------------------

    def is_unlocked(self, key: str) -> bool:
        return key in self._unlocked

    @property
    def unlocked(self) -> tuple[Achievement, ...]:
        return tuple(a for a in self._achievements if a.key in self._unlocked)

    @property
    def locked(self) -> tuple[Achievement, ...]:
        """Locked achievements, excluding hidden ones."""
        return tuple(
            a for a in self._achievements if a.key not in self._unlocked and not a.hidden
        )

    @property
    def completion(self) -> float:
        return len(self._unlocked) / len(self._achievements) if self._achievements else 0.0

    def unlocked_at(self, key: str) -> float | None:
        return self._unlocked.get(key)

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict[str, float]:
        return dict(self._unlocked)

    def load(self, data: dict[str, float]) -> None:
        self._unlocked = {k: float(v) for k, v in data.items()}
