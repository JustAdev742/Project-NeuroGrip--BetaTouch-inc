"""Training statistics and progress tracking.

Persisted per user, because the point of the training system is to show progress
over weeks, not within one session. The store is a plain JSON file written
atomically — small, inspectable, and trivially exportable for a clinician.

What is tracked and why:

* **Per-session records** — the raw history everything else derives from.
* **Per-exercise bests and adapted difficulty** — so a session resumes where the
  user actually is.
* **Rolling trend** — improving, plateaued or regressing. A plateau after steady
  improvement usually means the exercise has stopped being informative; a
  regression usually means fatigue or an electrode problem, and saying so is more
  useful than showing a lower number.
* **Streaks and totals** — the motivational surface for the achievements system.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from ..core.logging import get_logger
from ..core.types import clamp
from .exercises import Difficulty

__all__ = ["ExerciseProgress", "SessionRecord", "TrainingStats", "Trend"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SessionRecord:
    """One completed training session."""

    exercise: str
    difficulty: Difficulty
    trials: int
    mean_score: float
    best_score: float
    success_rate: float
    duration_s: float
    mean_latency_s: float = 0.0
    #: Unix timestamp.
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["difficulty"] = self.difficulty.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> SessionRecord:
        return cls(
            exercise=data["exercise"],
            difficulty=Difficulty(data.get("difficulty", "medium")),
            trials=int(data.get("trials", 0)),
            mean_score=float(data.get("mean_score", 0.0)),
            best_score=float(data.get("best_score", 0.0)),
            success_rate=float(data.get("success_rate", 0.0)),
            duration_s=float(data.get("duration_s", 0.0)),
            mean_latency_s=float(data.get("mean_latency_s", 0.0)),
            timestamp=float(data.get("timestamp", 0.0)),
        )


class Trend(str, Enum):
    """Direction of a user's recent performance."""

    IMPROVING = "improving"
    STEADY = "steady"
    DECLINING = "declining"
    INSUFFICIENT_DATA = "insufficient_data"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ").title()

    @property
    def symbol(self) -> str:
        return {"improving": "▲", "steady": "■", "declining": "▼"}.get(self.value, "·")


@dataclass(slots=True)
class ExerciseProgress:
    """Aggregated progress for one exercise."""

    exercise: str
    difficulty: Difficulty = Difficulty.MEDIUM
    sessions: int = 0
    total_trials: int = 0
    best_score: float = 0.0
    best_latency_s: float = 0.0
    last_score: float = 0.0
    total_duration_s: float = 0.0
    #: Mean score of the last few sessions, for the trend calculation.
    recent_scores: list[float] = field(default_factory=list)

    @property
    def mean_recent(self) -> float:
        return sum(self.recent_scores) / len(self.recent_scores) if self.recent_scores else 0.0

    @property
    def trend(self) -> Trend:
        """Compare the newest third of sessions with the oldest third."""
        if len(self.recent_scores) < 4:
            return Trend.INSUFFICIENT_DATA
        third = max(1, len(self.recent_scores) // 3)
        early = sum(self.recent_scores[:third]) / third
        late = sum(self.recent_scores[-third:]) / third
        delta = late - early
        if delta > 0.06:
            return Trend.IMPROVING
        if delta < -0.06:
            return Trend.DECLINING
        return Trend.STEADY

    @property
    def mastery(self) -> float:
        """0–1 mastery estimate combining level, consistency and best score."""
        return clamp(
            self.difficulty.scale * 0.5 + self.mean_recent * 0.3 + self.best_score * 0.2
        )


class TrainingStats:
    """Persistent training history for one user."""

    #: Number of recent session scores retained per exercise for trend analysis.
    RECENT_WINDOW = 12
    #: Full session records retained; older ones are summarised into the totals.
    HISTORY_LIMIT = 500

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._records: list[SessionRecord] = []
        self._progress: dict[str, ExerciseProgress] = {}
        self._streak_days = 0
        self._last_session_day: int | None = None
        if self._path is not None and self._path.exists():
            self.load()

    # -- recording ------------------------------------------------------------

    def record(self, record: SessionRecord) -> None:
        """Add a completed session and update the aggregates."""
        self._records.append(record)
        if len(self._records) > self.HISTORY_LIMIT:
            del self._records[0 : len(self._records) - self.HISTORY_LIMIT]

        progress = self._progress.setdefault(
            record.exercise, ExerciseProgress(exercise=record.exercise, difficulty=record.difficulty)
        )
        progress.sessions += 1
        progress.total_trials += record.trials
        progress.total_duration_s += record.duration_s
        progress.best_score = max(progress.best_score, record.best_score)
        progress.last_score = record.mean_score
        if record.mean_latency_s > 0:
            progress.best_latency_s = (
                min(progress.best_latency_s, record.mean_latency_s)
                if progress.best_latency_s > 0
                else record.mean_latency_s
            )
        progress.recent_scores.append(record.mean_score)
        del progress.recent_scores[: -self.RECENT_WINDOW]

        self._update_streak(record.timestamp or time.time())
        if self._path is not None:
            self.save()

    def _update_streak(self, timestamp: float) -> None:
        """Count consecutive calendar days with at least one session."""
        day = int(timestamp // 86400)
        # ``None``, not ``0``: day zero of the Unix epoch is a real day, and a
        # simulated clock starts there.
        if self._last_session_day is None:
            self._streak_days = 1
        elif day == self._last_session_day:
            pass  # same day; streak unchanged
        elif day == self._last_session_day + 1:
            self._streak_days += 1
        else:
            self._streak_days = 1
        self._last_session_day = day

    # -- queries --------------------------------------------------------------

    def progress(self, exercise: str) -> ExerciseProgress:
        return self._progress.get(exercise, ExerciseProgress(exercise=exercise))

    def all_progress(self) -> tuple[ExerciseProgress, ...]:
        return tuple(self._progress.values())

    def difficulty_for(self, exercise: str) -> Difficulty:
        """The adapted difficulty for an exercise."""
        return self.progress(exercise).difficulty

    def set_difficulty(self, exercise: str, difficulty: Difficulty) -> None:
        progress = self._progress.setdefault(
            exercise, ExerciseProgress(exercise=exercise, difficulty=difficulty)
        )
        progress.difficulty = difficulty
        if self._path is not None:
            self.save()

    def qualifying_streak(self, exercise: str, difficulty: Difficulty, threshold: float) -> int:
        """Consecutive recent sessions at ``difficulty`` scoring ``>= threshold``."""
        streak = 0
        for record in reversed(self._records):
            if record.exercise != exercise:
                continue
            if record.difficulty is not difficulty:
                break
            if record.mean_score < threshold:
                break
            streak += 1
        return streak

    def history(self, exercise: str | None = None, limit: int = 50) -> list[SessionRecord]:
        records = [r for r in self._records if exercise is None or r.exercise == exercise]
        return records[-limit:]

    def score_series(self, exercise: str, limit: int = 30) -> list[float]:
        """Mean scores over time, for the progress chart."""
        return [r.mean_score for r in self.history(exercise, limit)]

    @property
    def total_sessions(self) -> int:
        return len(self._records)

    @property
    def total_trials(self) -> int:
        return sum(p.total_trials for p in self._progress.values())

    @property
    def total_time_s(self) -> float:
        return sum(p.total_duration_s for p in self._progress.values())

    @property
    def streak_days(self) -> int:
        return self._streak_days

    @property
    def overall_mastery(self) -> float:
        """Mean mastery across every attempted exercise."""
        values = [p.mastery for p in self._progress.values()]
        return sum(values) / len(values) if values else 0.0

    def summary(self) -> dict[str, object]:
        """Everything the statistics screen needs, in one call."""
        return {
            "sessions": self.total_sessions,
            "trials": self.total_trials,
            "time_minutes": round(self.total_time_s / 60.0, 1),
            "streak_days": self.streak_days,
            "mastery": round(self.overall_mastery, 3),
            "exercises": {
                key: {
                    "difficulty": p.difficulty.value,
                    "sessions": p.sessions,
                    "best": round(p.best_score, 3),
                    "recent": round(p.mean_recent, 3),
                    "trend": p.trend.value,
                    "mastery": round(p.mastery, 3),
                }
                for key, p in self._progress.items()
            },
        }

    # -- persistence ----------------------------------------------------------

    def save(self, path: Path | str | None = None) -> None:
        """Write atomically so an interrupted save cannot corrupt the history."""
        target = Path(path) if path else self._path
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "streak_days": self._streak_days,
            "last_session_day": self._last_session_day,
            "progress": {
                key: {
                    "difficulty": p.difficulty.value,
                    "sessions": p.sessions,
                    "total_trials": p.total_trials,
                    "best_score": p.best_score,
                    "best_latency_s": p.best_latency_s,
                    "last_score": p.last_score,
                    "total_duration_s": p.total_duration_s,
                    "recent_scores": p.recent_scores,
                }
                for key, p in self._progress.items()
            },
            "records": [r.to_dict() for r in self._records],
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        temporary.replace(target)

    def load(self, path: Path | str | None = None) -> None:
        """Load history. A corrupt file is reported and ignored, never fatal."""
        target = Path(path) if path else self._path
        if target is None or not target.exists():
            return
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("could not read training statistics", path=str(target), error=str(exc))
            return

        self._streak_days = int(data.get("streak_days", 0))
        self._last_session_day = data.get("last_session_day")
        self._records = [SessionRecord.from_dict(r) for r in data.get("records", [])]
        self._progress = {}
        for key, entry in data.get("progress", {}).items():
            self._progress[key] = ExerciseProgress(
                exercise=key,
                difficulty=Difficulty(entry.get("difficulty", "medium")),
                sessions=int(entry.get("sessions", 0)),
                total_trials=int(entry.get("total_trials", 0)),
                best_score=float(entry.get("best_score", 0.0)),
                best_latency_s=float(entry.get("best_latency_s", 0.0)),
                last_score=float(entry.get("last_score", 0.0)),
                total_duration_s=float(entry.get("total_duration_s", 0.0)),
                recent_scores=[float(v) for v in entry.get("recent_scores", [])],
            )
