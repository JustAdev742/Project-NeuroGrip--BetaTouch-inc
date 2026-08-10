"""Training environment.

Learning to control a myoelectric prosthesis typically takes weeks, and most of
that time is spent guessing. This package exists to shorten it: five exercises
that each isolate one skill, immediate quantitative feedback, difficulty that
adapts to the individual, and progress that is visible over weeks rather than
minutes.

::

    TrainingSession
      ├─ Exercise        reaction | accuracy | isolation | strength | consistency
      ├─ TrainingStats   persistent history, trends, adapted difficulty
      └─ Achievements    motivation; never gates functionality

Exercises consume the same :class:`~neurogrip.emg.pipeline.EmgFrame` and
:class:`~neurogrip.control.controller.HandState` as the real control path, so
what a user practises is exactly what they will use. Training Mode runs with AI
assistance **off**, because an exercise that measures the user while an AI
quietly corrects them measures nothing.
"""

from __future__ import annotations

from .achievements import ACHIEVEMENTS, Achievement, AchievementTracker
from .exercises import (
    EXERCISES,
    ConsistencyTracker,
    Difficulty,
    Exercise,
    ExerciseState,
    FingerIsolation,
    GripAccuracy,
    ReactionTrainer,
    StrengthMeter,
    TrialResult,
    create_exercise,
)
from .session import SessionSummary, TrainingSession
from .stats import ExerciseProgress, SessionRecord, TrainingStats, Trend

__all__ = [
    "ACHIEVEMENTS",
    "EXERCISES",
    "Achievement",
    "AchievementTracker",
    "ConsistencyTracker",
    "Difficulty",
    "Exercise",
    "ExerciseProgress",
    "ExerciseState",
    "FingerIsolation",
    "GripAccuracy",
    "ReactionTrainer",
    "SessionRecord",
    "SessionSummary",
    "StrengthMeter",
    "TrainingSession",
    "TrainingStats",
    "Trend",
    "TrialResult",
    "create_exercise",
]
