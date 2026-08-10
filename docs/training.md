# Training

Learning myoelectric control takes weeks, and most of that time is spent without
feedback that tells the user what they are doing wrong. Drop-out is the normal
outcome. This subsystem exists to compress that: five exercises that each isolate
one skill, with immediate quantitative feedback.

## The exercises

| Exercise | Skill | Measures |
|---|---|---|
| **Reaction Trainer** | Latency | Time from visual cue to detected intent |
| **Grip Accuracy** | Proportionality | Error and steadiness against a target level |
| **Finger Isolation** | Selectivity | How much the *other* fingers moved |
| **Strength Meter** | Range and endurance | Sustained effort, and the decline across a session |
| **Consistency** | Repeatability | Variance *between* repetitions |

All five consume the same `EmgFrame` and `HandState` as the real control path, so
the skills transfer directly.

### Reaction Trainer

A prompt appears after a randomised delay; contract as fast as possible. Measures
the full round trip — the user's reaction *plus* the system's own latency
(filtering, dwell, classification) — so an improvement here is an improvement the
user will actually feel.

False starts cost a trial. The delay is randomised (and the spread widens with
difficulty) so it cannot be anticipated.

### Grip Accuracy

Hold a randomly chosen effort level within a tolerance band for 1.5 s. Scores
accuracy (65 %) and steadiness (35 %).

Leaving the band **restarts the hold**: the skill is *staying* there, not passing
through. The band narrows from ±18 % at Beginner to ±5 % at Expert.

Proportional control is the difference between picking up an egg and picking up a
hammer.

### Finger Isolation

Move the highlighted finger while keeping the others still. Scores how much the
others moved.

With two EMG channels the hand cannot address fingers independently by muscle
alone, so what this trains is the *pattern* — a controlled, low-amplitude
contraction the system maps to a single-digit grip. Beginners get the digits with
the most independent control; harder levels use all five.

### Strength Meter

Reach and hold a high effort level. Two purposes: building the endurance needed
for sustained grips, and tracking the **fatigue index** — the decline in
achievable effort across a session, which is the direct cause of control
degrading after twenty minutes of use.

Rest between repetitions is enforced. Without it this measures fatigue rather
than strength.

### Consistency

Repeat the same contraction ten times. Scores the variance *between* repetitions,
not the accuracy of any one.

This is the one that matters most for the classifier. A user whose "close" signal
varies from trial to trial gets inconsistent behaviour no matter how good the
algorithm is. A coefficient of variation below 0.10 is clinically good
repeatability; above 0.25 usually means the electrodes or the calibration need
attention rather than the user.

## Difficulty

Five levels, adapted automatically after every session:

- **Promote** after 2 consecutive sessions ≥ 80 %.
- **Demote** immediately below 40 %.

The asymmetry is deliberate: **promote slowly, demote quickly.** A user who is
struggling should be moved back to a level they can succeed at immediately,
because repeated failure is what makes people abandon rehabilitation exercises. A
user doing well can afford one more session at their current level.

## Progress

Persisted per user as plain JSON, written atomically — small, inspectable, and
trivially exportable for a clinician.

- Per-session records, per-exercise bests, adapted difficulty.
- **Trend** — improving, steady or declining, from the newest third of sessions
  against the oldest. A plateau after steady improvement usually means the
  exercise has stopped being informative; a regression usually means fatigue or
  an electrode problem, and saying so is more useful than showing a lower number.
- Streaks, totals, mastery.

## Achievements

Motivation is a real engineering concern here, not decoration. They reward the
behaviour that actually produces control:

- **turning up** — streaks and totals, because frequency beats intensity;
- **consistency** — rewarded more than peak performance;
- **breadth** — trying every exercise, since the skills are complementary.

**Nothing here gates functionality.** An achievement never unlocks a feature; the
hand is fully capable from the first minute.

## Why assistance is off

Training Mode sets `ai_enabled = False`. An exercise that measures grip accuracy
while an AI quietly corrects the user measures nothing, and a user who improves
only *with* assistance has not improved.

Force is also reduced (0.50) because exercises involve a lot of repetition.

## Running

From the touchscreen: **Train** → pick an exercise.

From the terminal:

```bash
neurogrip train --list
neurogrip train accuracy --difficulty hard
```

## Adding an exercise

Implement the `Exercise` protocol — `start`, `update`, `finished`, `results` —
and add it to `EXERCISES`. The session runner, scoring, statistics, achievements
and UI all pick it up automatically.

```python
class MyExercise(_ExerciseBase):
    key = "my_exercise"
    title = "My Exercise"
    description = "What it trains"
    trials = 10

    def update(self, emg, hand, now) -> ExerciseState:
        ...
```
