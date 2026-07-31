# Operating modes

A mode is a **bundle of parameters**, not a separate control implementation.
All four run the identical pipeline — EMG → intent → fusion → controller — and
differ only in what they configure.

That is the point: **a new mode cannot introduce a new way for the hand to move**,
and therefore cannot introduce a new way to move it unsafely. It also means the
safety and fusion code contains no `if mode == …` branches to get out of date.

## What actually differs

| | Manual | AI Assist | Sports | Training |
|---|---|---|---|---|
| AI enabled | **no** | yes | yes | **no** |
| Vision rate | **0 Hz** | 20 Hz | 30 Hz | **0 Hz** |
| Control rate | 200 Hz | 200 Hz | 250 Hz | 200 Hz |
| Force ceiling | 0.70 | 0.75 | 0.80 | 0.50 |
| Speed ceiling | 1.0× | 1.0× | **1.6×** | 1.0× |
| Intent dwell | 120 ms | 120 ms | **70 ms** | 100 ms |
| Cancel dwell | 40 ms | 40 ms | **30 ms** | 40 ms |
| Max intent age | 300 ms | 300 ms | **200 ms** | 300 ms |
| Plan hold | — | 600 ms | **300 ms** | — |
| Jerk limiting | yes | yes | **no** | yes |
| EMG weight | 1.00 | 0.60 | 0.75 | 1.00 |
| Vision weight | 0.00 | 0.40 | 0.25 | 0.00 |
| "AI DISABLED" banner | **yes** | no | no | **yes** |

---

## Manual

> Direct EMG control. Every finger follows the muscle signal. Nothing is
> interpreted, predicted or optimised.

For learning, diagnostics, calibration, and any moment the user wants the hand to
do exactly and only what they tell it.

The AI path is closed at **two independent points**: the policy sets
`ai_enabled = False`, and `vision_rate_hz = 0` means the camera is not merely
ignored — it is not run at all, so no stale perception result can influence
anything.

The interface says so, prominently and persistently. The banner is a
specification requirement and is asserted by
`test_manual_mode_shows_the_ai_disabled_banner`.

## AI Assist

> Shared control. You decide when. The AI decides how.

The primary mode.

```
camera perceives, continuously
        ▼
EMG expresses intent to grasp      ◀── nothing happens before this
        ▼
grasp planner is consulted (only now)
        ▼
plan → affordance force cap → user-effort scaling
        ▼
motion, cancellable at any moment
```

Perception runs continuously so a plan is *ready*; a plan is never *executed*
without intent. Those are deliberately separate steps.

The plan is held for 600 ms so the hand does not change its mind mid-reach when
the classifier flickers.

## Sports

> Optimised for speed. Still requires intent for every action.

Everything that adds latency is reduced: jerk limiting off, dwell halved, plan
refreshed twice as often, control loop 25 % faster, vision weighted lower because
it is the slow input.

What is *not* reduced is the requirement for user intent — the same fusion gates
apply, with the same structure.

The trade-off is stated plainly on screen: a shorter dwell means more accidental
activations. That is why this is not the default.

## Training

> Exercises and games. Assistance off, so you see your real control.

Hosts the five training exercises. Assistance is deliberately disabled: an
exercise that measures grip accuracy while an AI quietly corrects the user
measures nothing, and a user who improves only *with* assistance has not
improved.

Force is reduced to 0.50 because exercises involve a lot of repetition.

Hand control still flows through the same fusion path, so what the user practises
is exactly what they will use.

---

## Switching modes

Three ways:

1. **Touchscreen** — a button per mode on the dashboard.
2. **Hands-free** — a double flexor pulse within 900 ms cycles
   AI Assist → Manual → Sports. This matters: the hand you would reach for the
   touchscreen with *is* the prosthesis.
3. **Automatic fallback** — safety forces Manual when AI assistance becomes
   unavailable.

### Arbitration rules

- **Safety can veto.** A change is refused while a critical fault is active, and
  an AI mode is refused while AI assistance is unavailable.
- **Exit before enter, always.** The outgoing mode stops its motion before the
  incoming one configures the controller, so no command can straddle a change.
- **Position is held, never dropped.** Switching mode while carrying something
  must not drop it.
- **Debounced.** A minimum dwell (1 s) stops the hands-free gesture cycling modes
  faster than the user can read the screen.

### Automatic fallback, and coming back

When safety withdraws AI permission, the manager forces Manual — running a mode
whose premise no longer holds would be worse than switching to one that is honest
about what the hand can currently do.

It also **remembers what the user had chosen**. When the condition clears and
stays clear for 2 s, the mode is restored and the change is announced. The settle
time matters: a hand that flips between modes as a marginal fault chatters would
be worse than one that stays in Manual.

Restoration only happens for *automatic* fallbacks. A mode the user chose is
never overridden.

---

## Adding a mode

```python
MY_PROFILE = ModeProfile(
    mode=ModeId.MY_MODE,
    policy=FusionPolicy(mode=ModeId.MY_MODE, ...),
    motion_limits=MotionLimits(...),
    intent_settings=IntentSettings(...),
    title="My Mode",
    subtitle="What it is for",
    vision_rate_hz=20.0,
)

class MyMode(ModeBase):
    pass          # ModeBase supplies the entire cycle
```

Register it in `build_modes` and add it to `ModeId`. Fusion, control and safety
are untouched — which is the property that makes this safe to do.
