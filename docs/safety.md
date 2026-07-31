# Safety

> **This is a research prototype, not a medical device.** It has not been
> evaluated by any regulatory body. Nothing here constitutes a certification
> argument. It is an engineering safety case, written so that the design can be
> reviewed and challenged.

---

## Hazards

A tendon-driven hand attached to a person can hurt them, hurt others, or destroy
itself. The hazards this design takes seriously:

| # | Hazard | Consequence | Mitigation |
|---|---|---|---|
| H1 | Unintended motion | The hand grabs something the user did not want | Intent gate; no path from perception to motion |
| H2 | Excessive grip force | Crushed object; injury; socket pressure damage | Absolute force ceiling; per-object limits; contact-limited targets |
| H3 | Motion that will not stop | Cannot let go; trapped | Cancel gesture; e-stop; host and firmware watchdogs |
| H4 | Acting on bad sensor data | Erratic behaviour | Signal-quality gating; staleness limits; electrode-off detection |
| H5 | Host failure mid-grasp | Hand frozen or squeezing | **Firmware** watchdog, independent of the host |
| H6 | Thermal or electrical damage | Burns; fire; failure | Current, temperature and voltage limits at both ends |
| H7 | Silent degradation | User trusts a system that has quietly stopped working | Every degradation is logged, shown on screen, and given a remedy |
| H8 | Unexplained behaviour | Loss of trust; device abandoned | Every decision carries reasons; black-box recorder |

---

## H1: unintended motion

The primary hazard, and the reason for the architecture.

### The structural argument

`DecisionFusion.evaluate` is the only producer of motion decisions, and
`HandController` is the only consumer that can reach the actuators. Between them,
gate 3 is unconditional:

```python
if intent is None:
    return Decision(action=IDLE, ...)
if not intent.is_fresh(now, policy.max_intent_age_s):
    return Decision(action=IDLE, ...)
if not intent.requests_motion:
    return Decision(action=IDLE, ...)
```

There is no branch below that point reachable without a fresh, motion-requesting
intent from the user. The grasp planner is not even *called* until gate 7.

### Defence in depth

| Layer | Mechanism |
|---|---|
| Signal | A resting user produces exactly `0.0` activation — the onset threshold is a hard floor at `rest_mean + 3σ`, not a small number |
| Intent | Dwell time; a gesture must persist before it counts |
| Intent | Quality gate; an unusable signal expresses no intent regardless of amplitude |
| Fusion | Gate 3 (presence), gate 4 (confidence) |
| Mode | Manual and Training do not even run the camera |
| Control | Motion timeout; a command that never completes is cancelled |
| Firmware | No `SET_TARGETS` → drive is safed |

Tested by `test_a_visible_object_alone_never_moves_the_hand`,
`test_no_intent_means_no_motion_however_confident_the_vision`,
`test_the_planner_is_not_even_consulted_without_intent`, and the
`no-intent-no-motion` scenario.

---

## H2: excessive grip force

Force passes through five independent ceilings, and the **lowest always wins**:

```
applied = min( servo.max_force,        # hardware capability     0.85
               affordance.max_force,   # object class            e.g. 0.25 for fruit
               mode.force_ceiling,     # operating mode          0.50–0.80
               safety.force_ceiling,   # thermal/battery derate  0.0–1.0
               user_effort )           # the user's own contraction
```

The user's effort scales *within* the allowed band; it can never raise the
ceiling. Every affordance-table value is deliberately below the hand's
capability — a prosthesis that can crush an egg will eventually crush an egg.

On top of that, `AdaptiveGripController` stops each finger *where it made
contact* rather than driving it to the commanded closure, so force comes from a
regulated hold rather than from positional overdrive. Soft contact is detected
separately (a smaller current rise held for longer), because compliant objects —
the ones that least tolerate squeezing — barely resist at all.

Finally `GripForceRule` treats an estimated force above the limit as
**critical**: e-stop, latched.

---

## H3: motion that will not stop

Four independent ways to stop, in order of latency:

| Mechanism | Latency | Trigger |
|---|---|---|
| EMG cancel (co-contraction) | ~40 ms + one control cycle | The user, hands-free |
| On-screen STOP | one UI frame + one control cycle | The user, deliberately |
| Safety-triggered e-stop | synchronous with the trigger | Any critical fault |
| Firmware watchdog | 250–300 ms | Host silence, whatever the cause |

`EmergencyStop` notifies registered listeners synchronously, and the controller is
one of them, so triggering a stop cuts drive on whatever thread triggered it. The
check in `decision_tick` remains as a 100 Hz backstop for stops the monitor raises
on its own — but an emergency stop must not wait for a scheduler group that may
itself be the thing that has stalled.

That listener registration is exactly the kind of wiring that goes missing
silently: the mechanism existed and nothing used it until the production-readiness
audit, so `neurogrip test estop` and
`test_emergency_stop_reaches_the_actuators_without_the_decision_loop` now both
fail if it is ever removed again.

The cancel gesture bypasses the normal dwell machinery entirely: it is checked
before intent freshness and before the confidence gate, because an abort must
work even when it is the only thing the EMG system has managed to produce.

Cancel **holds position** rather than opening. If the user is holding a cup and
aborts, dropping it would be worse than stopping where they are.

The e-stop is a **latch**. It stays engaged until a human explicitly clears it,
and `acknowledge()` is refused while the underlying condition is still reported —
a user can acknowledge a fault, but cannot acknowledge away a hand that is still
over-temperature.

---

## H4: acting on bad sensor data

Every sensor input carries a quality assessment, and quality gates action:

- **EMG quality** from four independent indicators (saturation, noise floor,
  mains contamination measured *before* the notch filter removes it, dropouts).
  Below `FAIR`, AI assistance is disabled; at `UNUSABLE`, intent is suppressed
  entirely.
- **Electrode-off detection** covers both failure modes: a dead lead goes silent,
  while a lifted lead becomes a floating high-impedance input dominated by mains
  pickup.
- **Staleness.** Intent older than 300 ms cannot authorise motion; vision older
  than 500 ms cannot inform a plan. Evidence confidence also decays continuously
  with age.
- **Track stability.** A single-frame detection is not trusted; the tracker votes
  labels over a window, and a low agreement score reduces vision confidence
  regardless of the per-frame score.

---

## H5: host failure

**The most important property in this document is that safety does not depend on
this process being alive.**

Every `SET_TARGETS` refreshes a timeout inside the ESP32 firmware. If the host
crashes, is unplugged, hits an OOM kill or stalls in the kernel, the firmware
ramps to a safe hold and disables drive on its own, with no host involvement.

The host-side watchdogs are a second, independent layer:

| Watchdog | Timeout | Severity | Meaning |
|---|---|---|---|
| `control` | 100 ms | CRITICAL | The control loop stalled |
| `decision` | 200 ms | FALLBACK | The decision loop stalled |
| `emg` | 300 ms | FALLBACK | No EMG data |
| `vision` | 2 s | DEGRADED | Perception is stale |
| `ui` | 3 s | MINOR | The interface hung |

Verified by `test_firmware_watchdog_safes_the_hand_when_the_host_goes_quiet`,
which runs the production driver against the firmware emulator over the real
protocol.

### Recovering from a host failure

Two mechanisms, both deliberately conservative.

**A dropped link reconnects, but does not re-arm.** USB serial fails for reasons
that have nothing to do with software: a connector works loose, a hub browns out,
the CDC driver renumbers the device, the controller reboots after a power glitch.
`ReconnectingTransport` reopens the link with exponential backoff (0.5 s to 8 s)
and then calls `Esp32ServoBus.resync`, which replays the watchdog period, the
limits and the tendon calibration — a rebooted controller is running firmware
defaults, and reconnecting without restoring them would be a downgrade rather
than a recovery.

Reconnection never re-energises the actuators. Coming back from a disconnect is
not evidence that moving is safe, the user has not asked for motion since it
happened, and the firmware watchdog has already safed the drive by the time
anyone noticed.

**An unclean shutdown makes the next run more conservative.** A run marker in
`var/run-state.json` is claimed at startup and cleared on a clean stop, so the
next start can tell that the previous one never recorded an ending. When it
detects that, the system:

- comes up in **Manual mode with the AI disabled**, regardless of the configured
  default;
- tells the user what happened, including whether the hand was in motion at the
  last checkpoint;
- waits for the user to re-enable assistance.

That direction matters. A fault that crashes the process must not put the hand
straight back into the state that crashed it, which is exactly what a
`Restart=on-failure` service unit would otherwise do several times a second.
Nothing attempts to *resume* what was in progress: the hand is a limb, the user's
arm has moved, and whatever was in front of the camera is gone.

---

## The response ladder

Severity determines the response, and the mapping is the same everywhere:

| Severity | Response | Rationale |
|---|---|---|
| `MINOR` | Log | Informational |
| `DEGRADED` | Disable the affected feature | The hand still works |
| `FALLBACK` | Disable AI; keep direct control | **The user keeps their hand** |
| `CRITICAL` | E-stop, latched | Only when continuing is dangerous |

Only `CRITICAL` stops the hand. That asymmetry is deliberate: a person mid-task
with a failed camera needs their hand more than they need assistance. Turning a
degraded sensor into a dead limb would be its own hazard.

---

## Startup and shutdown

**Startup**, in order — nothing is energised until the checks have run:

1. Construct everything (no I/O).
2. Start the non-actuating services: EMG, vision, safety, diagnostics.
3. **Run the power-on self-test with the actuators still de-energised.**
   Motion tests are skipped; they require an explicit user request.
4. On failure: enter `DEGRADED`, force Manual Mode, show every failure with its
   remedy. Do not pretend to be healthy.
5. Home the hand.
6. Enable drive and enter the default mode.
7. Start the scheduler.

**Shutdown** is the exact reverse and always ends de-energised: stop the
scheduler, relax the grip so nothing is left clamped when power is removed,
disable drive, stop services in reverse start order, flush the black box.

`ServiceRegistry.start_all` stops everything already started if any service fails
to start — a partially initialised device must never be left holding actuator
power.

---

## H8: explainability

Every decision carries `reasons`. The dashboard renders them; the black-box
recorder stores them alongside the decision and the evidence that produced it.

The black box is a rolling in-memory buffer flushed to disk on any critical
fault, e-stop or watchdog expiry. After an incident — *"it squeezed too hard"*,
*"it moved when I didn't mean it to"* — the question is answerable. Without it,
every such report is unfalsifiable, which is not an acceptable position for a
device someone wears.

---

## Limits of this design

Stated plainly, because a safety case that only lists strengths is not a safety
case:

- **No redundant sensing.** One EMG channel pair, one camera. A single-point
  sensor failure degrades to manual; it does not have a backup.
- **No independent hardware safety channel.** The e-stop is software, running on
  the same MCU as the drive. A certifiable device needs a hardware interlock that
  removes actuator power without any software in the path.
- **Grip force is estimated, not measured.** Motor current is a proxy. The
  current-to-force constant is derived from the datasheet, not from measurement
  (`TODO(hardware)` in `control/force.py`). Fingertip force sensing would be
  substantially better.
- **Monocular depth.** Distance comes from class size priors and is approximate.
  It modulates approach speed and aperture only — never whether a grasp happens.
- **No formal verification.** The invariants are enforced structurally and tested
  extensively, but not proven.
- **Thermal model is not validated** against the real actuators.
- **The intent classifier is not personalised** by default. The threshold
  classifier works for everyone reasonably and for nobody optimally.
- **Firmware calibration is not persistent.** Endpoints and tendon slack live in
  RAM on the controller and are re-sent by the host at startup and after every
  reconnect. That is correct but not robust: a controller that reboots while the
  host is unresponsive runs on defaults until the host notices.
  (`TODO(persistence)` in `firmware/.../main.cpp`.)
- **The emergency stop is tested, not continuously monitored.**
  `neurogrip test estop` verifies the whole path — stop, de-energise, latch — but
  only when someone runs it. There is no periodic self-check that the path is
  still intact, and no way to detect a stop that has silently stopped working
  between tests.

Anyone taking this beyond a prototype should start with the second and third
items.
