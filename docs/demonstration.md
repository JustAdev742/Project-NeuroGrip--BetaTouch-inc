# Demonstration guide

For showing this to judges, a review panel, or anyone who has ten minutes and no
prior context. Written to be run from a laptop with no hardware, and to survive
the things that go wrong at a demonstration table.

---

## The one-sentence version

**The AI never moves the hand. The user does. The AI only decides *how*.**

Everything below is in service of making that claim checkable rather than
asserted.

---

## Before you present

```bash
neurogrip config --check          # or: PYTHONPATH=src python3 -m neurogrip config --check
neurogrip simulate all            # must print 5/5
pytest -q                         # 392 passed
```

If `simulate all` passes, the whole stack works. If you have hardware, also run
the bring-up sequence in [installation.md](installation.md) — especially
`neurogrip test estop`.

Have a terminal with a large font ready. Everything here is text output.

---

## The ten-minute demonstration

### 1. It runs (1 min)

```bash
neurogrip simulate all
```

Five scenarios, each with its pass criteria printed. Let them read it.

Then say what it is: about 27,000 lines of Python plus ESP32 firmware, running
EMG signal processing, computer vision, grasp planning, decision fusion, motion
control and safety monitoring, at 200 Hz, with no third-party dependencies in the
runtime core.

### 2. The claim, made checkable (3 min) — **the important part**

```bash
neurogrip simulate no-intent-no-motion
```

```
no-intent-no-motion: PASS (6.0 s)
  ✓ hand did not move without user intent (max travel 0.000)
  ✓ the camera did see the object (so inaction was a decision, not blindness)
```

Explain what that second line is doing. The camera is tracking a bottle at high
confidence for six seconds. The grasp planner has a plan ready. The hand does not
move — **0.000** — because no EMG intent arrived.

That is the difference between a prosthetic and a robot. The second assertion
exists because "it didn't move" is worthless without "and it could see something
to move towards".

Then show why it holds:

```bash
grep -n "no EMG intent available" -A 6 src/neurogrip/fusion/fusion.py
```

Gate 3 of seven, in `fusion.py`. If there is no fresh intent, the function
returns `IDLE` before any planning happens. It is not a policy anyone has to
remember — there is no code path from vision to motion.

If they ask about the architecture, the honest one-liner is: **one writer to the
actuators**. `control/controller.py` is the only module that writes to the servo
bus. Every safety property — force ceilings, velocity limits, emergency stop,
cancellation — needs exactly one implementation because there is no other path.

### 3. The user is always in charge (2 min)

```bash
neurogrip simulate user-cancel
neurogrip simulate vision-lost
```

- **user-cancel** — a co-contraction (tensing both muscle groups) stops motion
  mid-grasp. The user overrides the AI at any point, without a menu.
- **vision-lost** — the camera fails. The hand still closes to 0.92 under direct
  EMG control. Degradation goes towards *user control*, never towards a hand that
  stops responding.

That second one is worth dwelling on. The easy failure mode for an assistive
device is to become useless when its clever part breaks.

### 4. It knows when to be gentle (1 min)

```bash
neurogrip simulate fragile-object
```

```
  ✓ force limited for a fragile object (0.25 ≤ 0.45)
```

Vision identifies the object; the affordance table caps the force; fusion enforces
the lower of that and the mode's ceiling. An egg gets 0.25, a hammer gets more.

### 5. It is built to be worked on (2 min)

Pick whichever lands better with the audience.

**Swappable models** — the brief was HGGD-MCU, but nothing is hard-coded to it:

```bash
neurogrip info | grep -i planner
python3 -c "from neurogrip.vision.backend import available_backends; print(available_backends())"
```

Six vision backends, three grasp planners, all behind the same interface. The
AnyGrasp adapter is interesting because it *declines* grasps the hand cannot
reach — a 6-DoF model assumes a wrist, and this hand has five servos and no
wrist, so the adapter rejects approaches the user's arm is not making.

**Bring-up tooling** — for a hardware audience:

```bash
neurogrip test all --profile simulation
```

Link quality, per-finger travel, and an emergency-stop test that commands motion,
triggers the stop, and verifies the hand halted, the drive de-energised, and the
latch held.

**Configuration validation** — for a software audience:

```bash
neurogrip config --check --set servo.max_force=2.5 --set emg.offset_threshold=0.9
```

```
[error] servo.max_force: 2.5 is above the maximum of 1.0 — grip force is normalised
[error] emg.offset_threshold: 0.9 is not below emg.onset_threshold (0.22)
```

The second is the interesting one: both values are individually legal, and the
combination is not. Inverted hysteresis makes activation chatter at the threshold
— a symptom nobody would trace back to configuration.

### 6. Close (1 min)

Safety is structural, not procedural:

- fusion refuses to plan without fresh EMG intent;
- one module writes to the actuators;
- the **firmware** runs its own watchdog, so if this whole program crashes the
  hand safes itself without the host's help;
- an unclean shutdown is detected on the next start and the system comes back in
  Manual with the AI off, so a crash loop cannot re-enter the state that caused it.

---

## Questions you will get

**"Is this real or a simulation?"**
The simulation is the hardware layer only. Above it — filters, features, fusion,
control, safety — is the code that would run on the device. The tests even speak
the real serial protocol to an in-process firmware emulator. Point at
[testing.md](testing.md).

**"What happens if the AI is wrong?"**
Three answers, in order of increasing importance. It needs confidence above a
threshold. The user can cancel mid-motion with a co-contraction. And it cannot
initiate anything — being wrong means offering a bad grip for a motion the user
already asked for, not grabbing something.

**"Why five servos and not more?"**
Buildability. The architecture does not care: `HandKinematics` and the grip
library are where a different hand is described, and the wire protocol carries
per-finger records.

**"Did you train the model?"**
No, and be straightforward about it. HGGD-MCU is an existing architecture (Chen et
al., RA-L 2023). What is built here is the integration: the runtime-independent
decode, the mapping from parallel-jaw predictions to a five-finger hand, and the
classical fallback that runs when no weights are present.

**"How do you know the emergency stop works?"**
Two answers. On demand, `neurogrip test estop` commands motion, triggers the stop
and checks three things — motion ceased, drive de-energised, and the latch held so
the next command is refused. The third is the one that matters and the one nothing
else reveals.

But a test you have to remember to run is not much of a guarantee, so the stop
also checks itself while the system runs: it rehearses the signalling path every
30 seconds, and every 6 hours — when the hand is open and idle — it proves the
hardware path by actually cutting drive for 5 ms and watching it happen. Failure
disables AI assistance and says so; it does not stop the hand, because in Manual
every motion is directly driven by muscle.

That check exists for a specific reason worth admitting: the registration
connecting the stop to the motion controller was missing from the first version
of this system, and nothing revealed it. Everything worked. Every test passed.

**"What is not finished?"**
Answer this one straight; it lands better than deflecting. No trained weights ship
— vision runs a classical fallback unless you supply an ONNX or TFLite model. The
gesture classifier is threshold-based, with the interface for a learned one in
place. There is no wrist, so 6-DoF planners are partly constrained by what the
user's arm happens to be doing. The firmware does not yet persist calibration to
NVS. All of these are marked `TODO` in the source with the reason.

---

## If something goes wrong on the day

**A scenario fails.** Run it alone with `--log-level DEBUG`. Scenarios are
deterministic under the simulated clock, so a failure is real and reproducible,
not flakiness.

**No hardware, and you promised hardware.** Everything above runs without it. Say
so plainly — the simulated plant models rate limits, contact, stall, thermal
build-up and tendon slack, which is why the software can be tested at all.

**Someone challenges the safety claim.** Best possible outcome. Open
`fusion/fusion.py`, read gate 3 aloud, then run `no-intent-no-motion`. The claim
is checkable in about forty seconds.

**Something crashes.** Restart it and point out that the restart detected the
unclean shutdown and came back in Manual mode with the AI disabled — which is the
designed behaviour, on display.
