# Testing guide

The whole stack runs without hardware. That is not a convenience — it is the
reason the safety properties can be asserted at all, on every commit, instead of
being checked once during a bring-up session and assumed thereafter.

```bash
pytest                                   # everything, ~2 minutes
pytest tests/unit -q                     # fast: no full-system assembly
pytest tests/integration -q              # the assembled system
pytest -m "not slow"                     # skip the long ones
pytest -k SharedControl                  # the invariants that matter most
```

---

## What is real in a test

Only the bottom layer is simulated. Everything above the HAL is the production
code path:

| Layer | In tests |
|---|---|
| Servo bus | `SimulatedServoBus` — a real plant model: rate limits, contact, stall, thermal build-up, tendon slack |
| EMG source | `SimulatedEmgSource` — synthetic sEMG with mains hum, motion artefacts, electrode lift-off |
| Camera | `SimulatedCamera` — scenes with ground truth attached |
| Wire protocol | **Real.** The `esp32` driver speaks real NGP v1 frames to an in-process firmware emulator |
| EMG pipeline | **Real.** Actual biquads, actual Hudgins features |
| Vision pipeline | **Real.** Actual tracker, actual depth estimator |
| Fusion, planning, control, safety, modes | **Real.** Every gate, every rule |

A test that passes is therefore evidence about the shipped code, not about a
test double of it.

### Determinism

Every component takes a `Clock`. Tests inject `SimulatedClock`, which advances
only when asked. Consequences worth knowing:

- **No test sleeps.** A 60-second calibration procedure runs in milliseconds.
- **Runs are reproducible.** Same inputs, same interleaving, same result.
- **`0.0` is a real timestamp.** A simulated clock starts at zero, so `0.0`
  cannot mean "never set". Use `None`. This has caused four separate bugs; see
  [development.md](development.md).

Nothing in `src/` calls `time.monotonic()` directly. If you add code that does,
it will pass its own tests and make somebody else's flaky.

---

## The tests that matter most

Three classes assert the properties the architecture exists to guarantee. If
they fail, the system has stopped being a shared-control device, and no other
passing test compensates.

```bash
pytest -q \
  "tests/unit/test_fusion_and_safety.py::TestTheAiNeverActsAlone" \
  "tests/unit/test_fusion_and_safety.py::TestAssistanceDegradesToControlNeverToInaction" \
  "tests/integration/test_system.py::TestSharedControlInvariants"
```

CI runs these as a separate step so a failure is unmistakable in the log rather
than buried among several hundred dots.

What they assert:

- **The AI never acts alone.** Vision alone cannot produce motion. The
  `no-intent-no-motion` scenario keeps a bottle in frame at high confidence for
  six seconds and asserts zero finger travel — inaction as a decision, not as
  blindness.
- **A stale intent is not an intent.** Motion stops when EMG goes quiet, and a
  held plan is dropped rather than resumed.
- **Degradation goes towards user control.** Losing vision, or the model, or
  confidence, must produce a hand the user still directly controls — never a
  hand that stops responding.
- **The emergency stop still works.** `TestEstopIntegrity` covers the periodic
  self-check that watches the stop path between manual tests, including the
  case that motivated it: a listener registration silently disappearing.

---

## Test layout

```
tests/
  conftest.py              shared fixtures, all clock-injected
  unit/
    test_core.py           types, clock, events, state machine, config, buffers
    test_emg.py            filters, features, calibration, gestures, intent
    test_vision_and_ai.py  backends, tracking, depth, grasp planners
    test_control.py        trajectories, queue, contact, force, kinematics
    test_fusion_and_safety.py   the gates, the rules, the watchdogs
    test_hal_and_protocol.py    framing, CRC, encode/decode, device fallbacks
    test_modes_training_ui.py   mode policies, exercises, scene rendering
    test_reliability.py    calibration, reconnection, validation, profiles,
                           crash recovery, bring-up tools, replay, AnyGrasp
  integration/
    test_system.py         the assembled system, including the audit fixes
```

`test_reliability.py` is organised by *defect*, not by module. Each class
corresponds to something the production-readiness audit found — usually a
mechanism that existed and was connected to nothing. Unit-testing the mechanism
would have passed the whole time; only assembling the system showed the wire was
missing. The tests are written so removing the wire fails them again.

---

## Scenarios

Scenarios are end-to-end stories with pass criteria, run under a simulated clock.

```bash
neurogrip simulate --list
neurogrip simulate all
neurogrip simulate grasp-bottle --log-level DEBUG
```

| Scenario | Asserts |
|---|---|
| `grasp-bottle` | The normal path: see, intend, plan, grasp, hold |
| `no-intent-no-motion` | Vision alone never moves the hand |
| `user-cancel` | A co-contraction stops motion at any point |
| `vision-lost` | The hand still closes under direct control with no camera |
| `fragile-object` | Force is capped for something that would break |

Scenarios also run in CI. They are the closest thing to a demonstration that a
machine can check.

---

## Regression testing perception

Procedural scenes test *logic* — they can produce any situation on demand. They
cannot test *perception quality*, because the ground truth is whatever the
generator decided.

Recorded replay closes that gap:

```bash
# Replay a fixed reference scene through the real pipeline
neurogrip run --profile simulation --duration 10 \
  --set vision.backend=replay \
  --set vision.replay.path=data/vision/reference-bottle.jsonl
```

The bundled `data/vision/reference-bottle.jsonl` is 120 frames of a 500 ml bottle
being approached, including the false negatives and misclassifications a real
detector produces. Change a backend, replay it, and any difference is
attributable to your change rather than to a different random scene.

To capture your own — from hardware or from a known-good build:

```python
from neurogrip.vision.backends.replay import VisionRecorder

pipeline.recorder = VisionRecorder("var/recordings/session.jsonl", backend="hggd_mcu")
```

The format is JSON Lines: one result per line, in capture order. Line-oriented so
a recording truncated by a power loss loses only its last line, and text so it can
be read and trimmed by hand. Recordings are evidence, and evidence that needs a
special tool to read is evidence nobody reads.

---

## Hardware-in-the-loop

Tests marked `hardware` are skipped by default:

```bash
pytest -m hardware        # requires a connected device
```

For manual bring-up verification, use the tools rather than the test suite — they
report measurements, not just pass/fail:

```bash
neurogrip test link       # latency distribution, loss, framing errors
neurogrip test range      # per-finger travel and cross-coupling   ⚠ moves
neurogrip test estop      # the stop, the de-energise, the latch   ⚠ moves
neurogrip test all
```

See [hardware.md](hardware.md).

---

## Writing a test

Two rules, both learned the hard way.

**Inject the clock.** Take `clock` as a fixture and advance it explicitly. Never
sleep, never call `time.monotonic()`.

```python
def test_intent_goes_stale(clock, bus):
    engine = IntentEngine(classifier, clock, IntentSettings())
    ...
    clock.advance(0.5)
    assert not intent.is_fresh(clock.monotonic(), max_age=0.35)
```

**Assert behaviour, not implementation.** `assert not report.ok` says what
matters; `assert len(report.results) == 4` breaks the moment someone adds a
check.

For a test that needs the full system, use the `application` fixture in
`tests/integration/test_system.py` and drive it with `_run(app, seconds)`.

---

## Continuous integration

`.github/workflows/ci.yml`, on Python 3.11 and 3.12:

1. install `[dev]` only — **never** the optional extras
2. `ruff check src tests`
3. `mypy src` (non-blocking)
4. unit tests
5. integration tests
6. the shared-control invariants, as their own step
7. the five scenarios
8. a separate job building the ESP32 firmware

Step 1 is deliberate. That the stack runs on the standard library alone is a
property worth protecting; if someone adds a bare `import numpy` to a runtime
module, CI catches it rather than the first person to flash a board.
