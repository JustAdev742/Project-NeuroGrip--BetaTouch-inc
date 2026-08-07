# Configuration

TOML, parsed with the standard library's `tomllib`.

TOML rather than YAML deliberately: `tomllib` ships with Python, so the target
device needs no dependency to read its own configuration; TOML has an
unambiguous type system; and it cannot express the implicit conversions that make
YAML a source of field faults.

## Layers

Merged in increasing order of precedence:

```
config/default.toml          the shipped baseline
config/grasps.toml           grip presets (tuning data)
config/affordances.toml      object handling policy (tuning data)
config/<profile>.toml        --profile simulation | hardware
var/user.toml                device-level tuning, hand-edited, survives updates
var/profiles/<name>.json     the active user's saved preferences
NEUROGRIP__SECTION__KEY      environment
--set section.key=value      command line
```

Two things are called "profile" and both keep their names because both are
established. A *deployment* profile (`config/simulation.toml`) selects which
hardware the build talks to. A *user* profile (`var/profiles/alice.json`) holds
one person's preferences. They sit at different layers and never interact.

Nested tables merge key by key; every other type — including lists — is replaced
wholesale, which is what makes "override the enabled channel list" behave as
expected.

```bash
neurogrip config                        # print the merged result
neurogrip config servo                  # just one section
neurogrip config --check                # validate; exits non-zero on error
neurogrip run --set control.max_velocity=1.2 --set logging.level=DEBUG
NEUROGRIP__SERVO__PORT=/dev/ttyUSB1 neurogrip run
```

Environment and CLI values are parsed with TOML scalar rules, so `true`, `12`,
`1.5` and `"text"` all work; a bare string such as `/dev/ttyUSB0` needs no
quoting.

## Validation

Layered configuration has one dangerous property: every lookup has a default, so
a misspelled key is indistinguishable from an absent one. `max_forse = 0.4` does
not fail — it is ignored, and the hand runs at the 0.85 default. On a device that
squeezes things, silently ignoring a force limit is the worst failure mode a typo
could have.

`neurogrip config --check` runs three classes of check, and so does startup:

**Type and range.** `servo.max_force = 2.5` is not a valid force. Bounds are
"physically implausible" limits, not preferences — outside them is a mistake
rather than an unusual choice.

**Unknown keys.** Reported as *warnings*, not errors, because deployment profiles
legitimately carry keys a given build does not know about, and the known-path
list is maintained by hand and will lag the code.

**Cross-field consistency.** The valuable ones. Each corresponds to a failure
that is obvious in hindsight and baffling at the time:

| Check | Why it matters |
|---|---|
| Watchdog ≥ 2 × the period it guards | A shorter one trips on every healthy cycle |
| `servo.watchdog_ms` ≥ 3 control periods | The firmware would safe the drive during normal operation |
| `emg.offset_threshold` < `onset_threshold` | Inverted hysteresis makes activation chatter at the threshold |
| `emg.band_high_hz` < Nyquist | A band edge above half the sample rate is not a filter |
| `control.max_*` ≤ `servo.max_*` | The actuator limit wins, so a higher host limit is merely misleading (warning) |
| `fusion.max_intent_age_s` ≤ 2 × EMG watchdog | A stale intent could outlive the detection of a dead electrode (warning) |
| `runtime.vision_hz` ≤ camera fps | The pipeline would re-process frames it has already seen (warning) |

`[safety.estop_check]` controls how often the emergency stop verifies itself —
`rehearsal_interval_s` bounds how long a broken signalling path could go
unnoticed, and `proof_interval_s` how often the hardware path is proven by
actually cutting drive. See [safety.md](safety.md).

Errors refuse the boot, before any device is opened — the alternative is a
half-built system holding an open serial port while it reports a problem that was
knowable beforehand. Warnings are logged and shown on the diagnostics screen.

All issues are collected in one pass, so a single run tells you everything that
needs fixing.

## User profiles

Preferences follow the person, not the device. Two people sharing a hand need two
EMG calibrations and two sets of accessibility settings, and switching between
them must not mean recalibrating.

```bash
neurogrip profile list
neurogrip profile create alice
neurogrip profile use alice
neurogrip profile show alice
```

Changes made in the touchscreen UI — theme, text size, reduce motion, high
contrast, preferred mode — are written through to the active profile immediately,
not at shutdown. The device may lose power at any moment, and a setting the user
chose and then lost is worse than one they never had.

A profile may only carry preferences. The allowed prefixes are `ui.theme`,
`ui.accessibility.*`, `modes.default`, `emg.calibration_path` and `training.*`;
anything else is refused on save. Without that restriction a UI bug could persist
an arbitrary override — including a safety limit — into a file loaded on every
boot.

Servo calibration is deliberately *not* here: tendon slack belongs to the hand,
not to whoever is wearing it, so it lives in `var/servo-calibration.json` with a
`hand_id` that stops it being applied to a different unit.

A corrupt profile file costs the user their preferences and nothing else — the
store skips it, falls back, and starts.

## Safety constraint

**Configuration may make the device more conservative, never less.**

- `fusion.*` overrides are clamped against the built-in floors: force ceilings
  can only be lowered, confidence thresholds only raised.
- `ai_enabled` cannot be turned on for a mode whose built-in policy disables it.
  Manual Mode means manual.
- `SET_LIMITS` on the wire can only tighten the firmware's compiled-in ceilings.
- A missing required key is a **startup failure**, not a silent default. Silently
  defaulting a servo limit is a safety issue.

## Profiles

| Profile | Purpose |
|---|---|
| `default.toml` | Baseline. Documents every tunable. |
| `simulation.toml` | Everything simulated. CI, tests, development. |
| `hardware.toml` | The reference build. |

```bash
neurogrip run --profile simulation
neurogrip run --profile hardware
neurogrip run --config /etc/neurogrip/site.toml
```

## The settings that matter most

### `[servo]` — hard actuator limits

```toml
max_velocity = 2.0        # closure units/s: 2.0 = open→closed in 0.5 s
max_acceleration = 8.0
max_current_ma = 900
max_temperature_c = 65.0
max_force = 0.85          # absolute ceiling; no mode or plan may exceed it
watchdog_ms = 300         # firmware timeout on host silence
```

These are *safety* limits, not comfort settings. Modes may request slower motion,
never faster.

### `[emg]` — get `mains_hz` right

```toml
mains_hz = 50.0           # 60.0 in the Americas and parts of Asia
```

Wrong value = a large uncancelled artefact. It is the single most commonly
mis-set field.

```toml
onset_threshold = 0.22    # activation needed to register intent
offset_threshold = 0.12   # hysteresis band
dwell_s = 0.12            # persistence before intent counts
```

Lowering `onset_threshold` makes the hand easier to trigger *and* more likely to
trigger accidentally. For a prosthesis, false activations are much worse than
missed ones, which is why the defaults sit on the cautious side — and why the
training system exists to build the control that lets a user lower them safely.

Electrode roles are configuration, so a different placement is a config change:

```toml
[[emg.channels]]
index = 0
name = "Flexor"
role = "flexor"           # the classifier keys off the role, never the index
```

One `flexor` and one `extensor` channel are required; startup fails otherwise.

### `[vision]`

```toml
backend = "hggd_mcu"      # hggd_mcu | onnx_detector | mock | null
max_result_age_s = 0.5    # older results cannot inform a plan

[vision.hggd_mcu]
model_path = "models/hggd_mcu/hggd_mcu_int8.onnx"
score_threshold = 0.35
threads = 2               # leaves headroom for the 200 Hz control loop
```

### `[runtime]` — loop rates

```toml
control_hz = 200.0
emg_hz = 200.0
decision_hz = 100.0
vision_hz = 20.0
ui_hz = 15.0
diagnostics_hz = 2.0
```

Everything runs on one thread. If the diagnostics screen shows overruns, lower
`vision_hz` before anything else — it is the expensive group and the least
time-critical.

### `[ui]`

```toml
renderer = "tk"           # tk | text | null
theme = "dark"            # dark | light | high_contrast | auto

[ui.accessibility]
font_scale = 1.0          # 0.8–2.0; touch targets scale with it, never below 44 px
reduce_motion = false
high_contrast = false
show_numeric_values = true
```

### `[telemetry]`

```toml
blackbox = true           # leave this on
enabled = false           # continuous telemetry: development and clinical review
```

The black box keeps recent events in memory and writes them out when something
goes wrong. It is what makes "why did it do that?" answerable after an incident.

## Tuning data

`config/grasps.toml` and `config/affordances.toml` are *data a clinician can
edit*, not system configuration — which is why they are separate files.

```toml
[grasps.cylindrical]
pose = { thumb = 0.72, index = 0.82, middle = 0.84, ring = 0.82, pinky = 0.78 }
preshape = { thumb = 0.15, index = 0.10, middle = 0.10, ring = 0.12, pinky = 0.15 }
force = 0.62
speed = 0.95

[affordances.fruit]
grasps = ["spherical", "precision_pinch"]
max_force = 0.25          # the lowest ceiling in the table
speed_scale = 0.65
fragile = true
aliases = ["apple", "orange", "tomato", "egg"]
```

Unknown grasp names in either file are logged and skipped rather than fatal: a
config written for a newer build must not stop this one from starting.
