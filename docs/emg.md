# EMG

Turning muscle signals into a trustworthy statement of what the user wants.

## The chain

```
raw volts (electrode, post-amplifier)
    │
    ├─ DC block (0.5 Hz one-pole)      electrode half-cell potential
    ├─ notch 50/60 Hz + 3rd harmonic   mains interference
    ├─ band-pass 20–400 Hz             motion artefact ↓, noise ↑
    ├─ rectify + envelope              asymmetric: 30 ms attack, 150 ms release
    ├─ median filter (5)               isolated ADC glitches
    │
    ├─▶ normalise by calibration ──▶ activation ∈ [0, 1]
    ├─▶ features (MAV, RMS, WL, ZC, SSC)
    └─▶ quality (saturation, noise, mains, dropouts)
                    │
                    ▼
              gesture classification
                    │
                    ▼
     dwell · hysteresis · cancel fast-path
                    │
                    ▼
              IntentEstimate
```

**Nothing downstream sees microvolts.** The control layer works in normalised
activation, which is what makes the same code work for different users, different
electrodes and different front-end gains.

## Why each stage

**DC block first.** The electrode half-cell potential is a large, slowly varying
offset that would otherwise dominate everything downstream, including the
saturation check.

**Notch before band-pass.** Mains sits at 50/60 Hz, inside the EMG band, so it
cannot be removed by band-limiting. A high-Q notch (Q = 30) takes out the tone
while leaving the EMG energy either side of it: the filter self-test asserts
< 1 % at 50 Hz and > 90 % at 80 Hz.

**20–400 Hz.** Below 20 Hz is motion artefact and electrode movement; above about
400 Hz there is little surface-EMG energy left. This is the conventional band
(De Luca, 1997).

**Asymmetric envelope.** Attack 30 ms so the hand responds within the ~100 ms
that feels immediate; release 150 ms so the grip does not flutter during the
natural amplitude dips of a sustained contraction. The asymmetry is a control
decision, not cosmetics.

**Median on the envelope, not the raw signal.** A single ADC glitch would
otherwise produce a spurious activation spike and, in the worst case, an
unintended grasp.

## Calibration

Every user's signals differ — muscle mass, electrode placement, skin impedance,
amputation level — and the same user's signals differ between sessions. Nothing
downstream may use raw microvolts.

Per channel the wizard learns:

| Value | How | Why |
|---|---|---|
| `rest_mean`, `rest_std` | 5 s relaxed | The onset threshold is `rest_mean + 3σ`, which adapts automatically to a noisy electrode instead of using a fixed microvolt number |
| `mvc` | 4 s maximum contraction | The top of the activation scale |
| `full_scale` | 60 % of MVC | So the user does not have to maximally contract for full grip — sustained MVC is exhausting |

The MVC is the **95th percentile** of the hold, not the peak. Peaks are spiky and
unrepresentative of a level anyone can actually hold; using them would make full
activation unreachable in normal use.

A calibration with an MVC-to-rest ratio below 4× is **rejected**, because that is
almost always an electrode that is not making contact rather than a user who is
not trying.

```bash
neurogrip calibrate --output var/calibration.json
```

The wizard is a state machine driven one step at a time, so the same
implementation backs the touchscreen flow, the CLI and the tests.

### Activation is a hard floor

```python
if envelope <= onset_threshold:
    return 0.0
```

A resting user must produce *exactly zero*, not a small number that could
accumulate into motion.

### Auto-recalibration

Surface EMG drifts within a session: electrodes warm up, gel dries, skin sweats,
muscle fatigues. `AutoRecalibrator` watches for sustained rest and gently updates
the baseline.

It only ever adjusts the **rest baseline**, never the MVC. Lowering the effort
needed to trigger a grasp based on unlabelled data would be a safety change made
without the user's knowledge; raising the noise floor when the environment gets
noisier is the conservative direction. Adjustments beyond 2.5× are refused
outright — that is a fault, not drift, and recalibrating around it would hide it.

## Signal quality

Four independent indicators, so no single failure mode can masquerade as a
healthy signal:

| Indicator | Detects |
|---|---|
| Saturation | Samples pinned at the amplifier rails |
| Noise floor | Baseline RMS far above the calibrated rest level |
| Mains ratio | Interference, measured **before** the notch removes it |
| Dropouts | Samples lost to buffer overrun |

Mains is measured pre-notch deliberately: measuring interference downstream of
the filter that removes it would always report a clean signal.

**Electrode-off detection** covers both real failure modes:

- a dead lead goes *silent* — RMS below a quarter of the calibrated floor;
- a lifted lead becomes a *floating high-impedance input* dominated by mains
  pickup — a well-bonded differential electrode with a driven reference rejects
  mains down to around the noise floor, a floating one does not.

Quality gates action: below `FAIR`, AI assistance is disabled; at `UNUSABLE`,
intent is suppressed entirely regardless of amplitude.

## Gesture classification

Two implementations behind one interface.

**`ThresholdGestureClassifier` (default).** Deterministic rules over
flexor/extensor activation. Decision order is safety-driven:

1. **Cancel** (co-contraction) — tested first, wins unconditionally.
2. **Direction** — whichever group dominates, requiring a minimum separation so a
   sloppy contraction does not flip-flop.
3. **Rest** otherwise.

Ambiguous activation (both groups up, neither dominant, not enough for cancel)
reports `UNKNOWN` rather than guessing. The intent engine treats that as "do
nothing", which is the safe interpretation.

Note what is *not* in the classifier: any notion of "hold". Sustained effort is a
temporal property and belongs to the intent engine, which owns the clock.
Promoting to `HOLD` inside the classifier would restart the engine's dwell timer
on the very next frame, because the reported gesture would have changed.

Hysteresis (onset 0.22, offset 0.12) prevents chattering at the threshold, which
would otherwise produce a visibly twitching hand.

**`LinearGestureClassifier`.** A linear discriminant over the Hudgins feature
set, with weights loaded from a per-user JSON model. No weights ship — a linear
model trained on one person's electrodes is worthless on another's. When the
model is missing, the system falls back to the threshold classifier and says so
in the log rather than failing to start.

Why threshold by default: for a device someone depends on, a classifier whose
behaviour can be predicted, explained on screen and reproduced exactly is worth
more than a few points of offline accuracy. It also needs no training data, so a
new user is running after a 20-second calibration.

## Intent estimation

Three mechanisms turn a noisy per-frame classification into something worth
acting on:

| Mechanism | Value | Purpose |
|---|---|---|
| Dwell | 120 ms (70 ms in Sports) | Removes transients and classifier flicker |
| Cancel dwell | 40 ms | An abort must feel instant |
| Release window | 150 ms | A dip in effort must not drop an in-progress grasp |
| Confidence shaping | — | Folds in quality, dwell and margin so downstream gets one honest number |

`IntentEstimate.requests_motion` is the single predicate everything else keys off.
It is false while provisional (still accumulating dwell) and false for `REST`,
`CANCEL` and `UNKNOWN`.

A **double flexor pulse** within 900 ms is a `TOGGLE`, which cycles the operating
mode hands-free — which matters when the hand you would reach for the touchscreen
with *is* the prosthesis.

## Recording and replay

```bash
neurogrip record var/session.emg --seconds 60 --subject alex --label close
neurogrip replay var/session.emg --speed 2.0
```

Format: a JSON header line, then CSV rows of `timestamp,ch0,…,chN[,label]`.
Deliberately boring — a recording made today must still open in five years, with
`head` if necessary.

Recorded sessions are the backbone of EMG development: a user's real signals can
be captured once and then replayed through the *entire* pipeline every time the
code changes. That turns "does the classifier still recognise Alex's grasp?" into
a test.

Labels attached during recording (the training exercises do this automatically)
produce the supervised dataset needed to fit a per-user gesture model.

## Electrode placement

The reference layout is two channels:

| Channel | Site | Records |
|---|---|---|
| 0 | Flexor digitorum superficialis | Closing |
| 1 | Extensor digitorum communis | Opening |

Roles are **configuration, not code** (`[[emg.channels]]`), so a different
placement, or more channels, is a config change. The classifier keys off the
role, never the index.

Crosstalk between the groups is real (the simulator models 12 %), which is why
co-contraction detection needs a margin rather than a simple AND.
