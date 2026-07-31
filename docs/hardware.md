# Hardware

The reference build the software is tuned for. Everything here is replaceable —
that is what the HAL is for — but the defaults in `config/hardware.toml` assume
this.

## Bill of materials

| Part | Reference choice | Notes |
|---|---|---|
| Host | Raspberry Pi 4B 4 GB / CM4 | 64-bit Linux. The CM4 is preferred for the socket-mounted form factor. |
| Motor controller | ESP32-S3-DevKitC-1 | Second USB peripheral, larger SRAM |
| Actuators | 5 × metal-gear micro servo, ≥ 2.5 kg·cm | Metal gears are not optional; nylon strips under tendon load |
| Tendon | 0.6 mm braided fishing line, ≥ 20 kg | Braided, not monofilament: monofilament creeps and the calibration drifts |
| Return | 0.4 mm spring steel per finger | Passive extension; also what safes the hand when power is cut |
| EMG front end | 2-channel differential + driven reference | Driven right leg is what makes electrode-off detection work |
| ADC | ADS1256 (24-bit, 1 kSPS) | 16-bit is adequate; 24-bit gives headroom for poor contact |
| Electrodes | Ag/AgCl, or dry stainless for daily use | |
| Camera | Pi Camera Module 3, or any V4L2 device | Palm- or wrist-mounted, pointing along the reach axis |
| Display | 800×480 capacitive touchscreen | |
| Battery | 2S Li-ion, 2600 mAh, with a protection board | ~7.4 V nominal, 6.0 V empty, 8.4 V full |
| Regulation | 5 V 3 A buck for the servos, separate 5 V for the Pi | **Separate rails.** Servo inrush browns out an SBC sharing a rail. |
| Current sensing | 5 × INA181 + 0.1 Ω shunt | Contact detection depends on this |

## Power

```
2S Li-ion ─┬─ 5 V 3 A buck ──▶ servo rail (5 × ~700 mA peak)
           │                    └─ 1000 µF bulk at the servo header
           └─ 5 V 3 A buck ──▶ Pi + ESP32 + display
```

Two independent regulators. A single rail works on the bench and fails the first
time five servos start together — the Pi browns out mid-grasp, which is precisely
the failure the firmware watchdog then has to catch.

`config/hardware.toml` sets `max_current_ma = 900` per finger and the firmware
clamps at 1200 mA absolutely.

## Tendon routing

Each finger: servo horn → PTFE-lined channel → distal phalanx anchor. The return
spring runs on the dorsal side.

Two things that matter and are easy to get wrong:

**Slack.** Tension the line so the finger just begins to move at the servo's
neutral position. Remaining slack is compensated in software
(`ServoCalibration.slack`), but only up to a point — mechanical take-up is
better than a software offset because slack varies with temperature and use.

**PTFE lining.** Bare-channel routing wears through the line in weeks and the
friction is high enough to distort the current-based contact detection.

## Servo calibration

Two separate things, with different lifetimes. Getting them confused is the most
common bring-up mistake.

### Endpoints — set once, by hand, when you build the hand

Per finger: minimum pulse (fully open), maximum pulse (fully closed at the
mechanical stop), and direction. These describe how the hand was *assembled*, so
they live in configuration and the software does not try to discover them:

```toml
[servo.fingers.thumb]
min_pulse_us = 1000
max_pulse_us = 2000
inverted = false
```

Find them with the console, one finger at a time:

```bash
neurogrip console
neurogrip> arm
neurogrip> grip open
neurogrip> grip fist
```

Watch for the finger reaching its stop *before* the servo does. A servo driving
into a hard stop stalls, draws maximum current, and destroys the horn or the
tendon within minutes. Set `max_pulse_us` short of that point.

The firmware range-checks what it is sent (500–2500 µs, min < max) and rejects
anything outside it, so a typo cannot drive the horn past its mechanical limit.

The reference hand limits the thumb to 0.92 closure because at full travel it
collides with the index proximal phalanx — see `HandKinematics.COLLISION_RULES`,
which enforces this in software as well.

### Tendon slack — measured, and re-measured periodically

Between the horn and the fingertip is a length of fishing line whose effective
length changes: it is cut by hand at assembly, it stretches under load, and it
creeps over weeks of use. Some fraction of the servo's travel is consumed taking
up that slack before the finger moves at all.

If the software assumes zero slack, the first 20 % of every commanded motion does
nothing and the finger arrives late. So it is measured, not assumed:

```bash
neurogrip calibrate servo               # all five fingers, ~65 s
neurogrip calibrate servo --finger thumb --finger index
```

The wizard drives one finger at a time under low force (0.18) and slow creep,
watching the motor current. A slack tendon carries no load, so the current step
when it goes taut is what marks the boundary. It then continues to the end of
travel and reports:

```
  ✓ thumb: slack 0.21, travel to 0.96, 11→53 mA — ok
  ✗ ring: slack 0.58, travel to 0.61, 11→11 mA — tendon never went taut
        below 55% closure — re-string it
```

Two failures it catches that nothing else will:

- **Slack above 0.55** — the tendon is too long to calibrate around. It needs
  re-stringing; there is no software correction for it.
- **Travel ending below 0.80** — the finger binds early. The tendon is too short
  or the routing is fouled.

Results are written to `var/servo-calibration.json`, pushed to the firmware at
every startup, and re-pushed automatically after a link drop (a controller that
reboots comes back on defaults). The file records which hand it belongs to, so a
calibration cannot be silently applied to a different unit with different tendon
lengths.

Re-run it after re-stringing, after any mechanical work, and every month or so in
normal use. The UI nags after 30 days.

`TODO(persistence)`: the firmware still keeps calibration in RAM and relies on
the host re-sending it; NVS storage is not yet implemented.

## Camera calibration

The monocular depth estimator recovers distance from apparent size, which makes
every distance it reports depend on one number: the horizontal field of view.
Datasheet FOV is quoted for the sensor's full area, and a camera configured at a
cropped resolution has a materially narrower one — using 66° where the real value
is 58° biases every distance by about 12 %, consistently.

Measure it instead. Show the camera something of known width at several known
distances and read the apparent width in pixels:

```bash
neurogrip calibrate camera \
  --sample card:0.20:214 \
  --sample card:0.35:122 \
  --sample card:0.50:86
```

```
  FOV 65.2° horizontal (51.2° vertical), f = 500 px at 640×480, spread 0.4%
  per-sample distance error:
    card at 0.20 m                 +0.0 cm
    card at 0.35 m                 +0.1 cm
    card at 0.50 m                 -0.2 cm
  Set camera.fov_deg = 65.2 (currently 62.0)
```

A bank card is a good target: ISO/IEC 7810 ID-1 is 85.60 mm ± 0.12 mm. `a4`,
`can` and `disc` are also known.

If the samples disagree by more than 8 % the tool says so rather than averaging
bad data, and the per-sample residuals identify which measurement was wrong.

## Electrode placement

| Channel | Site | Records |
|---|---|---|
| 0 | Flexor digitorum superficialis | Closing |
| 1 | Extensor digitorum communis | Opening |

Roughly 3–5 cm distal to the elbow crease, over the muscle belly, electrodes 2 cm
apart along the fibre direction. The reference electrode goes on the olecranon —
electrically quiet bone.

Practical notes:

- Clean the skin with alcohol and let it dry. Skin impedance dominates signal
  quality, and this single step matters more than anything in the software.
- Crosstalk between the groups is real and unavoidable (the simulator models
  12 %). It is why co-contraction detection needs a margin rather than a simple
  AND of both channels.
- Run `neurogrip diagnose` after placement. If the EMG noise floor is above
  ~150 µV, the electrodes are the problem, not the code.

## Camera mounting

Palm or wrist, pointing along the reach axis, so the image centre is where the
user is aiming. Target selection depends on this: `VisionResult.primary` weights
centrality at 35 %.

Field of view goes in `[camera] fov_deg`; the monocular depth estimator derives
its focal length from it and will be wrong if it is wrong. Measure it rather than
copying the datasheet — see "Camera calibration" above.

## Assembly order

1. Print/machine the frame; ream the tendon channels and line them with PTFE.
2. Fit the servos; route and tension the tendons with the servos at neutral.
3. Wire the controller (see `firmware/.../include/pins.h`).
4. Flash the firmware: `cd firmware/esp32_motor_controller && pio run -t upload`.
5. **Set the servo endpoints before enabling drive with tendons attached.**
6. Fit the EMG front end and electrodes.
7. Bring-up sequence, in this order — each step can fail in a way that makes the
   next meaningless:

   ```bash
   neurogrip config --check     # the configuration is coherent
   neurogrip test link          # the link is good, not merely present
   neurogrip diagnose           # every device reports healthy
   neurogrip test range         # every finger reaches its travel      ⚠ moves
   neurogrip test estop         # the stop actually stops it           ⚠ moves
   neurogrip calibrate servo    # measure tendon slack                 ⚠ moves
   neurogrip calibrate camera --sample card:0.30:143 ...
   neurogrip calibrate emg      # the user's muscles
   ```

8. `neurogrip run --profile hardware`.

Do not skip `test estop`. An emergency stop that has never been tested is an
assumption, not a safety system — and the check that matters is the third one,
that the latch *holds* and refuses the next command. Nothing in normal operation
reveals a stop that engages but can be cleared by the next thing that comes
along.

## Bring-up checklist

- [ ] Servo rail holds ≥ 4.8 V with all five stalled
- [ ] Each finger reaches its stop without the servo stalling
- [ ] Contact detection fires on a rigid object (`neurogrip console` → `status`)
- [ ] Contact detection fires on a *compliant* object — this is the harder case
- [ ] EMG noise floor below 150 µV at rest
- [ ] EMG SNR above 20× at maximum contraction
- [ ] Camera delivers ≥ 20 fps at the configured resolution
- [ ] Firmware watchdog trips: unplug the host mid-grasp, drive must disable
- [ ] E-stop button engages and requires an explicit acknowledgement to clear

The last two are the ones that matter. Do not skip them.

## Known limitations of this build

- **No hardware safety interlock.** The e-stop is software on the same MCU as the
  drive. A certifiable device needs a contactor that removes actuator power with
  no software in the path.
- **No fingertip force sensing.** Grip force is inferred from motor current, and
  the current-to-force constant is nominal rather than measured.
- **No per-motor thermistors** on revision A; temperature is modelled from
  current, which over-estimates — the safe direction for a limit, but not a
  measurement.
- **Monocular camera**, so depth comes from class size priors.
