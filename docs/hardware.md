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

Per finger: minimum pulse (fully open), maximum pulse (fully closed at the
mechanical stop), and direction.

```bash
neurogrip console
neurogrip> arm
neurogrip> grip open
neurogrip> grip fist
```

Watch for the finger reaching its stop *before* the servo does. A servo driving
into a hard stop stalls, draws maximum current, and destroys the horn or the
tendon within minutes. Set `max_pulse_us` short of that point.

The reference hand limits the thumb to 0.92 closure because at full travel it
collides with the index proximal phalanx — see `HandKinematics.COLLISION_RULES`,
which enforces this in software as well.

`TODO(persistence)`: firmware calibration is currently lost on reset and re-sent
by the host at startup; NVS storage is not yet implemented.

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
its focal length from it and will be wrong if it is wrong.

## Assembly order

1. Print/machine the frame; ream the tendon channels and line them with PTFE.
2. Fit the servos; route and tension the tendons with the servos at neutral.
3. Wire the controller (see `firmware/.../include/pins.h`).
4. Flash the firmware: `cd firmware/esp32_motor_controller && pio run -t upload`.
5. **Calibrate the servo endpoints before enabling drive with tendons attached.**
6. Fit the EMG front end and electrodes.
7. `neurogrip diagnose` — everything should pass or warn, nothing should fail.
8. `neurogrip calibrate` for the user's EMG.
9. `neurogrip run --profile hardware`.

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
