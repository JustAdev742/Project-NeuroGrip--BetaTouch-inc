# micro:bit motor controller

NeuroGrip's reference motor controller: a BBC micro:bit v2 in an edge breakout
board, driving five servos, speaking NGP v1 over USB serial.

```bash
pip install uflash
uflash main.py            # or open main.py in python.microbit.org and download
```

The display shows one character of status: `-` idle, `=` drive enabled,
`?` watchdog tripped, `!` emergency stop.

---

## Power — read this first

**Do not power servos from the micro:bit's 3V pad.** It supplies roughly 190 mA.
One SG90 stalls at about 700 mA and five together can pull over 3 A. Powering
them from the board browns it out, resets it mid-motion, and can damage the
regulator.

```
   5 V supply (+) ──┬──────────────── servo V+   ×5
                    │
   5 V supply (−) ──┴──┬───────────── servo GND  ×5
                       │
   micro:bit GND ──────┘                ← common ground, not optional

   micro:bit P0/P1/P2/P8/P12 ────────── servo signal ×5
```

Without the common ground the servos see no valid signal reference and will
twitch, buzz, or sit still.

Budget roughly 1 A per servo for a 5 V supply. Four AA cells work for bench
testing one or two; a 5 V 3 A adapter or a BEC is right for five.

---

## Five servos needs a driver board

micro:bit MicroPython limits how many pins can output PWM at once — commonly
documented as three. Five direct channels is past that, and the symptom is
servos that twitch or ignore commands with nothing obviously wrong.

The firmware therefore prefers a **PCA9685 16-channel servo driver** on I2C, and
auto-detects one at address `0x40`:

| micro:bit (via breakout) | PCA9685 |
|---|---|
| P19 (SCL) | SCL |
| P20 (SDA) | SDA |
| 3V | VCC (logic) |
| GND | GND |
| — | V+ from the 5 V servo supply |

With a driver board found it logs `pca9685`; with none it falls back to direct
pins and logs `direct`. Direct drive is fine for bench-testing one to three
servos and unreliable for five.

Check which is in use:

```bash
neurogrip info | grep driver_board
```

---

## Pin map

Finger order matches `Finger` on the host: thumb, index, middle, ring, pinky.

| Finger | Direct pin | PCA9685 channel |
|---|---|---|
| Thumb | P0 | 0 |
| Index | P1 | 1 |
| Middle | P2 | 2 |
| Ring | P8 | 3 |
| Pinky | P12 | 4 |

P0–P2 are the large ring pads. P8 and P12 are free GPIO on v2. Avoided
deliberately: P3/P4/P6/P7/P9/P10 drive the LED matrix, P5/P11 are the buttons,
P19/P20 are I2C. The display therefore stays usable as a status indicator.

Change `PIN_SERVO` in `main.py` if your breakout wiring differs — nothing else
in the firmware refers to a pin.

---

## Emergency stop

**Press buttons A and B together.** Handled on the micro:bit itself, so it works
whether or not the host is running — the same reasoning as the watchdog. The
host learns about it from the `ESTOP_ENGAGED` event and latches its own stop.

Like the ESP32 build's button, this is a software-polled input, not a hardware
interlock. A certifiable device needs a contactor that removes actuator power
with no software in the path. See `docs/safety.md`, "Limits of this design".

---

## What this board cannot do

The ESP32 controller carries a shunt amplifier per finger. This one has no
current sensing, no position feedback and no bus-voltage measurement, so:

| Feature | Status on micro:bit |
|---|---|
| Contact detection | **Unavailable** — needs motor current |
| Adaptive grip force | **Unavailable** — needs motor current |
| Tendon-slack calibration | **Unavailable** — take-up detection *is* a current measurement |
| Stall / servo-timeout detection | **Unavailable** — cannot tell stalled from loaded |
| Reported position | Open-loop estimate, not a measurement |
| Bus voltage | Reported as 0; the supply self-test says "not measured" |

`MicrobitServoBus` declares the missing capabilities rather than reporting zeros
that would be mistaken for measurements, and the layers above check the
capability instead of assuming it. A zero meaning "no sensor" must never be read
as a zero meaning "no load".

Fitting an INA181 (or similar) per channel into the spare analog pins would
restore some of this; micro:bit has only three free analog inputs with the
display enabled, so it would not restore all five.

---

## Protocol

NGP v1, the same wire format as the ESP32 firmware — `docs/protocol.md` is the
contract and `src/neurogrip/hal/protocol.py` is the source of truth. Implemented
subset:

| Message | Status |
|---|---|
| `PING` / `PONG` | ✔ |
| `SET_TARGETS` | ✔ (coalesced by the host to 50 Hz) |
| `SET_LIMITS` | ✔ velocity only |
| `ENABLE` / `DISABLE` | ✔ |
| `ESTOP` / `CLEAR_ESTOP` | ✔ |
| `REQUEST_STATE` / `STATE` | ✔ streamed at 25 Hz |
| `SET_CALIBRATION` | ✔ applied; `slack` cannot be *measured* here |
| `SET_WATCHDOG` | ✔ |
| `SET_FORCE` | Accepted and ignored — needs current sensing |
| `HOME` | `ERROR/UNSUPPORTED` — no endstops, no feedback |
| `REBOOT` | `ERROR/UNSUPPORTED` — use the reset button |

The firmware loop runs at 50 Hz because that is the rate a hobby servo actually
has: one pulse per 20 ms frame. The host still runs its control loop at 200 Hz
and coalesces target writes down to 50, keeping the latest. Emergency stop,
enable, disable and calibration are never coalesced — those are events, not
samples.

### Watchdog

Every `SET_TARGETS` refreshes a firmware timeout (400 ms by default). If the host
stops talking for any reason — crash, unplugged cable, kernel stall — the
firmware de-energises the servos by itself. Safety does not depend on the Linux
side being alive.

De-energising **holds position** rather than driving open: releasing the PWM
signal lets the servo relax. If the hand is carrying something, dropping it is
worse than stopping.

---

## Bring-up

With the hand not yet built, and the servos loose on the bench:

```bash
neurogrip config --check
neurogrip test link           # round-trip latency, loss, framing errors
neurogrip test servos         # ⚠ moves — each servo, then all five
```

`test servos` sweeps each channel individually first, which is the only way to
work out which physical servo is on which channel, then all five together, which
is what finds an undersized supply. Watch the hardware while it runs: on a board
with no position feedback the reported positions are the firmware's own estimate,
and the tool says so.

```bash
neurogrip test servos --finger thumb --cycles 3     # one channel, repeatedly
neurogrip test servos --travel 0.4 --speed 0.3      # partly assembled: gentler
```

Once the tendons are strung, `neurogrip test range` and `neurogrip calibrate
servo` become meaningful — but the latter needs current sensing, so on this board
it will refuse rather than produce a slack figure it cannot measure.
