# ESP32 motor controller

Firmware for the five tendon-driven finger servos.

## What it does, and what it deliberately does not

**Does:** generate servo PWM, enforce velocity/acceleration/current/temperature
limits, stream telemetry at 100 Hz, and **safe the actuators by itself if the
host stops talking.**

**Does not:** plan trajectories, choose grasps, or know anything about intent.
Those live on the host, because a motion must be interruptible within one control
cycle — and a profile executing on the MCU would make a cancel wait for a round
trip.

## Why the watchdog lives here

If the Linux host crashes, is unplugged, hits an OOM kill or stalls in the
kernel, nothing on that side can react. So the timeout that safes the hand runs
on this processor, where a host failure cannot reach it. Every `SET_TARGETS`
refreshes it; 300 ms of silence and drive is disabled.

This is the single most important behaviour in the firmware. Do not make it
conditional on anything the host controls.

## Building

```bash
pip install platformio
cd firmware/esp32_motor_controller
pio run                # build
pio run -t upload      # flash
pio device monitor     # watch the log stream
```

## Protocol

NGP v1 over USB-CDC at 921600 baud. The wire format is defined in
`include/ngp_protocol.h` and must stay byte-for-byte compatible with
`src/neurogrip/hal/protocol.py`. `docs/protocol.md` documents the contract;
`tests/unit/test_hal_and_protocol.py` pins the encodings and is the arbiter when
the two disagree.

## Testing without hardware

`src/neurogrip/hal/servo/emulator.py` implements this firmware's behaviour in
Python. The host-side driver runs against it over an in-process loopback, so the
real framing, CRC, sequencing, e-stop latch and watchdog are all exercised in CI:

```bash
pytest tests/unit/test_hal_and_protocol.py -k Esp32Driver
```

**When this firmware changes, the emulator must change with it.** That coupling
is deliberate — it is what keeps the two implementations honest.

## Architecture

Two FreeRTOS tasks pinned to separate cores:

| Core | Task | Rate | Work |
|---|---|---|---|
| 0 | `comms` | 100 Hz | parse frames, emit telemetry |
| 1 | `control` | 200 Hz | servo update, limits, watchdog |

The control task never blocks on I/O. Shared state crosses between them through
a spinlock-protected block.

## Safety behaviour

| Condition | Response |
|---|---|
| Host silent > watchdog | Disable drive, hold position, emit `WATCHDOG_TRIP` |
| `ESTOP` message | Latch, disable drive, emit `ESTOP_ENGAGED` |
| Over-current | Emit `ERROR`, engage e-stop |
| Over-temperature | Derate the current limit, emit `THERMAL_THROTTLE` |
| Under-voltage | Set the `UNDERVOLTAGE` flag in telemetry |
| Power-on | Start **de-energised**; the host must explicitly enable |

Clearing the e-stop requires the magic word `0x5EA1`, so a corrupted frame cannot
re-enable drive by accident.

Limits sent by the host are clamped against absolute ceilings compiled into the
firmware: `SET_LIMITS` can only ever tighten them.

## Wiring

See `include/pins.h` for the pin map and `docs/hardware.md` for the harness.

Current sensing uses INA181 shunt amplifiers on ADC1 (ADC2 is unavailable while
Wi-Fi is active). Servo pins must be PWM-capable and on a timer group not shared
with the ADC.

## TODOs

- `TODO(persistence)`: write per-finger servo calibration to NVS so it survives a
  power cycle. Currently it is lost on reset and must be re-sent by the host.
- `TODO(hardware)`: calibrate the current-sense constant per unit against a bench
  supply; the value in `read_current_ma` is nominal.
- Per-motor thermistors are not fitted on revision A; temperature is modelled
  from current, which over-estimates (the safe direction for a limit).
