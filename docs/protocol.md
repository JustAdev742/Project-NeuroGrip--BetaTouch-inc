# NeuroGrip Protocol (NGP) v1

Host ⇄ motor controller, over USB serial.

Three implementations must stay byte-for-byte compatible:

| Side | File | Baud |
|---|---|---|
| Host | `src/neurogrip/hal/protocol.py` | — |
| micro:bit firmware | `firmware/microbit_servo_controller/main.py` | 115200 |
| ESP32 firmware | `firmware/esp32_motor_controller/include/ngp_protocol.h` | 921600 |

`tests/unit/test_hal_and_protocol.py` pins the encodings and is the arbiter when
they disagree.

Both boards speak the same protocol deliberately: the host driver, the framing,
the CRC checking and the watchdog contract are shared code rather than parallel
implementations, and shared code is code both boards' tests exercise.

The micro:bit implements a subset — `HOME` and `REBOOT` answer
`ERROR/UNSUPPORTED`, and `SET_FORCE` is accepted and ignored because force
control needs current sensing that board has not got. Its firmware loop runs at
50 Hz (one servo frame), so the host driver coalesces target writes down to that
rate; events — stop, enable, disable, calibration — are never coalesced.
`firmware/microbit_servo_controller/README.md` has the full table.

## Frame format

```
+------+------+--------+--------+-----+---------+--------+
| 0xA5 | 0x5A | LENGTH | MSG_ID | SEQ | PAYLOAD | CRC16  |
+------+------+--------+--------+-----+---------+--------+
  sync   sync    u8       u8      u8   LENGTH B    u16
```

Little-endian throughout. `LENGTH` counts payload bytes only. The CRC
(CCITT-FALSE, init `0xFFFF`) covers `LENGTH`, `MSG_ID`, `SEQ` and `PAYLOAD` —
everything except the sync pattern, which is not part of the message.

Reference: `crc16("123456789") == 0x29B1`.

The parser on both sides is incremental and resynchronising. A serial line that
comes up mid-frame, drops bytes, or receives ESP32 bootloader chatter must
recover without a reset. Every discarded byte is counted, so the diagnostics
screen can show link quality rather than just "it works / it doesn't".

## Design decisions

**Fixed-point, not floats.** Finger positions travel as `uint16` in units of
1/10000 of full closure. Integer maths keeps the MCU side cheap and removes any
float-format ambiguity between the two platforms.

**One command for all five fingers.** A grasp is a coordinated multi-finger
motion; one atomic `SET_TARGETS` per control cycle keeps the fingers
synchronised, halves the link traffic, and gives the firmware a single
unambiguous watchdog to refresh.

**State is pushed, not polled.** The controller streams `STATE` at 100 Hz; the
host only polls on demand for diagnostics. This keeps the round trip out of the
control loop's critical path.

**The MCU owns the safety timeout.** Every `SET_TARGETS` refreshes a firmware
watchdog. Safety must not depend on the Linux side being alive.

## Messages

### Host → MCU

| ID | Name | Payload | Notes |
|---|---|---|---|
| 0x01 | `PING` | `u32 token` | Replies `PONG` |
| 0x02 | `SET_TARGETS` | `u16[5] pos, u8 speed, u8 flags` | Refreshes the watchdog |
| 0x03 | `SET_LIMITS` | `u16 vel, u16 accel, u16 mA, u16 °C` | May only tighten |
| 0x04 | `ENABLE` | `u8 mask` | Bit 0 = thumb |
| 0x05 | `DISABLE` | `u8 mask` | |
| 0x06 | `ESTOP` | — | Latches |
| 0x07 | `CLEAR_ESTOP` | `u16 magic = 0x5EA1` | |
| 0x08 | `REQUEST_STATE` | — | |
| 0x09 | `SET_CALIBRATION` | `u8 finger, u16 min_us, u16 max_us, u8 inv` | |
| 0x0A | `HOME` | — | |
| 0x0B | `SET_FORCE` | `u8 mask, u8 force` | |
| 0x0C | `SET_WATCHDOG` | `u16 timeout_ms` | |
| 0x0D | `REBOOT` | — | |

`speed` encodes the speed scale as `scale × 127.5`, giving 0–2.0×.
Velocity and acceleration are transmitted in 1/1000 closure units.

`CLEAR_ESTOP` carries a magic word so a corrupted frame cannot re-enable drive by
accident.

### MCU → host

| ID | Name | Payload |
|---|---|---|
| 0x81 | `PONG` | `u32 token, u8 major, u8 minor, u8 patch, u32 uptime_ms` |
| 0x82 | `STATE` | see below |
| 0x83 | `EVENT` | `u8 code, u8 finger, u16 detail` |
| 0x84 | `ERROR` | `u8 code, u16 detail` |
| 0x85 | `LOG` | `u8 level, char[] text` |

`STATE` (47 bytes):

```
u8  sequence
u8  flags
    bit 0 ENABLED    bit 4 OVERCURRENT
    bit 1 MOVING     bit 5 OVERTEMP
    bit 2 ESTOP      bit 6 WATCHDOG_TRIPPED
    bit 3 HOMED      bit 7 UNDERVOLTAGE
[ u16 position, u16 target, u16 current_ma, i8 temperature_c ] × 5
u16 bus_voltage_mv
u32 uptime_ms
```

`finger = 0xFF` in an `EVENT` means "all fingers".

**Event codes:** 1 `STALL_DETECTED`, 2 `TARGET_REACHED`, 3 `CONTACT_DETECTED`,
4 `HOMING_COMPLETE`, 5 `ESTOP_ENGAGED`, 6 `ESTOP_RELEASED`, 7 `WATCHDOG_TRIP`,
8 `THERMAL_THROTTLE`.

**Error codes:** 1 `UNKNOWN_MESSAGE`, 2 `BAD_LENGTH`, 3 `BAD_PARAMETER`,
4 `NOT_ENABLED`, 5 `ESTOP_ACTIVE`, 6 `NOT_HOMED`, 7 `OVERCURRENT`, 8 `OVERTEMP`,
9 `UNDERVOLTAGE`, 10 `HARDWARE_FAULT`.

## Typical exchange

```
host                              MCU
 │── SET_WATCHDOG 300ms ─────────▶│
 │── SET_LIMITS ─────────────────▶│
 │── PING ───────────────────────▶│
 │◀────────────────────── PONG ───│  firmware version
 │── ENABLE 0x1F ────────────────▶│
 │── HOME ───────────────────────▶│
 │◀──── EVENT HOMING_COMPLETE ────│
 │                                │
 │── SET_TARGETS ────────────────▶│  every control cycle, 200 Hz
 │◀────────────────────── STATE ──│  every 10 ms, 100 Hz
 │◀── EVENT CONTACT_DETECTED ─────│  a finger met an object
 │                                │
 │      (host goes quiet)         │
 │◀───── EVENT WATCHDOG_TRIP ─────│  300 ms later: drive disabled
```

## Testing without hardware

`src/neurogrip/hal/servo/emulator.py` implements the MCU side in Python. Paired
with `LoopbackTransport`, the *production* driver runs against it over the *real*
framing and protocol — so encode/decode, sequencing, CRC recovery, the e-stop
latch and the firmware watchdog are all genuinely exercised in CI.

The loopback also models link degradation (latency, byte loss, bit corruption),
which is how `test_a_lossy_link_produces_crc_errors_but_stays_usable` proves the
parser resynchronises rather than acting on garbage.

## Adding a message

1. Add the ID to `MessageId` (Python) and `ngp_message_id_t` (C).
2. Add encode/decode functions to `protocol.py`.
3. Add a handler to `main.cpp` **and** to `emulator.py`.
4. Add a round-trip test.

Keep host→MCU IDs below `0x80` and MCU→host at or above it.
