"""NeuroGrip motor controller for the BBC micro:bit v2.

Speaks NGP v1 — the same wire format as the ESP32 controller — so the host
driver, the framing, the CRC checking and the watchdog contract are shared code
rather than a parallel implementation. Only a subset is implemented; see
NOT IMPLEMENTED below.

Flash with ``uflash main.py``, or open it in the Python editor and download.

READ THIS BEFORE CONNECTING SERVOS
==================================

**1. Do not power servos from the micro:bit's 3V pad.**

That rail supplies roughly 190 mA on a v2. One SG90 stalls at about 700 mA, and
five together can pull over 3 A. Powering them from the board browns it out,
resets it mid-motion, and can damage the regulator. Use a separate 5 V supply
(4×AA, a 5 V 3 A adapter, or a BEC) and **join the grounds**:

    external 5 V ──┬── servo V+  (all five)
    micro:bit GND ─┴── servo GND (all five)    ← common ground is essential
    micro:bit pins ─── servo signal

**2. Five simultaneous PWM channels is more than MicroPython reliably gives you.**

The micro:bit's MicroPython limits how many pins can output PWM at once —
commonly documented as three. This firmware therefore prefers a **PCA9685
16-channel servo driver** on I2C (P19/P20 on the edge breakout), which is the
usual way to drive this many servos from a micro:bit and also solves the power
problem, since those boards take the servo supply on their own terminal block.

It auto-detects the PCA9685 at start-up. With no driver board found it falls
back to driving pins directly and reports ``direct`` in the log, which is fine
for bench-testing one to three servos and unreliable for five.

**3. Signal pins are 3.3 V.**

Most hobby servos accept that as a valid logic high. A few need 5 V and will
jitter or ignore it — if one channel misbehaves on known-good wiring, a level
shifter on the signal line is the fix. The PCA9685 outputs at its own logic
level and avoids this entirely.

WHAT THIS BOARD CANNOT DO
=========================

The ESP32 controller reads a shunt amplifier per finger, which is what gives the
host motor current for contact detection, stall detection, force estimation and
tendon-slack calibration. **There is no current sensing here**, so this firmware
reports zero and the host must infer nothing from it. ``MicrobitServoBus``
declares reduced capabilities so the layers above degrade honestly instead of
believing a constant zero.

There is no position feedback either. A hobby servo has an internal
potentiometer but does not report it. The position in ``STATE`` is the
*commanded* position advanced through a rate limiter — an open-loop estimate, not
a measurement.

NOT IMPLEMENTED — the firmware replies ERROR/UNSUPPORTED:

    HOME     no endstops and no feedback; the host homes by commanding open
    REBOOT   use the reset button

``SET_CALIBRATION`` is accepted and applied to the pulse mapping, but the
``slack`` term it carries can only be *measured* on a board with current sensing.
Set it from a hand calibrated on an ESP32 controller, or leave it at zero.
"""

from microbit import (
    button_a,
    button_b,
    display,
    i2c,
    pin0,
    pin1,
    pin2,
    pin8,
    pin12,
    running_time,
    uart,
)
import struct

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FIRMWARE_VERSION = (1, 0, 0)
FINGER_COUNT = 5

#: Fallback direct-drive pins, in Finger order: thumb, index, middle, ring,
#: pinky. Chosen to avoid the pins the LED matrix needs (P3/P4/P6/P7/P9/P10),
#: the buttons (P5/P11) and I2C (P19/P20), so the display stays usable.
PIN_SERVO = (pin0, pin1, pin2, pin8, pin12)

#: PCA9685 channels used when a driver board is present.
PCA_CHANNEL = (0, 1, 2, 3, 4)
PCA_ADDRESS = 0x40

MIN_PULSE_US = 1000
MAX_PULSE_US = 2000

#: 20 ms — the standard hobby-servo frame. Also the reason the control loop runs
#: at 50 Hz rather than the host's 200: that is the rate the actuator has.
SERVO_PERIOD_US = 20000
LOOP_MS = 20

#: Maximum closure change per loop before the host sets a limit. Conservative:
#: a servo slammed between endpoints draws stall current and will brown out an
#: under-specified supply.
DEFAULT_STEP = 0.04

POSITION_SCALE = 10000
SERIAL_BAUD = 115200

# ---------------------------------------------------------------------------
# NGP v1 — must match src/neurogrip/hal/protocol.py byte for byte
# ---------------------------------------------------------------------------

SOF = b"\xa5\x5a"
MAX_PAYLOAD = 64

MSG_PING = 0x01
MSG_SET_TARGETS = 0x02
MSG_SET_LIMITS = 0x03
MSG_ENABLE = 0x04
MSG_DISABLE = 0x05
MSG_ESTOP = 0x06
MSG_CLEAR_ESTOP = 0x07
MSG_REQUEST_STATE = 0x08
MSG_SET_CALIBRATION = 0x09
MSG_HOME = 0x0A
MSG_SET_FORCE = 0x0B
MSG_SET_WATCHDOG = 0x0C
MSG_REBOOT = 0x0D

MSG_PONG = 0x81
MSG_STATE = 0x82
MSG_EVENT = 0x83
MSG_ERROR = 0x84

FLAG_ENABLED = 1 << 0
FLAG_MOVING = 1 << 1
FLAG_ESTOP = 1 << 2
FLAG_HOMED = 1 << 3
FLAG_WATCHDOG_TRIPPED = 1 << 6

EVENT_TARGET_REACHED = 2
EVENT_ESTOP_ENGAGED = 5
EVENT_ESTOP_RELEASED = 6
EVENT_WATCHDOG_TRIP = 7

ERR_BAD_CRC = 1
ERR_BAD_LENGTH = 2
ERR_UNKNOWN_MSG = 3
ERR_BAD_PARAMETER = 4
ERR_UNSUPPORTED = 5


def crc16(data):
    """CRC-16/CCITT-FALSE. Identical loop to the host and the ESP32 firmware."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


_sequence = 0


def send(msg_id, payload=b""):
    global _sequence
    _sequence = (_sequence + 1) & 0xFF
    body = bytes([len(payload), msg_id, _sequence]) + payload
    uart.write(SOF + body + struct.pack("<H", crc16(body)))


def send_error(code, detail=0):
    send(MSG_ERROR, struct.pack("<BB", code, detail & 0xFF))


def send_event(code, finger=0xFF, detail=0):
    send(MSG_EVENT, struct.pack("<BBH", code, finger, detail & 0xFFFF))


class FrameParser:
    """Incremental, resynchronising NGP frame parser.

    A serial line that comes up mid-frame, drops bytes, or receives boot-loader
    chatter must recover on its own without a reset — the host may reconnect at
    any moment and will not reboot this board to do it.
    """

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data):
        self._buffer.extend(data)
        frames = []
        while True:
            start = self._buffer.find(SOF)
            if start < 0:
                # Keep one byte: the sync pattern may straddle two reads.
                if len(self._buffer) > 1:
                    del self._buffer[: len(self._buffer) - 1]
                return frames
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 7:
                return frames
            length = self._buffer[2]
            if length > MAX_PAYLOAD:
                del self._buffer[:2]
                continue
            total = 7 + length
            if len(self._buffer) < total:
                return frames
            body = bytes(self._buffer[2 : 5 + length])
            crc_received = self._buffer[5 + length] | (self._buffer[6 + length] << 8)
            if crc16(body) == crc_received:
                frames.append((self._buffer[3], bytes(self._buffer[5 : 5 + length])))
                del self._buffer[:total]
            else:
                send_error(ERR_BAD_CRC, self._buffer[3])
                del self._buffer[:2]


# ---------------------------------------------------------------------------
# Output drivers
# ---------------------------------------------------------------------------


class DirectDriver:
    """PWM straight off the micro:bit pins.

    Adequate for one to three servos on the bench. Beyond that MicroPython runs
    out of PWM channels and the extra servos will twitch or sit still — which is
    why :func:`make_driver` prefers the PCA9685 when one is present.
    """

    name = "direct"

    def __init__(self):
        for pin in PIN_SERVO:
            pin.set_analog_period_microseconds(SERVO_PERIOD_US)

    def write_pulse(self, index, pulse_us):
        # write_analog takes 0–1023 as a fraction of the analog period.
        PIN_SERVO[index].write_analog(int(pulse_us * 1023 // SERVO_PERIOD_US))

    def release(self, index):
        """Stop driving so the servo relaxes instead of holding torque."""
        PIN_SERVO[index].write_analog(0)


class Pca9685Driver:
    """16-channel I2C PWM driver — the reliable way to run five servos.

    Only the registers this firmware needs are implemented; a full driver would
    be more code than the rest of this file for no benefit here.
    """

    name = "pca9685"

    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, address=PCA_ADDRESS):
        self._address = address
        self._write(self.MODE1, 0x00)
        # 50 Hz: prescale = round(25 MHz / (4096 × 50)) − 1 = 121
        self._write(self.MODE1, 0x10)  # sleep, required to set the prescale
        self._write(self.PRESCALE, 121)
        self._write(self.MODE1, 0x00)
        # The oscillator needs ~500 µs to stabilise; the loop's first tick is
        # 20 ms away, so no explicit delay is needed.
        self._write(self.MODE1, 0xA1)  # restart + auto-increment

    def _write(self, register, value):
        i2c.write(self._address, bytes([register, value]))

    def write_pulse(self, index, pulse_us):
        channel = PCA_CHANNEL[index]
        # 4096 counts per 20 ms frame.
        count = int(pulse_us * 4096 // SERVO_PERIOD_US)
        if count > 4095:
            count = 4095
        base = self.LED0_ON_L + 4 * channel
        i2c.write(
            self._address,
            bytes([base, 0, 0, count & 0xFF, (count >> 8) & 0x0F]),
        )

    def release(self, index):
        channel = PCA_CHANNEL[index]
        base = self.LED0_ON_L + 4 * channel
        # Bit 4 of the OFF_H register forces the output fully off.
        i2c.write(self._address, bytes([base, 0, 0, 0, 0x10]))


def make_driver():
    """Use a PCA9685 if one answers on I2C, otherwise drive pins directly."""
    try:
        if PCA_ADDRESS in i2c.scan():
            return Pca9685Driver()
    except Exception:
        pass
    return DirectDriver()


# ---------------------------------------------------------------------------
# Servo state
# ---------------------------------------------------------------------------


class Finger:
    """One servo channel. Open-loop: position is an estimate, not a measurement."""

    def __init__(self, index):
        self.index = index
        self.position = 0.0
        self.target = 0.0
        self.min_pulse = MIN_PULSE_US
        self.max_pulse = MAX_PULSE_US
        self.inverted = False
        self.slack = 0.0
        self.enabled = False

    def pulse_for(self, closure):
        travel = 0.0 if closure < 0.0 else (1.0 if closure > 1.0 else closure)
        # Map [0,1] of finger motion onto [slack,1] of servo motion. Must match
        # ServoCalibration.to_pulse() on the host exactly.
        travel = self.slack + travel * (1.0 - self.slack)
        if self.inverted:
            travel = 1.0 - travel
        return self.min_pulse + travel * (self.max_pulse - self.min_pulse)


fingers = [Finger(i) for i in range(FINGER_COUNT)]
driver = None

state = {
    "enabled_mask": 0,
    "estop": False,
    "step": DEFAULT_STEP,
    "speed_scale": 1.0,
    "watchdog_ms": 300,
    "last_command_ms": 0,
    "watchdog_tripped": False,
    "sequence": 0,
}


def safe_actuators(reason_event):
    """De-energise every channel, holding position rather than opening.

    Releasing the PWM signal lets the servo relax rather than back-driving hard.
    If the hand is carrying something, dropping it is worse than stopping.
    """
    state["enabled_mask"] = 0
    for finger in fingers:
        finger.enabled = False
        finger.target = finger.position
        driver.release(finger.index)
    if reason_event is not None:
        send_event(reason_event)


def engage_estop():
    state["estop"] = True
    safe_actuators(EVENT_ESTOP_ENGAGED)
    display.show("!")


# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------


def handle(msg_id, payload):
    if msg_id == MSG_PING:
        token = struct.unpack("<I", payload)[0] if len(payload) >= 4 else 0
        send(
            MSG_PONG,
            struct.pack(
                "<IBBBI",
                token,
                FIRMWARE_VERSION[0],
                FIRMWARE_VERSION[1],
                FIRMWARE_VERSION[2],
                running_time(),
            ),
        )

    elif msg_id == MSG_SET_TARGETS:
        if len(payload) != FINGER_COUNT * 2 + 2:
            send_error(ERR_BAD_LENGTH, len(payload))
            return
        state["last_command_ms"] = running_time()
        state["watchdog_tripped"] = False
        if state["estop"]:
            return
        values = struct.unpack("<HHHHHBB", payload)
        for i in range(FINGER_COUNT):
            fingers[i].target = values[i] / POSITION_SCALE
        state["speed_scale"] = values[FINGER_COUNT] / 127.5

    elif msg_id == MSG_SET_LIMITS:
        if len(payload) != 8:
            send_error(ERR_BAD_LENGTH, len(payload))
            return
        max_velocity = struct.unpack("<HHHH", payload)[0] / 1000.0
        # Closure units per second, converted to a per-loop step.
        state["step"] = max(0.002, max_velocity * LOOP_MS / 1000.0)
        state["last_command_ms"] = running_time()

    elif msg_id == MSG_ENABLE:
        if state["estop"]:
            send_error(ERR_BAD_PARAMETER, 0)
            return
        mask = payload[0] if payload else 0x1F
        state["enabled_mask"] |= mask & 0x1F
        for finger in fingers:
            if state["enabled_mask"] & (1 << finger.index):
                finger.enabled = True
        state["last_command_ms"] = running_time()
        display.show("=")

    elif msg_id == MSG_DISABLE:
        mask = payload[0] if payload else 0x1F
        state["enabled_mask"] &= ~(mask & 0x1F) & 0x1F
        for finger in fingers:
            if not state["enabled_mask"] & (1 << finger.index):
                finger.enabled = False
                driver.release(finger.index)
        display.show("-")

    elif msg_id == MSG_ESTOP:
        engage_estop()

    elif msg_id == MSG_CLEAR_ESTOP:
        state["estop"] = False
        send_event(EVENT_ESTOP_RELEASED)
        display.show("-")

    elif msg_id == MSG_REQUEST_STATE:
        send_state()

    elif msg_id == MSG_SET_CALIBRATION:
        if len(payload) != 7:
            send_error(ERR_BAD_LENGTH, len(payload))
            return
        index, min_pulse, max_pulse, inverted, slack = struct.unpack("<BHHBB", payload)
        if index >= FINGER_COUNT:
            send_error(ERR_BAD_PARAMETER, index)
            return
        # Reject endpoints outside what a hobby servo accepts: a bad calibration
        # would drive the horn past its mechanical stop.
        if min_pulse < 500 or max_pulse > 2500 or min_pulse >= max_pulse:
            send_error(ERR_BAD_PARAMETER, index)
            return
        finger = fingers[index]
        finger.min_pulse = min_pulse
        finger.max_pulse = max_pulse
        finger.inverted = inverted != 0
        finger.slack = slack / 255.0

    elif msg_id == MSG_SET_WATCHDOG:
        if len(payload) != 2:
            send_error(ERR_BAD_LENGTH, len(payload))
            return
        state["watchdog_ms"] = struct.unpack("<H", payload)[0]
        state["last_command_ms"] = running_time()

    elif msg_id == MSG_SET_FORCE:
        # Accepted and ignored. Force control needs current sensing, which this
        # board has not got. Accepting keeps the host's normal command sequence
        # working, and the host already knows not to trust force here because
        # the driver does not declare the capability.
        state["last_command_ms"] = running_time()

    elif msg_id in (MSG_HOME, MSG_REBOOT):
        send_error(ERR_UNSUPPORTED, msg_id)

    else:
        send_error(ERR_UNKNOWN_MSG, msg_id)


def send_state():
    state["sequence"] = (state["sequence"] + 1) & 0xFF
    flags = 0
    if state["enabled_mask"]:
        flags |= FLAG_ENABLED
    if state["estop"]:
        flags |= FLAG_ESTOP
    if state["watchdog_tripped"]:
        flags |= FLAG_WATCHDOG_TRIPPED
    for finger in fingers:
        if abs(finger.target - finger.position) > 0.002:
            flags |= FLAG_MOVING
            break

    parts = [struct.pack("<BB", state["sequence"], flags)]
    for finger in fingers:
        parts.append(
            struct.pack(
                "<HHHb",
                int(finger.position * POSITION_SCALE),
                int(finger.target * POSITION_SCALE),
                0,  # no current sensing on this board
                25,  # no temperature sensing either
            )
        )
    # Bus voltage is not measured. Report zero rather than a plausible constant:
    # the host's supply check must not pass on a number nobody read.
    parts.append(struct.pack("<HI", 0, running_time()))
    send(MSG_STATE, b"".join(parts))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def step_motion():
    """Advance every enabled finger towards its target, rate-limited."""
    step = state["step"] * max(0.05, min(2.0, state["speed_scale"]))
    for finger in fingers:
        if not finger.enabled:
            continue
        error = finger.target - finger.position
        if abs(error) <= step:
            if abs(error) > 1e-6:
                finger.position = finger.target
                send_event(EVENT_TARGET_REACHED, finger.index)
            else:
                continue  # already there; no need to rewrite the same pulse
        else:
            finger.position += step if error > 0 else -step
        driver.write_pulse(finger.index, finger.pulse_for(finger.position))


def main():
    global driver

    uart.init(baudrate=SERIAL_BAUD)
    driver = make_driver()
    parser = FrameParser()
    display.show("-")
    state["last_command_ms"] = running_time()
    next_loop = running_time()
    state_divider = 0

    while True:
        now = running_time()

        # Buttons A+B together are the hardware emergency stop. Read here, on
        # the controller, so it works whether or not the host is alive — the
        # same reasoning as the watchdog below. The host learns about it from
        # the ESTOP_ENGAGED event.
        if button_a.is_pressed() and button_b.is_pressed() and not state["estop"]:
            engage_estop()

        if uart.any():
            for msg_id, payload in parser.feed(uart.read()):
                try:
                    handle(msg_id, payload)
                except Exception:
                    # A malformed message must never take the controller down:
                    # the hand would be left energised with nobody driving it.
                    send_error(ERR_BAD_PARAMETER, msg_id)

        if now < next_loop:
            continue
        next_loop = now + LOOP_MS

        # Firmware watchdog. If the host stops talking for any reason at all —
        # crash, unplugged cable, kernel stall — the actuators are safed here,
        # with no host involvement. This is what makes the system's safety
        # independent of the Linux side being alive.
        if (
            state["watchdog_ms"]
            and not state["estop"]
            and state["enabled_mask"]
            and now - state["last_command_ms"] > state["watchdog_ms"]
        ):
            state["watchdog_tripped"] = True
            safe_actuators(EVENT_WATCHDOG_TRIP)
            display.show("?")

        step_motion()

        # State at 25 Hz — every second loop. Enough for the host's staleness
        # checks without spending the serial budget on telemetry.
        state_divider += 1
        if state_divider >= 2:
            state_divider = 0
            send_state()


main()
