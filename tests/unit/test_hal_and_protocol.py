"""HAL: framing, the NGP protocol, the servo bus and the firmware emulator."""

from __future__ import annotations

import pytest

from neurogrip.core.errors import ProtocolError
from neurogrip.core.types import Finger, HandPose
from neurogrip.hal.protocol import (
    ErrorCode,
    EventCode,
    FingerTelemetry,
    StatusFlags,
    decode_error,
    decode_event,
    decode_pong,
    decode_set_limits,
    decode_set_targets,
    decode_state,
    encode_clear_estop,
    encode_error,
    encode_event,
    encode_pong,
    encode_set_limits,
    encode_set_targets,
    encode_state,
    position_from_wire,
    position_to_wire,
)
from neurogrip.hal.servo.base import ServoCalibration, ServoLimits
from neurogrip.hal.servo.emulator import Esp32Emulator
from neurogrip.hal.servo.esp32 import Esp32ServoBus
from neurogrip.hal.servo.simulated import ContactModel, SimulatedServoBus
from neurogrip.hal.transport.framing import FrameParser, crc16_ccitt, encode_frame
from neurogrip.hal.transport.loopback import LoopbackTransport


class TestFraming:
    def test_round_trip(self):
        parser = FrameParser()
        frames = list(parser.feed(encode_frame(0x02, 7, b"payload")))
        assert len(frames) == 1
        assert frames[0].msg_id == 0x02
        assert frames[0].sequence == 7
        assert frames[0].payload == b"payload"

    def test_split_across_reads(self):
        parser = FrameParser()
        data = encode_frame(0x02, 1, b"abcdef")
        assert list(parser.feed(data[:4])) == []
        frames = list(parser.feed(data[4:]))
        assert len(frames) == 1

    def test_several_frames_in_one_read(self):
        parser = FrameParser()
        data = encode_frame(1, 1, b"a") + encode_frame(2, 2, b"bb") + encode_frame(3, 3, b"")
        assert len(list(parser.feed(data))) == 3

    def test_resynchronises_after_leading_garbage(self):
        parser = FrameParser()
        frames = list(parser.feed(b"\x00\xff\x12noise" + encode_frame(5, 5, b"ok")))
        assert len(frames) == 1
        assert parser.resyncs >= 1

    def test_corrupt_frame_is_dropped_and_the_link_recovers(self):
        parser = FrameParser()
        good = encode_frame(1, 1, b"hello")
        corrupt = bytearray(encode_frame(2, 2, b"world"))
        corrupt[6] ^= 0xFF  # flip a payload bit
        frames = list(parser.feed(bytes(corrupt) + good))
        assert parser.crc_errors == 1
        assert len(frames) == 1
        assert frames[0].payload == b"hello"

    def test_crc_matches_the_known_vector(self):
        # CRC-16/CCITT-FALSE of "123456789" is 0x29B1. The firmware uses the
        # same loop, so this pins both implementations.
        assert crc16_ccitt(b"123456789") == 0x29B1

    def test_rejects_an_oversized_payload(self):
        with pytest.raises(ValueError):
            encode_frame(1, 1, b"x" * 300)

    def test_buffer_cannot_grow_without_bound(self):
        parser = FrameParser(max_buffer=128)
        for _ in range(20):
            list(parser.feed(b"\xa5" * 64))
        assert parser.pending_bytes <= 128


class TestProtocol:
    def test_position_quantisation_round_trips(self):
        for value in (0.0, 0.25, 0.5, 0.75, 1.0):
            assert position_from_wire(position_to_wire(value)) == pytest.approx(value, abs=1e-4)

    def test_positions_are_clamped_on_the_wire(self):
        assert position_to_wire(-1.0) == 0
        assert position_to_wire(2.0) == 10_000

    def test_set_targets_round_trips(self):
        pose = HandPose((0.1, 0.25, 0.5, 0.75, 0.9))
        decoded, speed, _flags = decode_set_targets(encode_set_targets(pose, speed_scale=1.5))
        assert decoded.values == pytest.approx(pose.values, abs=1e-3)
        assert speed == pytest.approx(1.5, abs=0.02)

    def test_set_limits_round_trips(self):
        payload = encode_set_limits(
            max_velocity=2.5, max_acceleration=9.0, max_current_ma=850, max_temperature_c=65
        )
        velocity, accel, current, temperature = decode_set_limits(payload)
        assert velocity == pytest.approx(2.5)
        assert accel == pytest.approx(9.0)
        assert current == 850
        assert temperature == 65

    def test_state_round_trips(self):
        telemetry = tuple(
            FingerTelemetry(
                finger=Finger(i), position=i / 10, target=i / 8, current_ma=100 + i, temperature_c=30 + i
            )
            for i in range(5)
        )
        message = decode_state(
            encode_state(
                sequence=42,
                flags=StatusFlags.ENABLED | StatusFlags.MOVING,
                fingers=telemetry,
                bus_voltage_v=7.4,
                uptime_ms=123456,
            )
        )
        assert message.sequence == 42
        assert message.enabled and message.moving and not message.estop_active
        assert message.bus_voltage_v == pytest.approx(7.4, abs=1e-3)
        assert message.fingers[3].current_ma == 103

    def test_short_payloads_are_rejected(self):
        with pytest.raises(ProtocolError):
            decode_state(b"\x00\x01")

    def test_clear_estop_carries_a_magic_word(self):
        # A corrupted frame must not be able to re-enable drive by accident.
        assert encode_clear_estop() == (0x5EA1).to_bytes(2, "little")

    def test_event_and_error_round_trip(self):
        code, finger, detail = decode_event(encode_event(EventCode.CONTACT_DETECTED, 2, 99))
        assert code is EventCode.CONTACT_DETECTED and finger == 2 and detail == 99
        assert decode_error(encode_error(ErrorCode.ESTOP_ACTIVE, 7)) == (ErrorCode.ESTOP_ACTIVE, 7)

    def test_pong_round_trip(self):
        token, version, uptime = decode_pong(encode_pong(0xC0FFEE, (1, 2, 3), 5000))
        assert token == 0xC0FFEE and version == (1, 2, 3) and uptime == 5000


class TestServoCalibration:
    def test_pulse_mapping_covers_the_configured_range(self):
        calibration = ServoCalibration(Finger.INDEX, min_pulse_us=1000, max_pulse_us=2000)
        assert calibration.to_pulse(0.0) == 1000
        assert calibration.to_pulse(1.0) == 2000

    def test_inversion_flips_the_mapping(self):
        calibration = ServoCalibration(Finger.INDEX, 1000, 2000, inverted=True)
        assert calibration.to_pulse(0.0) == 2000
        assert calibration.to_pulse(1.0) == 1000

    def test_slack_offsets_the_zero_point_and_round_trips(self):
        calibration = ServoCalibration(Finger.INDEX, 1000, 2000, slack=0.2)
        assert calibration.to_pulse(0.0) == 1200
        assert calibration.from_pulse(calibration.to_pulse(0.6)) == pytest.approx(0.6, abs=1e-3)

    def test_limits_validation_rejects_nonsense(self):
        from neurogrip.core.errors import ConfigurationError

        with pytest.raises(ConfigurationError):
            ServoLimits(max_velocity=-1).validate()
        with pytest.raises(ConfigurationError):
            ServoLimits(max_force=2.0).validate()


class TestSimulatedServoBus:
    def test_moves_towards_the_target(self, clock, servo_bus):
        servo_bus.write_targets(HandPose.uniform(0.7))
        for _ in range(300):
            clock.advance(0.005)
            state = servo_bus.read_state()
        assert state.pose.is_close(HandPose.uniform(0.7), 0.05)

    def test_respects_the_velocity_limit(self, clock):
        bus = SimulatedServoBus(clock, limits=ServoLimits(max_velocity=0.5))
        bus.open()
        bus.enable()
        bus.write_targets(HandPose.closed_hand())
        peak = 0.0
        for _ in range(400):
            clock.advance(0.005)
            state = bus.read_state()
            peak = max(peak, max(abs(f.velocity) for f in state.fingers))
        assert peak <= 0.55

    def test_contact_stops_the_fingers_and_raises_current(self, clock, servo_bus):
        servo_bus.set_contact(ContactModel.uniform(0.4, stiffness=1.5))
        servo_bus.write_targets(HandPose.uniform(0.9), force=0.7)
        for _ in range(400):
            clock.advance(0.005)
            state = servo_bus.read_state()
        assert max(state.pose) < 0.5
        assert state.total_current_ma > 300
        assert state.any_stalled

    def test_disabled_fingers_relax(self, clock, servo_bus):
        servo_bus.write_targets(HandPose.uniform(0.6))
        for _ in range(300):
            clock.advance(0.005)
            servo_bus.read_state()
        servo_bus.disable()
        for _ in range(400):
            clock.advance(0.005)
            state = servo_bus.read_state()
        assert max(state.pose) < 0.2

    def test_emergency_stop_freezes_and_ignores_commands(self, clock, servo_bus):
        servo_bus.write_targets(HandPose.uniform(0.5))
        for _ in range(100):
            clock.advance(0.005)
            servo_bus.read_state()
        servo_bus.emergency_stop()
        frozen = servo_bus.read_state().pose
        servo_bus.write_targets(HandPose.closed_hand())
        for _ in range(200):
            clock.advance(0.005)
            state = servo_bus.read_state()
        assert state.estop
        assert state.pose.max_difference(frozen) < 0.35  # relaxing, not driving closed


class TestEsp32DriverAgainstTheEmulator:
    """The production driver, the real protocol and the real framing — end to end."""

    def _link(self, clock, **transport_kwargs):
        plant = SimulatedServoBus(clock)
        emulator = Esp32Emulator(clock, plant=plant)
        transport = LoopbackTransport(emulator, clock, **transport_kwargs)
        bus = Esp32ServoBus(transport, clock)
        bus.open()
        return bus, emulator, plant, transport

    def test_state_flows_from_the_emulator_to_the_driver(self, clock):
        bus, _emulator, _plant, _transport = self._link(clock)
        bus.enable()
        for _ in range(50):
            clock.advance(0.005)
            state = bus.read_state()
        assert state.comms_ok
        assert bus.link_stats()["frames"] > 0
        assert bus.link_stats()["crc_errors"] == 0

    def test_targets_reach_the_plant(self, clock):
        bus, _emulator, _plant, _transport = self._link(clock)
        bus.enable()
        bus.home()
        for _ in range(400):
            clock.advance(0.005)
            bus.write_targets(HandPose.uniform(0.6))
            state = bus.read_state()
        assert state.pose.is_close(HandPose.uniform(0.6), 0.08)

    def test_firmware_watchdog_safes_the_hand_when_the_host_goes_quiet(self, clock):
        """The most important firmware behaviour: a host crash must not leave the
        hand squeezing whatever it was holding."""
        bus, _emulator, _plant, _transport = self._link(clock)
        bus.enable()
        for _ in range(50):
            clock.advance(0.005)
            bus.write_targets(HandPose.uniform(0.5))
            bus.read_state()
        assert bus.read_state().enabled

        for _ in range(200):  # the host stops issuing commands
            clock.advance(0.005)
            state = bus.read_state()
        assert not state.enabled
        assert "firmware_watchdog" in state.faults

    def test_telemetry_going_silent_marks_the_link_stale(self, clock):
        bus, _emulator, _plant, transport = self._link(clock)
        bus.enable()
        for _ in range(50):
            clock.advance(0.005)
            bus.read_state()
        transport.close()
        clock.advance(1.0)
        state = bus.read_state()
        assert not state.comms_ok
        assert "telemetry_stale" in state.faults

    def test_a_lossy_link_produces_crc_errors_but_stays_usable(self, clock):
        bus, _emulator, _plant, _transport = self._link(clock, corrupt_probability=0.02)
        bus.enable()
        for _ in range(400):
            clock.advance(0.005)
            bus.write_targets(HandPose.uniform(0.4))
            bus.read_state()
        stats = bus.link_stats()
        assert stats["crc_errors"] > 0  # the model really did corrupt bytes
        assert stats["frames"] > 0  # and the parser still recovered frames

    def test_estop_latches_locally_even_if_the_link_is_dead(self, clock):
        bus, _emulator, _plant, transport = self._link(clock)
        bus.enable()
        transport.close()
        bus.emergency_stop()  # must not raise
        bus.write_targets(HandPose.closed_hand())  # must be a no-op
        assert bus.read_state().estop
