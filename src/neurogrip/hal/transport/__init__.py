"""Byte-stream transports for the motor-controller link."""

from __future__ import annotations

from .base import Transport
from .framing import MAX_PAYLOAD, SYNC, Frame, FrameParser, crc16_ccitt, encode_frame
from .loopback import EmulatedDevice, LoopbackTransport
from .serial_link import SerialTransport, list_serial_ports

__all__ = [
    "MAX_PAYLOAD",
    "SYNC",
    "EmulatedDevice",
    "Frame",
    "FrameParser",
    "LoopbackTransport",
    "SerialTransport",
    "Transport",
    "crc16_ccitt",
    "encode_frame",
    "list_serial_ports",
]
