"""Byte-stream transport abstraction.

A :class:`Transport` is anything that moves bytes: a USB serial port, a UART, a
TCP socket to a remote test rig, or an in-process loopback to the firmware
emulator. The servo driver above it only speaks frames, so migrating the motor
controller from USB-serial to CAN or BLE means adding one file here.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..base import DeviceInfo

__all__ = ["Transport"]


@runtime_checkable
class Transport(Protocol):
    """Non-blocking byte-stream link."""

    def open(self) -> None:
        """Open the link. Raises ``DeviceNotAvailableError`` when unavailable."""
        ...

    def close(self) -> None:
        """Close the link. Idempotent, never raises."""
        ...

    @property
    def is_open(self) -> bool:
        ...

    def write(self, data: bytes) -> int:
        """Write ``data``; returns bytes accepted. Must not block indefinitely."""
        ...

    def read(self, max_bytes: int = 4096) -> bytes:
        """Read up to ``max_bytes``; returns ``b""`` when nothing is available."""
        ...

    def info(self) -> DeviceInfo:
        ...
