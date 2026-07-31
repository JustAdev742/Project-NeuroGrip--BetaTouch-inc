"""In-process transport connected to a firmware emulator.

This is the piece that lets the *entire* stack — including the real framing
codec, the real protocol encoders and the real servo driver — run and be tested
with no hardware attached. The only substituted component is the physical wire.

It also models the properties that make serial links interesting: configurable
latency, byte loss and corruption. ``tests/unit/test_transport.py`` uses those to
prove the parser resynchronises and the driver reports a comms fault rather than
acting on garbage.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Protocol

from ...core.clock import Clock, RealClock
from ..base import DeviceCapability, DeviceInfo, DeviceKind

__all__ = ["EmulatedDevice", "LoopbackTransport"]


class EmulatedDevice(Protocol):
    """The MCU side of a loopback link."""

    def on_host_bytes(self, data: bytes) -> None:
        """Called with bytes the host wrote."""
        ...

    def poll(self, now: float) -> bytes:
        """Return bytes the device wishes to send (may be empty)."""
        ...

    def reset(self) -> None:
        """Reset device state, as if power-cycled."""
        ...


class LoopbackTransport:
    """Connects the host driver to an :class:`EmulatedDevice` in the same process."""

    def __init__(
        self,
        device: EmulatedDevice,
        clock: Clock | None = None,
        *,
        latency: float = 0.0,
        drop_probability: float = 0.0,
        corrupt_probability: float = 0.0,
        seed: int = 12345,
    ) -> None:
        self._device = device
        self._clock = clock or RealClock()
        self._latency = max(0.0, latency)
        self._drop = drop_probability
        self._corrupt = corrupt_probability
        self._random = random.Random(seed)
        self._inbound: deque[tuple[float, int]] = deque()
        self._open = False
        self.bytes_written = 0
        self.bytes_read = 0

    # -- Transport protocol ---------------------------------------------------

    def open(self) -> None:
        self._device.reset()
        self._inbound.clear()
        self._open = True

    def close(self) -> None:
        self._open = False
        self._inbound.clear()

    @property
    def is_open(self) -> bool:
        return self._open

    def write(self, data: bytes) -> int:
        if not self._open:
            return 0
        self.bytes_written += len(data)
        self._device.on_host_bytes(self._degrade(data))
        return len(data)

    def read(self, max_bytes: int = 4096) -> bytes:
        if not self._open:
            return b""
        now = self._clock.monotonic()
        produced = self._device.poll(now)
        if produced:
            deliver_at = now + self._latency
            for byte in self._degrade(produced):
                self._inbound.append((deliver_at, byte))

        out = bytearray()
        while self._inbound and len(out) < max_bytes and self._inbound[0][0] <= now:
            out.append(self._inbound.popleft()[1])
        self.bytes_read += len(out)
        return bytes(out)

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="loopback",
            kind=DeviceKind.TRANSPORT,
            driver="loopback",
            connection=f"in-process:{type(self._device).__name__}",
            capabilities=frozenset({DeviceCapability.SIMULATED}),
            extra={
                "latency_s": self._latency,
                "drop_probability": self._drop,
                "corrupt_probability": self._corrupt,
            },
        )

    # -- link degradation -----------------------------------------------------

    def _degrade(self, data: bytes) -> bytes:
        """Apply the configured loss/corruption model to a byte string."""
        if self._drop <= 0 and self._corrupt <= 0:
            return data
        out = bytearray()
        for byte in data:
            if self._drop > 0 and self._random.random() < self._drop:
                continue
            if self._corrupt > 0 and self._random.random() < self._corrupt:
                byte ^= 1 << self._random.randrange(8)
            out.append(byte)
        return bytes(out)

    def set_link_quality(self, *, drop: float = 0.0, corrupt: float = 0.0, latency: float = 0.0) -> None:
        """Reconfigure the degradation model at runtime (used by fault-injection tests)."""
        self._drop = drop
        self._corrupt = corrupt
        self._latency = max(0.0, latency)
