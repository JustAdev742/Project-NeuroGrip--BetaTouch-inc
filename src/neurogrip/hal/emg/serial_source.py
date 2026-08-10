"""Serial EMG front end.

Reads a fixed-format sample stream from an acquisition board (an ADS1115/ADS1256
bridge, a Teensy, or a second ESP32). The board streams NGP frames carrying an
``EMG_SAMPLES`` payload:

    seq(u8) | channels(u8) | count(u8) | interval_us(u16) | [int16 × channels] × count

Samples are transmitted as signed 16-bit ADC counts and converted here using the
configured reference voltage and gain, so the rest of the stack sees volts.

Batching several samples per frame matters at 1 kHz: one frame per sample would
spend most of the link budget on framing overhead.
"""

from __future__ import annotations

import struct
from collections import deque
from collections.abc import Sequence

from ...core.clock import Clock, RealClock
from ...core.errors import CommunicationError
from ...core.logging import get_logger
from ..base import DeviceInfo, DeviceKind
from ..transport.base import Transport
from ..transport.framing import FrameParser
from .base import EmgChannelSpec, EmgSample, EmgSourceStats

__all__ = ["EMG_SAMPLES_MSG_ID", "SerialEmgSource"]

log = get_logger(__name__)

#: Message id used by the EMG front end within the shared NGP framing.
EMG_SAMPLES_MSG_ID = 0x90

_HEADER_FMT = "<BBBH"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


class SerialEmgSource:
    """EMG acquisition over a framed byte transport."""

    def __init__(
        self,
        transport: Transport,
        channels: Sequence[EmgChannelSpec],
        clock: Clock | None = None,
        *,
        sample_rate_hz: float = 1000.0,
        adc_reference_v: float = 4.096,
        adc_bits: int = 16,
        amplifier_gain: float = 1000.0,
        buffer_limit: int = 8192,
    ) -> None:
        self._transport = transport
        self._channels = tuple(channels)
        self._clock = clock or RealClock()
        self._rate = sample_rate_hz
        self._parser = FrameParser()
        self._buffer: deque[EmgSample] = deque(maxlen=buffer_limit)
        self._stats = EmgSourceStats()
        self._last_sequence: int | None = None
        # Volts per ADC count, referred back to the electrode.
        self._scale = adc_reference_v / (2 ** (adc_bits - 1)) / amplifier_gain

    def open(self) -> None:
        self._transport.open()
        self._parser.reset()
        self._buffer.clear()
        self._last_sequence = None

    def close(self) -> None:
        self._transport.close()

    @property
    def is_open(self) -> bool:
        return self._transport.is_open

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="emg",
            kind=DeviceKind.EMG,
            driver="serial-ngp",
            connection=self._transport.info().connection,
            extra={"channels": len(self._channels), "rate_hz": self._rate, **self._parser.stats()},
        )

    @property
    def sample_rate_hz(self) -> float:
        return self._rate

    @property
    def channels(self) -> Sequence[EmgChannelSpec]:
        return self._channels

    def dropped_samples(self) -> int:
        return self._stats.dropped

    def read(self) -> list[EmgSample]:
        """Pump the transport and return every buffered sample."""
        try:
            data = self._transport.read()
        except CommunicationError as exc:
            self._stats.note_error(str(exc))
            log.throttled(
                "emg-read", "error", "EMG transport error", now=self._clock.monotonic(), error=str(exc)
            )
            return []

        for frame in self._parser.feed(data):
            if frame.msg_id != EMG_SAMPLES_MSG_ID:
                continue
            self._decode(frame.payload)

        out = list(self._buffer)
        self._buffer.clear()
        self._stats.samples_read += len(out)
        if out:
            self._stats.last_timestamp = out[-1].timestamp
        return out

    def _decode(self, payload: bytes) -> None:
        if len(payload) < _HEADER_SIZE:
            self._stats.note_error("short EMG frame")
            return
        sequence, channel_count, count, interval_us = struct.unpack_from(_HEADER_FMT, payload, 0)

        if self._last_sequence is not None:
            gap = (sequence - self._last_sequence - 1) & 0xFF
            if gap:
                # Lost frames mean lost samples; report it rather than hide it.
                self._stats.dropped += gap * count
                log.throttled(
                    "emg-gap",
                    "warning",
                    "EMG frame gap detected",
                    now=self._clock.monotonic(),
                    frames_lost=gap,
                )
        self._last_sequence = sequence

        expected = _HEADER_SIZE + channel_count * count * 2
        if len(payload) != expected:
            self._stats.note_error(f"EMG frame length mismatch: {len(payload)} != {expected}")
            return

        interval = interval_us / 1_000_000.0
        # The last sample in the batch is the freshest; timestamp backwards from now.
        now = self._clock.monotonic()
        base = now - (count - 1) * interval
        offset = _HEADER_SIZE
        for index in range(count):
            raw = struct.unpack_from(f"<{channel_count}h", payload, offset)
            offset += channel_count * 2
            values = tuple(counts * self._scale for counts in raw[: len(self._channels)])
            if len(self._buffer) == self._buffer.maxlen:
                self._stats.dropped += 1
            self._buffer.append(EmgSample(timestamp=base + index * interval, values=values))


def encode_emg_frame(
    sequence: int, samples: Sequence[Sequence[int]], interval_us: int
) -> bytes:  # pragma: no cover - used by hardware bring-up tools
    """Encode an ``EMG_SAMPLES`` payload (mirrors the front-end firmware)."""
    if not samples:
        return b""
    channel_count = len(samples[0])
    out = bytearray(
        struct.pack(_HEADER_FMT, sequence & 0xFF, channel_count, len(samples), interval_us)
    )
    for sample in samples:
        out += struct.pack(f"<{channel_count}h", *sample)
    return bytes(out)
