"""Byte framing and integrity checking for the NeuroGrip Protocol.

Frame layout (little-endian throughout)::

    +------+------+--------+--------+-----+---------+--------+
    | 0xA5 | 0x5A | LENGTH | MSG_ID | SEQ | PAYLOAD | CRC16  |
    +------+------+--------+--------+-----+---------+--------+
      sync   sync    u8       u8      u8   LENGTH B    u16

``LENGTH`` counts payload bytes only. The CRC (CCITT-FALSE, init 0xFFFF) covers
``LENGTH``, ``MSG_ID``, ``SEQ`` and ``PAYLOAD`` — everything except the sync
pattern, which is not part of the message.

The parser is incremental and resynchronising: a serial line that comes up mid
frame, drops bytes, or receives ESP32 boot-loader chatter must recover on its own
without a reset. Every discarded byte is counted so the diagnostics screen can
show link quality rather than just "it works / it doesn't".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

__all__ = ["MAX_PAYLOAD", "SYNC", "Frame", "FrameParser", "crc16_ccitt", "encode_frame"]

SYNC = b"\xa5\x5a"
MAX_PAYLOAD = 255
_HEADER_SIZE = 3  # length, msg_id, seq
_CRC_SIZE = 2


def crc16_ccitt(data: bytes, seed: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE.

    Bitwise rather than table-driven: frames are at most ~40 bytes at 200 Hz, so
    the cost is negligible, and the firmware uses the identical loop, which makes
    the two implementations trivially comparable during bring-up.
    """
    crc = seed
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


@dataclass(frozen=True, slots=True)
class Frame:
    """A decoded frame."""

    msg_id: int
    sequence: int
    payload: bytes


def encode_frame(msg_id: int, sequence: int, payload: bytes = b"") -> bytes:
    """Serialise one frame."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too long: {len(payload)} > {MAX_PAYLOAD}")
    body = bytes((len(payload), msg_id & 0xFF, sequence & 0xFF)) + payload
    crc = crc16_ccitt(body)
    return SYNC + body + bytes((crc & 0xFF, (crc >> 8) & 0xFF))


class FrameParser:
    """Incremental frame decoder tolerant of noise and partial reads."""

    __slots__ = ("_buffer", "_max_buffer", "bytes_dropped", "crc_errors", "frames_decoded", "resyncs")

    def __init__(self, max_buffer: int = 4096) -> None:
        self._buffer = bytearray()
        self._max_buffer = max_buffer
        #: Diagnostics counters — exposed on the Diagnostics ▸ Communication panel.
        self.resyncs = 0
        self.crc_errors = 0
        self.frames_decoded = 0
        self.bytes_dropped = 0

    def feed(self, data: bytes) -> Iterator[Frame]:
        """Append ``data`` and yield every complete frame it produced."""
        if not data:
            return
        self._buffer.extend(data)
        if len(self._buffer) > self._max_buffer:
            # Runaway garbage: keep only the tail that could still hold a frame.
            excess = len(self._buffer) - self._max_buffer
            del self._buffer[:excess]
            self.bytes_dropped += excess
            self.resyncs += 1
        yield from self._drain()

    def _drain(self) -> Iterator[Frame]:
        while True:
            start = self._buffer.find(SYNC)
            if start < 0:
                # Keep the last byte: it may be the first half of a split sync word.
                keep = 1 if self._buffer.endswith(SYNC[:1]) else 0
                dropped = len(self._buffer) - keep
                if dropped > 0:
                    self.bytes_dropped += dropped
                    del self._buffer[:dropped]
                return
            if start > 0:
                self.bytes_dropped += start
                self.resyncs += 1
                del self._buffer[:start]

            if len(self._buffer) < len(SYNC) + _HEADER_SIZE:
                return

            length = self._buffer[len(SYNC)]
            total = len(SYNC) + _HEADER_SIZE + length + _CRC_SIZE
            if len(self._buffer) < total:
                return

            body = bytes(self._buffer[len(SYNC) : len(SYNC) + _HEADER_SIZE + length])
            crc_lo = self._buffer[total - 2]
            crc_hi = self._buffer[total - 1]
            received = crc_lo | (crc_hi << 8)

            if crc16_ccitt(body) != received:
                # Corrupt frame: drop the sync word and rescan from the next byte,
                # in case the "sync" was actually payload data.
                self.crc_errors += 1
                self.bytes_dropped += len(SYNC)
                del self._buffer[: len(SYNC)]
                continue

            del self._buffer[:total]
            self.frames_decoded += 1
            yield Frame(msg_id=body[1], sequence=body[2], payload=body[_HEADER_SIZE:])

    def reset(self) -> None:
        """Discard buffered bytes (called when the link is reopened)."""
        self._buffer.clear()

    @property
    def pending_bytes(self) -> int:
        return len(self._buffer)

    def stats(self) -> dict[str, int]:
        return {
            "frames": self.frames_decoded,
            "crc_errors": self.crc_errors,
            "resyncs": self.resyncs,
            "bytes_dropped": self.bytes_dropped,
            "pending": len(self._buffer),
        }
