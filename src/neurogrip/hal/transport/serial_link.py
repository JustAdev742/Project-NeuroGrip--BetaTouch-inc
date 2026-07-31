"""Serial-port transport (``pyserial``).

``pyserial`` is imported lazily inside :meth:`SerialTransport.open` rather than at
module import time. That is deliberate: the module must be importable on a
development laptop and in CI where the hardware extra is not installed, so that
``neurogrip diagnose`` can still *report* "serial driver not installed" instead of
crashing on import.
"""

from __future__ import annotations

from typing import Any

from ...core.errors import CommunicationError, DeviceNotAvailableError
from ...core.logging import get_logger
from ..base import DeviceInfo, DeviceKind

__all__ = ["SerialTransport", "list_serial_ports"]

log = get_logger(__name__)


def list_serial_ports() -> list[dict[str, str]]:
    """Enumerate candidate serial ports for the Settings ▸ Connection screen.

    Returns an empty list (not an error) when ``pyserial`` is unavailable, because
    port discovery is a convenience feature, not a functional requirement.
    """
    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]
    except ImportError:
        return []
    return [
        {
            "device": port.device,
            "description": port.description or "",
            "hwid": port.hwid or "",
            "manufacturer": getattr(port, "manufacturer", "") or "",
        }
        for port in list_ports.comports()
    ]


class SerialTransport:
    """Byte-stream transport over a UART / USB-CDC port."""

    def __init__(
        self,
        port: str,
        *,
        baudrate: int = 921_600,
        read_timeout: float = 0.0,
        write_timeout: float = 0.05,
        rtscts: bool = False,
        dtr_reset: bool = False,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._read_timeout = read_timeout
        self._write_timeout = write_timeout
        self._rtscts = rtscts
        #: ESP32 dev boards reset when DTR/RTS toggle. Off by default so that
        #: reopening the link does not reboot the motor controller mid-grasp.
        self._dtr_reset = dtr_reset
        self._serial: Any = None

    def open(self) -> None:
        if self._serial is not None:
            return
        try:
            import serial  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DeviceNotAvailableError(
                "pyserial is not installed; install the 'hardware' extra",
                context={"port": self._port},
            ) from exc

        try:
            handle = serial.Serial()
            handle.port = self._port
            handle.baudrate = self._baudrate
            handle.timeout = self._read_timeout
            handle.write_timeout = self._write_timeout
            handle.rtscts = self._rtscts
            if not self._dtr_reset:
                handle.dtr = False
                handle.rts = False
            handle.open()
            handle.reset_input_buffer()
            handle.reset_output_buffer()
        except Exception as exc:  # pyserial raises several unrelated types
            raise DeviceNotAvailableError(
                f"cannot open serial port: {exc}",
                context={"port": self._port, "baudrate": self._baudrate},
            ) from exc

        self._serial = handle
        log.info("serial port opened", port=self._port, baudrate=self._baudrate)

    def close(self) -> None:
        handle, self._serial = self._serial, None
        if handle is None:
            return
        try:
            handle.close()
        except Exception:
            log.warning("error while closing serial port", port=self._port)

    @property
    def is_open(self) -> bool:
        return self._serial is not None and bool(getattr(self._serial, "is_open", False))

    def write(self, data: bytes) -> int:
        if self._serial is None:
            raise CommunicationError("serial port is not open", context={"port": self._port})
        try:
            return int(self._serial.write(data) or 0)
        except Exception as exc:
            raise CommunicationError(
                f"serial write failed: {exc}", context={"port": self._port}
            ) from exc

    def read(self, max_bytes: int = 4096) -> bytes:
        if self._serial is None:
            raise CommunicationError("serial port is not open", context={"port": self._port})
        try:
            waiting = int(getattr(self._serial, "in_waiting", 0) or 0)
            if waiting <= 0:
                return b""
            return bytes(self._serial.read(min(waiting, max_bytes)))
        except Exception as exc:
            raise CommunicationError(
                f"serial read failed: {exc}", context={"port": self._port}
            ) from exc

    def info(self) -> DeviceInfo:
        return DeviceInfo(
            name="serial",
            kind=DeviceKind.TRANSPORT,
            driver="pyserial",
            connection=f"{self._port}@{self._baudrate}",
            extra={"rtscts": self._rtscts, "dtr_reset": self._dtr_reset},
        )
