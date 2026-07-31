"""Hardware abstraction layer.

Rule enforced by this package boundary: **no vendor library, file descriptor or
pin number appears outside ``neurogrip.hal``.** Every peripheral is reached
through a ``Protocol`` (``ServoBus``, ``EmgSource``, ``CameraSource``,
``Transport``), and :class:`~neurogrip.hal.factory.HardwareFactory` is the single
place that maps configuration onto concrete drivers.

Practical consequence: the entire stack -- including the production ESP32 driver
and the real wire protocol -- runs against in-process simulations, which is what
makes the integration tests in ``tests/integration`` possible.
"""

from __future__ import annotations

from .base import Device, DeviceCapability, DeviceInfo, DeviceKind
from .factory import HardwareBundle, HardwareFactory

__all__ = [
    "Device",
    "DeviceCapability",
    "DeviceInfo",
    "DeviceKind",
    "HardwareBundle",
    "HardwareFactory",
]
