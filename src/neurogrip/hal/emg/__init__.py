"""EMG acquisition drivers."""

from __future__ import annotations

from .base import EmgChannelSpec, EmgSample, EmgSource, EmgSourceStats
from .replay import ReplayEmgSource, load_recording
from .serial_source import SerialEmgSource
from .simulated import DEFAULT_CHANNELS, SimulatedEmgSource

__all__ = [
    "DEFAULT_CHANNELS",
    "EmgChannelSpec",
    "EmgSample",
    "EmgSource",
    "EmgSourceStats",
    "ReplayEmgSource",
    "SerialEmgSource",
    "SimulatedEmgSource",
    "load_recording",
]
