"""Shared pytest fixtures.

Everything here builds on :class:`~neurogrip.core.clock.SimulatedClock`, so tests
are deterministic and run far faster than real time. No test sleeps.
"""

from __future__ import annotations

import logging

import pytest

from neurogrip.core.clock import SimulatedClock
from neurogrip.core.config import ConfigLoader
from neurogrip.core.events import EventBus
from neurogrip.emg.calibration import ChannelCalibration, EmgCalibration
from neurogrip.hal.emg.simulated import DEFAULT_CHANNELS, SimulatedEmgSource
from neurogrip.hal.servo.simulated import SimulatedServoBus


@pytest.fixture(autouse=True)
def _quiet_logging():
    """Keep test output readable; tests assert on behaviour, not on log lines."""
    logging.disable(logging.WARNING)
    yield
    logging.disable(logging.NOTSET)


@pytest.fixture
def clock() -> SimulatedClock:
    return SimulatedClock()


@pytest.fixture
def bus(clock: SimulatedClock) -> EventBus:
    return EventBus(clock)


@pytest.fixture
def config():
    """Minimal in-memory configuration."""
    return ConfigLoader().add_mapping({"hardware": {"simulate": True}}).build()


@pytest.fixture
def emg_source(clock: SimulatedClock) -> SimulatedEmgSource:
    source = SimulatedEmgSource(clock)
    source.open()
    return source


@pytest.fixture
def servo_bus(clock: SimulatedClock) -> SimulatedServoBus:
    bus = SimulatedServoBus(clock)
    bus.open()
    bus.enable()
    return bus


@pytest.fixture
def calibration() -> EmgCalibration:
    """A realistic calibration matching the simulated source's signal levels."""
    cal = EmgCalibration(subject="test")
    for spec in DEFAULT_CHANNELS:
        cal.set(
            ChannelCalibration(
                channel=spec.index,
                name=spec.name,
                role=spec.role,
                rest_mean=1.0e-5,
                rest_std=2.0e-6,
                mvc=9.0e-4,
                full_scale=5.4e-4,
                samples=5000,
            )
        )
    return cal
