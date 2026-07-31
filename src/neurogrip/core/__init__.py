"""Framework primitives shared by every NeuroGrip subsystem.

Nothing in :mod:`neurogrip.core` knows about EMG, vision, servos or the UI. It
supplies the vocabulary (types), the plumbing (events, config, container,
lifecycle) and the temporal substrate (clock, rate) that the domain modules build
on. The dependency arrow points one way only: domain code imports ``core``, never
the reverse.
"""

from __future__ import annotations

from .clock import Clock, Deadline, RealClock, SimulatedClock, Stopwatch
from .config import Config, ConfigLoader, bind_dataclass, deep_merge
from .container import Container, ServiceRegistry
from .errors import NeuroGripError, Severity
from .events import Event, EventBus, QueuedSubscriber, Subscription
from .lifecycle import HealthReport, HealthStatus, Service, ServiceBase, TickResult
from .logging import StructuredLogger, configure_logging, get_logger
from .rate import LoopMonitor, LoopStats, RateTimer
from .ringbuffer import RingBuffer, RunningStats, SlidingWindow
from .state import StateChange, StateMachine, SystemState, Transition
from .topics import Topics
from .types import (
    FINGER_COUNT,
    Finger,
    FingerVector,
    GraspType,
    HandPose,
    IntentKind,
    ModeId,
    Range,
    clamp,
    lerp,
    normalise,
)

__all__ = [
    "FINGER_COUNT",
    "Clock",
    "Config",
    "ConfigLoader",
    "Container",
    "Deadline",
    "Event",
    "EventBus",
    "Finger",
    "FingerVector",
    "GraspType",
    "HandPose",
    "HealthReport",
    "HealthStatus",
    "IntentKind",
    "LoopMonitor",
    "LoopStats",
    "ModeId",
    "NeuroGripError",
    "QueuedSubscriber",
    "Range",
    "RateTimer",
    "RealClock",
    "RingBuffer",
    "RunningStats",
    "Service",
    "ServiceBase",
    "ServiceRegistry",
    "Severity",
    "SimulatedClock",
    "SlidingWindow",
    "StateChange",
    "StateMachine",
    "Stopwatch",
    "StructuredLogger",
    "Subscription",
    "SystemState",
    "TickResult",
    "Topics",
    "Transition",
    "bind_dataclass",
    "clamp",
    "configure_logging",
    "deep_merge",
    "get_logger",
    "lerp",
    "normalise",
]
