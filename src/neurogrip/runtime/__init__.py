"""Runtime: scheduling, sensor services and the composition root.

``application.py`` is the file to read first. It is the only place that
constructs collaborators, so it is a complete, honest map of the system.

Concurrency model: **one thread**. Every periodic task is a rate group on the
cooperative :class:`~neurogrip.runtime.scheduler.Scheduler`, which removes data
races between the control loop and everything else by construction. Work that
genuinely cannot be bounded — disk writes, telemetry — runs on a
:class:`~neurogrip.core.events.QueuedSubscriber` worker instead.
"""

from __future__ import annotations

from .application import Application, build_application
from .bootstrap import load_configuration
from .scheduler import RateGroup, Scheduler
from .services import EmgService, VisionService

__all__ = [
    "Application",
    "EmgService",
    "RateGroup",
    "Scheduler",
    "VisionService",
    "build_application",
    "load_configuration",
]
