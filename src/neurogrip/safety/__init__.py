"""Safety layer.

Three cooperating mechanisms, at three timescales:

* :mod:`~neurogrip.safety.watchdog` — *"is everything still running?"* Named
  timeouts on the control loop, EMG, servo telemetry, vision and the UI.
* :mod:`~neurogrip.safety.rules` — *"is everything within limits?"* Independent
  predicates over grip force, current, temperature, communication, battery,
  sensor quality and servo tracking.
* :mod:`~neurogrip.safety.integrity` — *"does the stop still work?"* Periodically
  rehearses the e-stop signalling path and, when the hand is idle, proves the
  hardware path by actually cutting drive and watching it happen.
* :mod:`~neurogrip.safety.monitor` — folds these into a single
  :class:`~neurogrip.safety.monitor.SafetyState` that gates motion and AI
  assistance, and engages the latching
  :class:`~neurogrip.safety.estop.EmergencyStop` on critical faults.

The response ladder:

===========  ==========================================================
severity     response
===========  ==========================================================
MINOR        logged
DEGRADED     the affected feature is disabled; the hand works normally
FALLBACK     AI assistance off; direct manual control continues
CRITICAL     emergency stop, latched until a human acknowledges
===========  ==========================================================

Note that only ``CRITICAL`` stops the hand. Everything short of that keeps the
user in control of their own limb — which is the point.

There is a fourth mechanism outside this package and outside this process: the
ESP32 firmware's own command-stream watchdog. If the host stops talking for any
reason at all, including a crash that prevents any of the above from running, the
firmware safes the actuators by itself.
"""

from __future__ import annotations

from .estop import EmergencyStop, EstopRecord, EstopSource
from .integrity import EstopIntegrityRule, EstopSelfCheck, IntegrityStatus
from .monitor import SafetyMonitor, SafetyState
from .rules import DEFAULT_RULES, Fault, SafetyContext, SafetyRule
from .watchdog import Watchdog, WatchdogExpiry, WatchdogGroup

__all__ = [
    "DEFAULT_RULES",
    "EmergencyStop",
    "EstopIntegrityRule",
    "EstopRecord",
    "EstopSelfCheck",
    "EstopSource",
    "Fault",
    "IntegrityStatus",
    "SafetyContext",
    "SafetyMonitor",
    "SafetyRule",
    "SafetyState",
    "Watchdog",
    "WatchdogExpiry",
    "WatchdogGroup",
]
