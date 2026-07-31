# API reference

The interfaces you implement to change hardware, swap a model, or embed the
system in something else. Everything here is a `Protocol` — structural, not
inherited — so an implementation needs no import from us beyond the types it
exchanges.

Full signatures live in the source; this is the map and the contracts that are
not expressible in a type.

---

## Contents

- [Composition root](#composition-root)
- [Hardware abstraction](#hardware-abstraction) — servo bus, EMG, camera, transport
- [Vision backends](#vision-backends)
- [Grasp planners](#grasp-planners)
- [Services and lifecycle](#services-and-lifecycle)
- [Events](#events)
- [Errors](#errors)
- [Clock](#clock)
- [Extending safety](#extending-safety)

---

## Composition root

```python
from neurogrip.runtime.bootstrap import load_configuration
from neurogrip.runtime.application import build_application

config = load_configuration(profile="simulation")
app = build_application(config)
app.start(allow_motion=True)
try:
    app.run()            # blocks; installs signal handlers
finally:
    app.stop()
```

`build_application` is the only place collaborators are constructed. Read
`runtime/application.py` to see how the system fits together — there is nowhere
else for wiring to hide.

For stepping manually (tests, embedding):

```python
app.scheduler.step()     # run whichever rate groups are due
app.clock.advance(0.005)
```

Useful members: `app.controller`, `app.emg`, `app.vision`, `app.fusion`,
`app.modes`, `app.safety`, `app.diagnostics`, `app.training`,
`app.servo_calibration`, `app.profiles`, `app.run_marker`.

---

## Hardware abstraction

### `ServoBus`

`neurogrip.hal.servo.base.ServoBus` — the five finger actuators as one bus.

```python
def open() -> None
def close() -> None
is_open: bool
def info() -> DeviceInfo
def enable(mask: int = ALL_FINGERS_MASK) -> None
def disable(mask: int = ALL_FINGERS_MASK) -> None
def set_limits(limits: ServoLimits) -> None
def set_calibration(calibration: ServoCalibration) -> None
def write_targets(pose: HandPose, *, speed_scale: float, force: float) -> None
def read_state() -> ServoBusState
def home() -> None
def emergency_stop() -> None
def clear_emergency_stop() -> None
```

Contracts a type cannot express:

- **`read_state` must never block and never raise.** A control loop that a serial
  read can knock over is not a control loop you want on a limb. Report transport
  failure as `comms_ok=False`; the safety layer handles it.
- **`emergency_stop` is callable from any thread**, including a signal handler,
  and must not raise. It must take effect without waiting for the control loop.
- **Everything else is single-threaded**, called only from the control group.
- Addressed as one bus, not five servos, because a grasp is a coordinated motion:
  one atomic five-target command per cycle keeps the fingers synchronised and
  gives the firmware one unambiguous watchdog to refresh.

Implementations: `SimulatedServoBus`, `Esp32ServoBus`, `Esp32Emulator`.

### `EmgSource`

`neurogrip.hal.emg.base.EmgSource`

```python
def open() / close() / is_open
channels: tuple[EmgChannelSpec, ...]
sample_rate_hz: float
def read() -> list[EmgSample]      # non-blocking; [] when nothing is ready
def info() -> DeviceInfo
```

Return raw volts. Do not filter, do not normalise — the pipeline owns that, and a
source that pre-processes makes the calibration wrong in a way that is very hard
to find.

Implementations: `SimulatedEmgSource`, `SerialEmgSource`, `ReplayEmgSource`.

### `Camera`

`neurogrip.hal.camera.base.Camera`

```python
def open() / close() / is_open
settings: CameraSettings
def read() -> Frame | None         # None when no new frame is ready
def info() -> DeviceInfo
```

Implementations: `SimulatedCamera`, `OpenCvCamera`.

### `Transport`

`neurogrip.hal.transport.base.Transport` — a byte stream. Implement this to move
the motor controller onto CAN, BLE, or a TCP link to a test rig; the driver above
only speaks frames.

```python
def open() / close() / is_open
def write(data: bytes) -> int
def read(max_bytes: int = 4096) -> bytes    # b"" when nothing available
def info() -> DeviceInfo
```

Wrap any transport in `ReconnectingTransport` to get supervised recovery:

```python
from neurogrip.hal.transport.reconnecting import ReconnectingTransport

link = ReconnectingTransport(my_transport, clock, on_reconnect=driver.resync)
```

`on_reconnect` fires after a successful reopen so the driver can replay state the
far end lost. Reconnection never re-energises the actuators — coming back from a
disconnect is not evidence that moving is safe.

---

## Vision backends

`neurogrip.vision.backend.VisionBackend`

```python
def initialize() -> None           # may raise ModelLoadError
def shutdown() -> None             # idempotent, must not raise
def info() -> BackendInfo
capabilities: VisionCapability     # a Flag, not a class check
def process(frame: Frame) -> VisionResult
```

**`process` must not raise for ordinary inference failure.** Return a
`VisionResult` with `error` set. Vision is an assistive input; a model exception
must never reach the control loop.

Declare capabilities honestly. A backend that advertises `DETECTION` without a
classification head makes fusion wait for a label that never arrives.

```python
from neurogrip.vision.backend import register_backend

def _build(config):
    return MyBackend(MySettings.from_config(config))

register_backend("my_backend", _build)     # at import time
```

Then set `vision.backend = "my_backend"`.

Bundled: `hggd_mcu` (default), `onnx_detector`, `anygrasp`, `replay`, `mock`,
`null`.

### Recording and replay

```python
from neurogrip.vision.backends.replay import VisionRecorder, load_recording

pipeline.recorder = VisionRecorder("var/rec.jsonl", backend="hggd_mcu")
results = load_recording("data/vision/reference-bottle.jsonl")
```

---

## Grasp planners

`neurogrip.ai.grasp.base.GraspPlanner`

```python
name: str
def plan(context: GraspContext) -> GraspPlan | None
```

**Return `None` rather than a poor plan.** The composite chain falls through to
the next planner and finally to a conservative default. A planner that always
produces something removes that safety net.

A planner decides *how* to grasp. It never decides *whether* — that comes from
EMG intent and is enforced in `neurogrip.fusion`, above every planner.

```python
from neurogrip.ai.grasp import register_planner

register_planner("mine", lambda grips, affordances, kinematics: MyPlanner(...))
```

Then list it in `[ai] planners`, in priority order:

```toml
[ai]
planners = ["mine", "hggd", "heuristic"]
```

Bundled: `hggd` (planar candidates), `anygrasp` (6-DoF poses), `heuristic`
(affordance table, no model required).

`GraspContext` carries `intent`, `vision`, `current_pose`, `mode`,
`force_ceiling`, `speed_ceiling`, `holding`. It is one immutable object rather
than an argument list so adding a signal — an IMU, a second camera — does not
change every planner's signature.

### A note for 6-DoF planners

This hand has **no powered wrist**. Approach direction is whatever the user's arm
is doing, and software cannot change it. A planner that assumes it can place the
manipulator in any pose it proposes will pre-shape for a geometry the hand will
never be in. `AnyGraspPlanner` shows the handling: reject candidates whose
approach is far from where the hand points, attenuate confidence smoothly in
between, and return `None` rather than plan for an unreachable pose.

---

## Services and lifecycle

`neurogrip.core.lifecycle.Service`

```python
name: str
def start() -> None
def stop() -> None
running: bool
def health() -> HealthReport
```

Subclass `ServiceBase` for the bookkeeping. Register with
`app.services.add(service)`; start order is registration order and stop is the
reverse.

`health()` must be cheap — it is polled by diagnostics — and must never raise.

---

## Events

`neurogrip.core.events.EventBus`, topics in `neurogrip.core.topics.Topics`.

```python
subscription = bus.subscribe("hand.*", on_hand_event)
bus.publish(Topics.MOTION_CANCELLED, {"reason": "user"}, source="my-module")
subscription.cancel()
```

Wildcards: exact (`"a.b"`), prefix (`"a.*"`), global (`"*"`).

A handler that raises does not reach the publisher, and one that raises
repeatedly is quarantined. Publishing is synchronous — do not block in a handler.

The bus is for observation. Anything that must happen for the system to be safe
goes through a direct call, not a subscription, because a quarantined handler is
a silently missing behaviour.

---

## Errors

`neurogrip.core.errors`, all deriving from `NeuroGripError` and carrying a
`Severity`:

| Severity | Meaning |
|---|---|
| `MINOR` | Logged, no behaviour change |
| `DEGRADED` | A capability is lost; assistance is reduced |
| `FALLBACK` | Drop to Manual mode |
| `CRITICAL` | Emergency stop |

`ConfigurationError`, `DeviceError`, `DeviceNotAvailableError`,
`CommunicationError`, `ProtocolError`, `CalibrationError`, `VisionError`,
`ModelLoadError`, `PlanningError`, `ModeTransitionError`, `SafetyViolation`,
`EmergencyStopActive`.

Severity is what the safety layer acts on, so choose it deliberately: raising
`CRITICAL` for a missing model file stops the hand for something the user could
have worked around.

---

## Clock

`neurogrip.core.clock.Clock`

```python
def monotonic() -> float
def wall_time() -> float
def sleep(seconds: float) -> None
```

`RealClock` in production, `SimulatedClock` in tests and scenarios.

**Never call `time.monotonic()` directly.** Take a `Clock`. Code that reads the
real clock passes its own tests and makes everyone else's flaky.

**`0.0` is a valid timestamp.** A `SimulatedClock` starts at zero, so `0.0`
cannot double as "never set" — use `None`. This has caused four separate bugs in
this codebase; see [development.md](development.md).

---

## Extending safety

`neurogrip.safety.rules.SafetyRule`

```python
name: str
def evaluate(context: SafetyContext) -> SafetyFinding | None
```

Return `None` when there is nothing to report. Rules are evaluated every decision
cycle and must be pure and fast — no I/O, no allocation-heavy work.

```python
app.safety.add_rule(MyRule())
```

A rule that fires produces a finding with a severity, which the monitor turns
into degradation, a mode fallback, or an emergency stop. Before adding one, read
[safety.md](safety.md) — particularly the note on why rules must tolerate
absent inputs at startup, which is where several early false positives came from.
