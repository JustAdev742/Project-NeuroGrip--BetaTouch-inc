# Development

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

The runtime core is standard library only, so you can also just run it:

```bash
PYTHONPATH=src python3 -m neurogrip simulate all
```

## Testing

```bash
pytest                          # everything, ~70 s
pytest tests/unit -q            # fast: no I/O, no sleeping
pytest tests/integration -q     # the whole system, simulated
pytest -k fusion                # one area
```

**Tests never sleep.** Every component takes a `Clock`, so the integration suite
drives the entire application under a `SimulatedClock` far faster than real time.
If you find yourself writing `time.sleep` in a test, the component under test is
missing a clock injection.

### Layers

| | What it covers |
|---|---|
| `tests/unit` | One module at a time, with fakes for its collaborators |
| `tests/integration` | The whole application on simulated hardware |
| `neurogrip simulate` | Scenarios, also run as tests so they cannot rot |

The integration tests exercise *production* code above the HAL — real filters,
real fusion gates, the real ESP32 driver speaking the real wire protocol to an
in-process firmware emulator, real safety rules. Only the physical wire is
substituted.

### The tests that matter most

`TestTheAiNeverActsAlone` and `TestSharedControlInvariants` encode the project's
central rule. **If either ever fails, the system has stopped being a
shared-control device.** Treat a failure there as a design regression, not a test
to adjust.

## Conventions

**Type hints everywhere.** `from __future__ import annotations` at the top of
every module.

**Frozen dataclasses for values.** `HandPose`, `IntentEstimate`, `Decision`,
`GraspPlan` and friends are immutable, which is what makes it safe to pass state
snapshots between rate groups.

**Protocols for interfaces**, not ABCs. Structural typing means an implementation
does not have to import the interface it satisfies.

**Inject the clock.** Never call `time.monotonic()` directly.

**Never use `0.0` as a "not set" sentinel for a timestamp.** A simulated or
freshly booted clock legitimately reads zero. Use `None`. This bug has been
introduced — and fixed — in the watchdog, the mode manager, the training streak
counter and the contact detector; the pattern is easy to repeat.

**Comment the *why*.** The what is in the code. Comments earn their place by
explaining a trade-off, a failure mode, or a decision that looks arbitrary
without context.

**Errors carry severity.** Everything derives from `NeuroGripError` and declares
a `Severity`, which is what the safety layer maps onto an action.

## Extending

| To add… | Do this |
|---|---|
| Vision backend | Implement `VisionBackend`, call `register_backend` |
| Grasp planner | Implement `GraspPlanner`, add to `[ai] planners` |
| Servo bus | Implement `ServoBus`, add a branch to `HardwareFactory` |
| EMG front end | Implement `EmgSource` |
| Safety rule | Implement `SafetyRule`, add it to the monitor |
| Operating mode | Define a `ModeProfile`, register in `build_modes` |
| Training exercise | Implement `Exercise`, add to `EXERCISES` |
| UI screen | Write `ViewModel → Scene`, add a route |
| Object class | Add a table to `config/affordances.toml` — no code |
| Grip preset | Add a table to `config/grasps.toml` — no code |

Each of these is a single insertion point by design. If an addition requires
touching three packages, the abstraction is wrong — say so rather than working
around it.

## Debugging

```bash
neurogrip console               # interactive; `arm` unlocks the dangerous commands
neurogrip run --set logging.level=DEBUG
neurogrip diagnose              # self-tests plus a health report
neurogrip config                # the merged configuration and its sources
```

Inside the console: `status`, `intent`, `decision` (with the evidence that
produced it), `vision`, `metrics`, `topics`, `selftest`.

The **black-box recorder** writes `var/blackbox/incident-*.json` on any critical
fault, e-stop or watchdog expiry — the recent event history with full payloads.
That is usually the fastest way to understand something that happened once.

## Adding a scenario

```python
def _my_scenario() -> Scenario:
    def setup(world, app):
        world.place_object("cup", width_m=0.08)

    def check(world, app) -> tuple[bool, str]:
        return (app.controller.state.holding, "hand is holding the cup")

    return Scenario(
        name="my-scenario",
        description="What this demonstrates",
        steps=(ScenarioStep(1.0, setup, "cup enters view"),),
        duration_s=6.0,
        checks=(check,),
    )
```

Register in `DEMO_SCENARIOS` and add the name to the parametrised integration
test so it runs in CI.

## Performance

Loop timing is reported live (`neurogrip diagnose`, or the Diagnostics screen):
target vs actual rate, jitter, p95 period, overruns.

Rough budgets on a Pi 4:

| Group | Rate | Budget |
|---|---|---|
| control | 200 Hz | < 1 ms |
| emg | 200 Hz | < 2 ms |
| decision | 100 Hz | < 3 ms |
| vision | 20 Hz | < 30 ms |

Everything is on one thread, so if overruns appear, lower `vision_hz` first — it
is the expensive group and the least time-critical.

## Release checklist

- [ ] `pytest` — all green, including the invariant tests
- [ ] `ruff check src tests`
- [ ] `mypy src`
- [ ] `neurogrip simulate all` — 5/5
- [ ] `neurogrip diagnose` on real hardware
- [ ] Firmware watchdog verified by unplugging the host mid-grasp
- [ ] E-stop verified, including that it requires acknowledgement
- [ ] `docs/safety.md` "Limits of this design" still accurate
