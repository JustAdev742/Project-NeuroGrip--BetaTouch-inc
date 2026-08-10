"""Debug console.

A text command interface exposed both on the touchscreen's hidden debug page and
over the CLI. It exists because during bring-up you always need to poke at
something the UI does not expose yet, and the alternative — adding a temporary
button — is how UIs rot.

Commands are read-only by default. Anything that can move the hand or change a
limit is marked ``dangerous`` and requires the console to be explicitly armed,
so an accidental keystroke cannot command a limb.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from dataclasses import dataclass, field

from ..core.logging import get_logger

__all__ = ["Command", "ConsoleResult", "DebugConsole"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConsoleResult:
    """Result of executing a command."""

    ok: bool
    output: str = ""
    lines: tuple[str, ...] = field(default_factory=tuple)

    @property
    def text(self) -> str:
        return self.output or "\n".join(self.lines)


@dataclass(frozen=True, slots=True)
class Command:
    """A registered console command."""

    name: str
    help: str
    handler: Callable[[list[str]], ConsoleResult]
    usage: str = ""
    #: Commands that can move the hand or change limits.
    dangerous: bool = False


class DebugConsole:
    """Command registry and dispatcher."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._history: list[str] = []
        #: Dangerous commands are refused until this is set.
        self.armed = False
        self.register("help", "List commands", self._help, usage="help [command]")
        self.register("history", "Show recent commands", self._history_cmd)

    # -- registration ---------------------------------------------------------

    def register(
        self,
        name: str,
        help_text: str,
        handler: Callable[[list[str]], ConsoleResult],
        *,
        usage: str = "",
        dangerous: bool = False,
    ) -> None:
        self._commands[name] = Command(
            name=name, help=help_text, handler=handler, usage=usage or name, dangerous=dangerous
        )

    @property
    def commands(self) -> tuple[Command, ...]:
        return tuple(sorted(self._commands.values(), key=lambda c: c.name))

    # -- execution ------------------------------------------------------------

    def execute(self, line: str) -> ConsoleResult:
        """Parse and run one command line."""
        line = line.strip()
        if not line:
            return ConsoleResult(ok=True)

        self._history.append(line)
        del self._history[:-100]

        try:
            parts = shlex.split(line)
        except ValueError as exc:
            return ConsoleResult(ok=False, output=f"parse error: {exc}")

        name, args = parts[0], parts[1:]
        command = self._commands.get(name)
        if command is None:
            suggestions = [c for c in self._commands if c.startswith(name[:2])]
            hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
            return ConsoleResult(ok=False, output=f"unknown command '{name}'.{hint}")

        if command.dangerous and not self.armed:
            return ConsoleResult(
                ok=False,
                output=(
                    f"'{name}' can move the hand or change limits. "
                    "Run 'arm' first if you really mean it."
                ),
            )

        try:
            result = command.handler(args)
        except Exception as exc:
            log.warning("console command failed", command=name, error=str(exc))
            return ConsoleResult(ok=False, output=f"{name} failed: {exc}")

        if command.dangerous:
            log.warning("dangerous console command executed", command=name, args=args)
        return result

    # -- built-ins ------------------------------------------------------------

    def _help(self, args: list[str]) -> ConsoleResult:
        if args:
            command = self._commands.get(args[0])
            if command is None:
                return ConsoleResult(ok=False, output=f"unknown command '{args[0]}'")
            marker = " [DANGEROUS]" if command.dangerous else ""
            return ConsoleResult(
                ok=True, lines=(f"{command.usage}{marker}", f"  {command.help}")
            )
        lines = [
            f"{'!' if c.dangerous else ' '} {c.name:<16} {c.help}" for c in self.commands
        ]
        return ConsoleResult(ok=True, lines=tuple(lines))

    def _history_cmd(self, args: list[str]) -> ConsoleResult:
        limit = int(args[0]) if args and args[0].isdigit() else 20
        return ConsoleResult(ok=True, lines=tuple(self._history[-limit:]))


def build_console(application) -> DebugConsole:
    """Wire the standard commands to a running application.

    Takes the application by duck type rather than by import to avoid a circular
    dependency between diagnostics and runtime.
    """
    console = DebugConsole()

    def arm(args: list[str]) -> ConsoleResult:
        console.armed = not args or args[0] != "off"
        return ConsoleResult(
            ok=True, output="console ARMED — dangerous commands enabled" if console.armed else "disarmed"
        )

    def status(args: list[str]) -> ConsoleResult:
        hand = application.controller.state
        mode = application.modes.current
        safety = application.safety.state
        return ConsoleResult(
            ok=True,
            lines=(
                f"mode      : {mode.value if mode else 'none'}",
                f"pose      : {hand.pose}",
                f"activity  : {hand.activity}",
                f"holding   : {hand.holding}   force {hand.force:.2f}",
                f"current   : {hand.total_current_ma} mA   {hand.temperature_c:.0f} °C",
                f"safety    : motion={safety.motion_allowed} ai={safety.ai_allowed} "
                f"{safety.primary_reason}",
                f"estop     : {safety.estop_engaged}",
            ),
        )

    def intent(args: list[str]) -> ConsoleResult:
        estimate = application.intent_engine.latest
        return ConsoleResult(
            ok=True,
            lines=(
                f"kind       : {estimate.kind.value}",
                f"confidence : {estimate.confidence:.3f}",
                f"strength   : {estimate.strength:.3f}",
                f"quality    : {estimate.quality.label}",
                f"activations: {[round(a, 3) for a in estimate.activations]}",
                *(f"  · {reason}" for reason in estimate.reasons),
            ),
        )

    def decision(args: list[str]) -> ConsoleResult:
        mode = application.modes.active
        current = mode.last_decision if mode else None
        if current is None:
            return ConsoleResult(ok=True, output="no decision yet")
        return ConsoleResult(
            ok=True,
            lines=(
                f"action     : {current.action.value}",
                f"confidence : {current.confidence:.3f}",
                *(f"  · {reason}" for reason in current.reasons),
                *(
                    ("evidence:", *(f"  {line}" for line in current.evidence.describe()))
                    if current.evidence
                    else ()
                ),
            ),
        )

    def vision(args: list[str]) -> ConsoleResult:
        pipeline = application.vision
        if pipeline is None:
            return ConsoleResult(ok=True, output="vision is not configured")
        result = pipeline.latest
        stats = pipeline.stats()
        lines = [
            f"backend    : {stats.backend}",
            f"fps        : {stats.fps:.1f}   latency {stats.mean_latency_ms:.1f} ms",
            f"tracks     : {stats.active_tracks}   errors {stats.errors}",
        ]
        lines.extend(
            f"  {d.label:<10} {d.confidence:.2f}  track {d.track_id} age {d.age}"
            for d in result.detections
        )
        if result.best_grasp:
            grasp = result.best_grasp
            lines.append(
                f"  grasp @({grasp.center_x:.2f},{grasp.center_y:.2f}) "
                f"{grasp.angle_degrees:.0f}° w={grasp.width:.2f} q={grasp.quality:.2f}"
            )
        return ConsoleResult(ok=True, lines=tuple(lines))

    def metrics(args: list[str]) -> ConsoleResult:
        snapshot = application.metrics.snapshot()
        lines: list[str] = []
        for group, values in snapshot.items():
            lines.append(f"[{group}]")
            for key, value in sorted(values.items()):
                lines.append(f"  {key:<30} {value}")
        return ConsoleResult(ok=True, lines=tuple(lines))

    def topics(args: list[str]) -> ConsoleResult:
        events = application.bus.history(args[0] if args else None, limit=25)
        return ConsoleResult(
            ok=True, lines=tuple(f"{e.timestamp:9.3f} {e.topic:<28} {e.source}" for e in events)
        )

    def selftest(args: list[str]) -> ConsoleResult:
        report = application.diagnostics.selftest.run(allow_motion="--motion" in args)
        return ConsoleResult(
            ok=report.ok,
            lines=(*(str(r) for r in report.results), "", report.summary()),
        )

    def estop(args: list[str]) -> ConsoleResult:
        application.safety.trigger_estop("debug console", source="user:console")
        return ConsoleResult(ok=True, output="emergency stop engaged")

    def acknowledge(args: list[str]) -> ConsoleResult:
        cleared = application.safety.acknowledge("user:console")
        return ConsoleResult(
            ok=cleared,
            output="faults acknowledged" if cleared else "refused: critical faults still active",
        )

    def mode(args: list[str]) -> ConsoleResult:
        from ..core.types import ModeId

        if not args:
            current = application.modes.current.value if application.modes.current else "none"
            return ConsoleResult(
                ok=True,
                output=f"current: {current}; "
                f"available: {', '.join(m.value for m in application.modes.available)}",
            )
        try:
            target = ModeId(args[0])
        except ValueError:
            return ConsoleResult(ok=False, output=f"unknown mode '{args[0]}'")
        ok = application.modes.activate(target, reason="debug console")
        return ConsoleResult(ok=ok, output=f"mode change {'accepted' if ok else 'rejected'}")

    def grip(args: list[str]) -> ConsoleResult:
        from ..core.types import GraspType

        if not args:
            return ConsoleResult(
                ok=True,
                output="available: " + ", ".join(g.value for g in application.controller.grips.available),
            )
        try:
            grasp = GraspType(args[0])
        except ValueError:
            return ConsoleResult(ok=False, output=f"unknown grip '{args[0]}'")
        result = application.controller.apply_grip(grasp, source="console")
        return ConsoleResult(ok=result.accepted, output=result.reason or f"applying {grasp.label}")

    console.register("arm", "Enable dangerous commands (arm off to disable)", arm)
    console.register("status", "Hand, mode and safety summary", status)
    console.register("intent", "Current EMG intent estimate", intent)
    console.register("decision", "Most recent fusion decision and its evidence", decision)
    console.register("vision", "Vision backend status and detections", vision)
    console.register("metrics", "Dump the metrics registry", metrics)
    console.register("topics", "Recent event-bus traffic", topics, usage="topics [topic]")
    console.register("selftest", "Run the self-test suite", selftest, usage="selftest [--motion]")
    console.register("estop", "Engage the emergency stop", estop, dangerous=True)
    console.register("ack", "Acknowledge faults and release e-stop", acknowledge, dangerous=True)
    console.register("mode", "Show or change the operating mode", mode, usage="mode [name]", dangerous=True)
    console.register("grip", "Apply a grip preset", grip, usage="grip [name]", dangerous=True)
    return console
