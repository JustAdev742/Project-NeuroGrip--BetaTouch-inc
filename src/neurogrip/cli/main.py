"""Command-line interface.

    neurogrip run          start the system
    neurogrip simulate     run a scenario against the simulated hardware
    neurogrip diagnose     run the self-tests and print a health report
    neurogrip calibrate    run the EMG calibration wizard
    neurogrip train        run a training exercise from the terminal
    neurogrip record       capture raw EMG to a file
    neurogrip replay       replay a recording through the live pipeline
    neurogrip console      interactive debug console
    neurogrip config       print the merged configuration
    neurogrip info         print system and hardware information

Every subcommand shares ``--config``, ``--profile``, ``--set`` and ``--log-level``,
so any of them can be pointed at any configuration.
"""

from __future__ import annotations

import argparse
import sys

from .. import __version__
from ..core.clock import SimulatedClock
from ..core.logging import configure_logging, get_logger
from ..core.types import Finger
from ..runtime.application import build_application
from ..runtime.bootstrap import load_configuration

__all__ = ["build_parser", "main"]

log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurogrip",
        description="AI-assisted prosthetic hand control system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"neurogrip {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", help="path to a TOML configuration file")
    common.add_argument("--profile", "-p", help="profile name in config/ (e.g. simulation)")
    common.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a configuration key (repeatable)",
    )
    common.add_argument("--log-level", help="override the logging level")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", parents=[common], help="start the system")
    run.add_argument("--no-motion", action="store_true", help="start without energising the actuators")
    run.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = forever)")

    simulate = subparsers.add_parser(
        "simulate", parents=[common], help="run a scenario against simulated hardware"
    )
    simulate.add_argument("scenario", nargs="?", default="all", help="scenario name, or 'all'")
    simulate.add_argument("--list", action="store_true", help="list the available scenarios")
    simulate.add_argument("--timeline", action="store_true", help="print the sampled timeline")

    subparsers.add_parser("diagnose", parents=[common], help="run self-tests and report health")

    calibrate = subparsers.add_parser(
        "calibrate", parents=[common], help="run a calibration wizard"
    )
    calibrate.add_argument(
        "target",
        nargs="?",
        default="emg",
        choices=("emg", "servo", "camera"),
        help="emg: muscle signals | servo: tendon slack | camera: field of view",
    )
    calibrate.add_argument("--output", help="where to save the calibration")
    calibrate.add_argument(
        "--finger",
        action="append",
        default=[],
        help="servo only: calibrate just this finger (repeatable)",
    )
    calibrate.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="TARGET:DISTANCE_M:WIDTH_PX",
        help="camera only: one measurement, e.g. card:0.30:214 (repeatable)",
    )

    test = subparsers.add_parser(
        "test", parents=[common], help="hardware bring-up tests"
    )
    test.add_argument(
        "tool",
        choices=("servos", "link", "range", "estop", "all"),
        help=(
            "servos: sweep every servo (use this before the hand is built) | "
            "link: communication quality | range: motor travel on a finished hand | "
            "estop: emergency stop"
        ),
    )
    test.add_argument(
        "--samples", type=int, default=200, help="link only: number of round trips"
    )
    test.add_argument(
        "--finger",
        action="append",
        default=[],
        help="servos only: sweep just this finger (repeatable)",
    )
    test.add_argument(
        "--cycles", type=int, default=1, help="servos only: sweeps per channel"
    )
    test.add_argument(
        "--travel",
        type=float,
        default=1.0,
        help="servos only: fraction of full closure to sweep (0.05–1.0)",
    )
    test.add_argument(
        "--speed", type=float, default=0.5, help="servos only: speed scale (0.05–1.0)"
    )

    profile_cmd = subparsers.add_parser(
        "profile", parents=[common], help="manage saved user profiles"
    )
    profile_cmd.add_argument(
        "action", choices=("list", "show", "create", "use", "delete"), help="what to do"
    )
    profile_cmd.add_argument("name", nargs="?", default="", help="profile name")

    train = subparsers.add_parser("train", parents=[common], help="run a training exercise")
    train.add_argument("exercise", nargs="?", default="", help="exercise key")
    train.add_argument("--list", action="store_true", help="list the available exercises")
    train.add_argument("--difficulty", default="", help="beginner|easy|medium|hard|expert")

    record = subparsers.add_parser("record", parents=[common], help="capture raw EMG to a file")
    record.add_argument("output", help="output file path")
    record.add_argument("--seconds", type=float, default=30.0, help="recording length")
    record.add_argument("--subject", default="default", help="subject identifier")
    record.add_argument("--label", default="", help="label applied to every sample")

    replay = subparsers.add_parser("replay", parents=[common], help="replay a recording")
    replay.add_argument("recording", help="path to a recording")
    replay.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")

    subparsers.add_parser("console", parents=[common], help="interactive debug console")

    config_cmd = subparsers.add_parser("config", parents=[common], help="print the merged configuration")
    config_cmd.add_argument("path", nargs="?", default="", help="print only this key or section")
    config_cmd.add_argument(
        "--check",
        action="store_true",
        help="validate instead of printing; exits non-zero on any error",
    )

    subparsers.add_parser("info", parents=[common], help="print system and hardware information")

    return parser


def _load(args: argparse.Namespace):
    overrides = list(args.overrides)
    if args.log_level:
        overrides.append(f'logging.level="{args.log_level}"')
    return load_configuration(
        config_path=args.config, profile=args.profile, overrides=overrides
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_run(args: argparse.Namespace) -> int:
    config = _load(args)
    application = build_application(config)

    if not application.start(allow_motion=not args.no_motion):
        log.error("startup was refused")
        return 2

    if args.duration > 0:
        deadline = application.clock.monotonic() + args.duration
        application.scheduler.run(until=lambda: application.clock.monotonic() >= deadline)
        application.stop()
    else:
        application.run()
    return 0


def command_simulate(args: argparse.Namespace) -> int:
    from ..simulation import DEMO_SCENARIOS, ScenarioRunner, build_scenario

    if args.list:
        for name, factory in DEMO_SCENARIOS.items():
            print(f"{name:<24} {factory().description}")
        return 0

    overrides = list(args.overrides)
    args.overrides = overrides
    if not args.config and not args.profile:
        args.profile = "simulation"
    config = _load(args)
    # Scenarios print their own report; a live text UI would interleave with it.
    config = config.with_overlay({"ui": {"renderer": "null"}}, source="simulate")
    if not config.get_bool("hardware.simulate", False):
        print("simulate requires a simulated profile (try --profile simulation)", file=sys.stderr)
        return 2

    names = list(DEMO_SCENARIOS) if args.scenario == "all" else [args.scenario]
    failures = 0

    for name in names:
        clock = SimulatedClock()
        application = build_application(config, clock)
        application.start(allow_motion=True)
        runner = ScenarioRunner(application, clock)
        try:
            result = runner.run(build_scenario(name))
        except KeyError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(result.report())
        if args.timeline:
            for sample in result.timeline:
                print(
                    f"  t={sample['t']:5.2f}  pose={sample['pose']}  "
                    f"intent={sample['intent']:<7} action={sample['action']:<9} "
                    f"{'AI:' + sample['grasp'] if sample['ai'] else ''}"
                )
        if not result.passed:
            failures += 1
        application.stop()

    print(f"\n{len(names) - failures}/{len(names)} scenarios passed")
    return 1 if failures else 0


def command_diagnose(args: argparse.Namespace) -> int:
    config = _load(args)
    application = build_application(config)
    application.services.start_all()
    try:
        if application.estop_check is not None:
            # Establish a verdict before reporting one. In a long-running system
            # the checker paces itself, but a one-shot `diagnose` would otherwise
            # always report "not yet verified" — and a warning that always fires
            # is a warning people learn to skip.
            application.estop_check.tick()
            for _ in range(20):
                application.clock.sleep(0.01)
                application.controller.tick()
                application.estop_check.tick()

        report = application.diagnostics.selftest.run(allow_motion=False)
        print("\nSelf-test")
        print("─" * 60)
        for result in report.results:
            print(f"  {result.outcome.symbol} {result.name:<26} {result.message}")
        print(f"\n  {report.summary()}")
        if report.remedies():
            print("\nSuggested actions:")
            for remedy in report.remedies():
                print(f"  • {remedy}")

        application.diagnostics.tick()
        print("\nHealth")
        print("─" * 60)
        for health in application.health():
            print(f"  [{health.status.label:<8}] {health.name:<16} {health.detail}")

        snapshot = application.diagnostics.snapshot
        if snapshot is not None:
            system = snapshot.system
            print("\nResources")
            print("─" * 60)
            print(f"  CPU        {system.cpu_percent:5.1f}%   {system.cpu_temperature_c:.0f} °C")
            print(
                f"  Memory     {system.memory_percent:5.1f}%   "
                f"{system.memory_used_mb:.0f}/{system.memory_total_mb:.0f} MB"
            )
            print(f"  Battery    {snapshot.battery.percentage:5.1f}%   {snapshot.battery.voltage_v:.2f} V")
        return 0 if report.ok else 1
    finally:
        application.services.stop_all()


def command_calibrate(args: argparse.Namespace) -> int:
    if args.target == "servo":
        return _calibrate_servo(args)
    if args.target == "camera":
        return _calibrate_camera(args)
    return _calibrate_emg(args)


def _calibrate_camera(args: argparse.Namespace) -> int:
    """Solve camera field of view from measurements of a known target.

    Takes measurements rather than frames, so it works with a tape measure and a
    bank card and needs neither a camera attached nor OpenCV installed.
    """
    from ..core.errors import CalibrationError
    from ..vision.calibration import REFERENCE_TARGETS, CameraCalibrationWizard

    config = _load(args)
    wizard = CameraCalibrationWizard(
        config.get_int("camera.width", 640), config.get_int("camera.height", 480)
    )

    if not args.sample:
        print("Measure a target of known width at several distances, then pass each as")
        print("  --sample TARGET:DISTANCE_M:WIDTH_PX\n")
        print("Known targets:")
        for name, width in REFERENCE_TARGETS.items():
            print(f"  {name:<8} {width * 1000:6.1f} mm wide")
        print("\nExample:")
        print("  neurogrip calibrate camera --sample card:0.20:214 \\")
        print("                             --sample card:0.35:122 \\")
        print("                             --sample card:0.50:86")
        return 2

    try:
        for raw in args.sample:
            parts = raw.split(":")
            if len(parts) != 3:
                print(f"malformed sample {raw!r}; want TARGET:DISTANCE_M:WIDTH_PX", file=sys.stderr)
                return 2
            wizard.add_measurement(parts[0], float(parts[1]), float(parts[2]))
        calibration = wizard.solve()
    except (CalibrationError, ValueError) as exc:
        print(f"calibration failed: {exc}", file=sys.stderr)
        return 1

    print(f"\n  {calibration.describe()}")
    print("\n  per-sample distance error:")
    for label, error in wizard.residuals(calibration):
        print(f"    {label:<28} {error * 100:+6.1f} cm")

    if not calibration.is_trustworthy:
        print(f"\n  ! {calibration.notes}")
    print(
        f"\n  Set camera.fov_deg = {calibration.horizontal_fov_deg:.1f} "
        f"(currently {config.get_float('camera.fov_deg', 62.0):.1f})"
    )
    if args.output:
        calibration.save(args.output)
        print(f"  saved to {args.output}")
    return 0 if calibration.is_trustworthy else 1


def _calibrate_servo(args: argparse.Namespace) -> int:
    """Measure per-finger tendon slack by driving each finger under low force."""
    from ..control.servo_calibration import ServoCalibrationPhase

    config = _load(args)
    application = build_application(config)
    if application.servo_calibration is None:
        print("no servo calibration wizard is configured", file=sys.stderr)
        return 2

    if not application.start(allow_motion=True):
        print("startup was refused; cannot calibrate", file=sys.stderr)
        return 2

    try:
        fingers = None
        if args.finger:
            try:
                fingers = tuple(Finger[name.upper()] for name in args.finger)
            except KeyError:
                names = ", ".join(f.name.lower() for f in Finger)
                print(f"unknown finger; choose from: {names}", file=sys.stderr)
                return 2

        wizard = application.servo_calibration
        print("\nServo calibration — the hand will move one finger at a time.\n")
        wizard.start(fingers)

        last_title = ""
        while not wizard.progress().finished:
            application.scheduler.step()
            if isinstance(application.clock, SimulatedClock):
                application.clock.advance(0.005)
            progress = wizard.progress()
            title = f"{progress.title} · {progress.instruction}"
            if title != last_title and progress.instruction:
                last_title = title
                print(f"  ▸ {title}")

        print()
        for result in wizard.results:
            symbol = "✓" if result.ok else "✗"
            print(f"  {symbol} {result.describe()}")

        if wizard.phase is ServoCalibrationPhase.FAILED:
            print("\n  ✗ Calibration failed — fix the mechanical problems above and re-run.")
            return 1

        path = args.output or application.servo_calibration_path
        application.servo_calibration_path = path
        if application.save_servo_calibration():
            print(f"\n  ✓ Saved to {path}")
        return 0
    finally:
        application.stop()


def _calibrate_emg(args: argparse.Namespace) -> int:
    from ..emg.calibration import CalibrationPhase

    config = _load(args)
    application = build_application(config)
    application.services.start_all()
    try:
        service = application.emg
        if service.wizard is None:
            print("no calibration wizard is configured", file=sys.stderr)
            return 2

        print("\nEMG calibration — follow the prompts.\n")
        service.start_calibration()
        last_title = ""
        while True:
            application.scheduler.step()
            if isinstance(application.clock, SimulatedClock):
                application.clock.advance(0.005)
            progress = service.wizard.progress()
            if progress.title != last_title and progress.instruction:
                last_title = progress.title
                print(f"  ▸ {progress.title}: {progress.instruction}")
            if progress.finished:
                break

        if service.wizard.phase is CalibrationPhase.FAILED:
            print(f"\n  ✗ Calibration failed: {progress.message}")
            return 1

        result = service.wizard.result
        print("\n  ✓ Calibration complete\n")
        for channel in (result.channels.values() if result else ()):
            print(
                f"    {channel.name:<10} rest {channel.rest_mean * 1e6:7.1f} µV   "
                f"MVC {channel.mvc * 1e6:8.1f} µV   SNR {channel.snr_ratio:6.1f}×"
            )
        if args.output and result is not None:
            result.save(args.output)
            print(f"\n  saved to {args.output}")
        return 0
    finally:
        application.services.stop_all()


def command_test(args: argparse.Namespace) -> int:
    """Run hardware bring-up tests.

    Separate from ``diagnose``, which gates startup and must be quick. These move
    the hand and deliberately trigger the emergency stop, so they are opt-in.
    """
    from ..diagnostics.bringup import EstopTester, LinkTester, RangeTester, ServoSweepTest
    from ..hal.base import DeviceCapability

    config = _load(args)
    application = build_application(config)

    tools = ("link", "servos", "range", "estop") if args.tool == "all" else (args.tool,)
    needs_motion = any(t in ("servos", "range", "estop") for t in tools)

    if not application.start(allow_motion=needs_motion):
        print("startup was refused; cannot run bring-up tests", file=sys.stderr)
        return 2

    failures = 0
    try:
        for tool in tools:
            if tool == "link":
                report = LinkTester(
                    application.hardware.servo_bus, application.clock, samples=args.samples
                ).run()
            elif tool == "servos":
                capabilities = application.hardware.servo_bus.info().capabilities
                report = ServoSweepTest(
                    application.controller,
                    application.clock,
                    cycles=args.cycles,
                    travel=args.travel,
                    speed=args.speed,
                    has_feedback=DeviceCapability.POSITION_FEEDBACK in capabilities,
                ).run(_fingers_from(args) or tuple(Finger))
            elif tool == "range":
                report = RangeTester(application.controller, application.clock).run()
            else:
                report = EstopTester(
                    application.controller, application.safety, application.clock
                ).run()

            print(f"\n{report.summary()}")
            for line in report.describe():
                print(line)
            if not report.ok:
                failures += 1
    finally:
        application.stop()

    if failures:
        print(f"\n{failures} tool(s) reported failures", file=sys.stderr)
    return 1 if failures else 0


def _fingers_from(args: argparse.Namespace) -> tuple:
    """Parse repeated ``--finger`` options into a tuple, or ``()`` for all."""
    if not getattr(args, "finger", None):
        return ()
    try:
        return tuple(Finger[name.upper()] for name in args.finger)
    except KeyError:
        names = ", ".join(f.name.lower() for f in Finger)
        raise SystemExit(f"unknown finger; choose from: {names}") from None


def command_profile(args: argparse.Namespace) -> int:
    """Manage saved user profiles."""
    from ..core.errors import ConfigurationError
    from ..core.profiles import ProfileStore

    config = _load(args)
    store = ProfileStore(config.get_str("ui.profile_path", "var/profiles"))
    active = store.active_name()

    try:
        if args.action == "list":
            names = store.names()
            if not names:
                print("no profiles yet; one is created on first use")
                return 0
            for name in names:
                marker = "*" if name == active else " "
                profile = store.load(name)
                print(f" {marker} {name:<16} {profile.display_name:<20} "
                      f"{len(profile.settings)} setting(s)")
            return 0

        if not args.name:
            print(f"{args.action} needs a profile name", file=sys.stderr)
            return 2

        if args.action == "show":
            profile = store.load(args.name)
            print(f"{profile.name} ({profile.display_name})")
            for path, value in sorted(profile.settings.items()):
                print(f"  {path:<36} {value}")
            return 0

        if args.action == "create":
            store.create(args.name)
            print(f"created {args.name}")
            return 0

        if args.action == "use":
            store.set_active(args.name)
            print(f"active profile is now {args.name}")
            return 0

        store.delete(args.name)
        print(f"deleted {args.name}")
        return 0
    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2


def command_train(args: argparse.Namespace) -> int:
    from ..training.exercises import EXERCISES, Difficulty

    if args.list or not args.exercise:
        print("\nAvailable exercises:\n")
        for key, cls in EXERCISES.items():
            print(f"  {key:<14} {cls.title:<20} {cls.description}")
        return 0

    config = _load(args)
    application = build_application(config)
    application.services.start_all()
    try:
        difficulty = Difficulty(args.difficulty) if args.difficulty else None
        if not application.training.start(args.exercise, difficulty):
            print(f"unknown exercise '{args.exercise}'", file=sys.stderr)
            return 2

        print(f"\n{args.exercise} — follow the prompts on screen.\n")
        while application.training.active:
            application.scheduler.step()
            if isinstance(application.clock, SimulatedClock):
                application.clock.advance(0.005)
            state = application.training.state
            if state is not None:
                bar = "█" * int(state.actual * 20) + "░" * (20 - int(state.actual * 20))
                print(
                    f"\r  {state.prompt:<38} [{bar}] "
                    f"{state.trial + 1}/{state.trials_total}  {state.feedback:<26}",
                    end="",
                    flush=True,
                )

        summary = application.training.summary
        if summary is not None:
            print("\n")
            print(f"  {'★' * summary.stars}{'☆' * (3 - summary.stars)}  {summary.mean_score * 100:.0f}%")
            print(f"  {summary.advice}")
            for achievement in summary.achievements:
                print(f"  🏆 {achievement}")
        return 0
    finally:
        application.services.stop_all()


def command_record(args: argparse.Namespace) -> int:
    config = _load(args)
    application = build_application(config)
    application.services.start_all()
    try:
        application.emg.start_recording(args.output, subject=args.subject)
        if args.label:
            application.emg.label_recording(args.label)
        print(f"recording {args.seconds:.0f}s to {args.output} …")
        deadline = application.clock.monotonic() + args.seconds
        while application.clock.monotonic() < deadline:
            application.scheduler.step()
            if isinstance(application.clock, SimulatedClock):
                application.clock.advance(0.005)
        info = application.emg.stop_recording()
        if info is not None:
            print(
                f"  {info.samples} samples over {info.duration_s:.1f}s "
                f"({info.channels} channels) → {info.path}"
            )
        return 0
    finally:
        application.services.stop_all()


def command_replay(args: argparse.Namespace) -> int:
    config = _load(args).with_overlay(
        {
            "hardware": {"simulate": True},
            "emg": {
                "driver": "replay",
                "recording": args.recording,
                "replay_speed": args.speed,
                "replay_loop": False,
            },
        },
        source="replay",
    )
    application = build_application(config)
    application.services.start_all()
    try:
        source = application.emg.source
        print(f"replaying {args.recording} at {args.speed}× …\n")
        while not getattr(source, "finished", True):
            application.scheduler.step()
            if isinstance(application.clock, SimulatedClock):
                application.clock.advance(0.005)
            intent = application.emg.intent
            if intent is not None:
                print(
                    f"\r  {getattr(source, 'progress', 0.0) * 100:5.1f}%  "
                    f"intent={intent.kind.value:<8} conf={intent.confidence:.2f} "
                    f"strength={intent.strength:.2f}",
                    end="",
                    flush=True,
                )
        print("\n\nreplay complete")
        return 0
    finally:
        application.services.stop_all()


def command_console(args: argparse.Namespace) -> int:
    config = _load(args)
    application = build_application(config)
    application.start(allow_motion=False)
    console = application.console
    if console is None:
        print("console unavailable", file=sys.stderr)
        return 2

    print("NeuroGrip debug console. Type 'help' for commands, 'quit' to exit.\n")
    try:
        while True:
            try:
                line = input("neurogrip> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if line in ("quit", "exit"):
                break
            # Advance the system so the console reflects live state.
            for _ in range(20):
                application.scheduler.step()
            result = console.execute(line)
            if result.text:
                print(result.text)
    finally:
        application.stop()
    return 0


def command_config(args: argparse.Namespace) -> int:
    import json

    config = _load(args)

    if args.check:
        from ..core.validation import validate_config

        report = validate_config(config)
        for issue in report.issues:
            stream = sys.stderr if issue.severity.value == "error" else sys.stdout
            print(str(issue), file=stream)
        if report.ok:
            print(
                f"configuration OK ({len(report.warnings)} warning(s)) — "
                f"sources: {', '.join(config.sources) or 'built-in defaults'}"
            )
            return 0
        print(f"\n{len(report.errors)} error(s) must be fixed before starting", file=sys.stderr)
        return 2

    data = config.section(args.path).as_dict() if args.path else config.as_dict()
    print(json.dumps(data, indent=2, default=str))
    print(f"\n# sources: {', '.join(config.sources) or 'built-in defaults'}", file=sys.stderr)
    return 0


def command_info(args: argparse.Namespace) -> int:
    config = _load(args)
    application = build_application(config)
    print(f"\nNeuroGrip {__version__}")
    print("─" * 60)
    for key, value in application.describe().items():
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value) or "—"
        print(f"  {key:<18} {value}")
    return 0


COMMANDS = {
    "run": command_run,
    "simulate": command_simulate,
    "diagnose": command_diagnose,
    "calibrate": command_calibrate,
    "test": command_test,
    "profile": command_profile,
    "train": command_train,
    "record": command_record,
    "replay": command_replay,
    "console": command_console,
    "config": command_config,
    "info": command_info,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Minimal logging until the configuration is loaded, so early failures are
    # still visible.
    configure_logging(level=args.log_level or "INFO", console=True)

    handler = COMMANDS.get(args.command)
    if handler is None:  # pragma: no cover - argparse enforces this
        parser.error(f"unknown command {args.command}")

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except Exception as exc:
        log.critical("fatal error", error=str(exc), exc_info=True)
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
