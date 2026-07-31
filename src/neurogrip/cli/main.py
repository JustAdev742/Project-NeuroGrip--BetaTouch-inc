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

    calibrate = subparsers.add_parser("calibrate", parents=[common], help="run the EMG calibration wizard")
    calibrate.add_argument("--output", help="where to save the calibration")

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
