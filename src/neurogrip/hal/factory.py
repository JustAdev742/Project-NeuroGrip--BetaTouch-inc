"""Configuration-driven device construction.

This is the only place in the codebase that decides *which* concrete driver to
instantiate. Everything above it receives an interface. Adding a new servo bus or
camera means adding a branch here and a driver module — no other file changes.

Two behaviours deserve highlighting:

* **Graceful degradation.** If a device configured as ``required = false`` cannot
  be opened, the factory logs it and substitutes the simulated implementation, so
  a broken camera yields a hand that still works manually. Devices marked
  required propagate the error and abort startup.
* **Explicit simulation.** ``[hardware] simulate = true`` selects simulated
  drivers for everything, which is the mode CI and Training Mode run in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.clock import Clock
from ..core.config import Config
from ..core.errors import ConfigurationError, DeviceError
from ..core.logging import get_logger
from .camera.base import CameraSettings, CameraSource, PixelFormat
from .camera.opencv import OpenCvCamera
from .camera.simulated import SceneObject, SimulatedCamera
from .emg.base import EmgChannelSpec, EmgSource
from .emg.replay import ReplayEmgSource
from .emg.serial_source import SerialEmgSource
from .emg.simulated import DEFAULT_CHANNELS, SimulatedEmgSource
from .servo.base import ServoBus, ServoLimits
from .servo.emulator import Esp32Emulator
from .servo.esp32 import Esp32ServoBus
from .servo.simulated import SimulatedServoBus
from .system import (
    ConnectivityProbe,
    PowerSource,
    SimulatedPowerSource,
    SysfsPowerSource,
    SystemProbe,
)
from .transport.base import Transport
from .transport.loopback import LoopbackTransport
from .transport.reconnecting import ReconnectingTransport
from .transport.serial_link import SerialTransport

__all__ = ["HardwareBundle", "HardwareFactory"]

log = get_logger(__name__)


@dataclass(slots=True)
class HardwareBundle:
    """Every device the runtime needs, already constructed but not yet opened."""

    servo_bus: ServoBus
    emg_source: EmgSource
    camera: CameraSource | None
    power: PowerSource
    system: SystemProbe
    connectivity: ConnectivityProbe
    #: Populated when the servo bus runs against the in-process firmware emulator;
    #: the simulation harness uses it to inject contact models.
    emulator: Esp32Emulator | None = None
    #: Populated in simulation so the scenario runner can drive the plant directly.
    simulated_plant: SimulatedServoBus | None = None
    notes: tuple[str, ...] = ()

    def describe(self) -> dict[str, str]:
        """Human-readable summary for the System Information screen."""
        return {
            "servo_bus": str(self.servo_bus.info()),
            "emg": str(self.emg_source.info()),
            "camera": str(self.camera.info()) if self.camera else "none",
            "power": str(self.power.info()),
        }


class HardwareFactory:
    """Builds a :class:`HardwareBundle` from configuration."""

    def __init__(self, config: Config, clock: Clock) -> None:
        self._config = config
        self._clock = clock

    # -- entry point ----------------------------------------------------------

    def build(self) -> HardwareBundle:
        hardware = self._config.section("hardware")
        simulate = hardware.get_bool("simulate", False)
        notes: list[str] = []

        servo_bus, emulator, plant = self._build_servo_bus(simulate, notes)
        emg_source = self._build_emg(simulate, notes)
        camera = self._build_camera(simulate, notes)
        power = (
            SimulatedPowerSource(clock=self._clock)
            if simulate
            else SysfsPowerSource(self._config.get_str("power.node", "/sys/class/power_supply/BAT0"))
        )

        return HardwareBundle(
            servo_bus=servo_bus,
            emg_source=emg_source,
            camera=camera,
            power=power,
            system=SystemProbe(),
            connectivity=ConnectivityProbe(simulated=simulate),
            emulator=emulator,
            simulated_plant=plant,
            notes=tuple(notes),
        )

    # -- servo ----------------------------------------------------------------

    def _build_servo_bus(
        self, simulate: bool, notes: list[str]
    ) -> tuple[ServoBus, Esp32Emulator | None, SimulatedServoBus | None]:
        section = self._config.section("servo")
        limits = ServoLimits(
            max_velocity=section.get_float("max_velocity", 2.0),
            max_acceleration=section.get_float("max_acceleration", 8.0),
            max_current_ma=section.get_int("max_current_ma", 900),
            max_temperature_c=section.get_float("max_temperature_c", 65.0),
            max_force=section.get_float("max_force", 0.85),
        )
        limits.validate()

        driver = "simulated" if simulate else section.get_str("driver", "esp32")

        if driver == "simulated":
            plant = SimulatedServoBus(self._clock, limits=limits)
            notes.append("servo: simulated plant")
            return plant, None, plant

        if driver == "emulator":
            # Real driver + real protocol against the in-process firmware emulator.
            plant = SimulatedServoBus(self._clock, limits=limits)
            emulator = Esp32Emulator(
                self._clock, plant=plant, watchdog_ms=section.get_int("watchdog_ms", 300)
            )
            transport = LoopbackTransport(
                emulator,
                self._clock,
                latency=section.get_float("emulator_latency_s", 0.001),
            )
            notes.append("servo: esp32 driver against firmware emulator")
            return (
                Esp32ServoBus(
                    transport,
                    self._clock,
                    limits=limits,
                    watchdog_ms=section.get_int("watchdog_ms", 300),
                    state_timeout=section.get_float("state_timeout_s", 0.25),
                ),
                emulator,
                plant,
            )

        if driver == "esp32":
            transport = self._build_transport(section)
            bus = Esp32ServoBus(
                transport,
                self._clock,
                limits=limits,
                watchdog_ms=section.get_int("watchdog_ms", 300),
                state_timeout=section.get_float("state_timeout_s", 0.25),
            )
            if isinstance(transport, ReconnectingTransport):
                # Replaying limits and calibration is the driver's job, but only
                # the transport knows when the link came back.
                transport.on_reconnect = bus.resync
            if section.get_bool("required", True):
                return bus, None, None
            try:
                bus.open()
                bus.close()
            except DeviceError as exc:
                log.warning("motor controller unavailable, using simulation", error=str(exc))
                notes.append(f"servo: fell back to simulation ({exc})")
                plant = SimulatedServoBus(self._clock, limits=limits)
                return plant, None, plant
            return bus, None, None

        raise ConfigurationError(f"unknown servo driver '{driver}'")

    def _build_transport(self, section: Config) -> Transport:
        serial = SerialTransport(
            section.get_str("port", "/dev/ttyUSB0"),
            baudrate=section.get_int("baud", 921_600),
            rtscts=section.get_bool("rtscts", False),
            dtr_reset=section.get_bool("dtr_reset", False),
        )
        if not section.get_bool("reconnect", True):
            return serial
        return ReconnectingTransport(
            serial,
            self._clock,
            initial_backoff_s=section.get_float("reconnect_backoff_s", 0.5),
            max_backoff_s=section.get_float("reconnect_max_backoff_s", 8.0),
        )

    # -- EMG ------------------------------------------------------------------

    def _build_emg(self, simulate: bool, notes: list[str]) -> EmgSource:
        section = self._config.section("emg")
        channels = self._emg_channels(section)
        rate = section.get_float("sample_rate_hz", 1000.0)
        driver = "simulated" if simulate else section.get_str("driver", "serial")

        if driver == "simulated":
            notes.append("emg: synthetic source")
            return SimulatedEmgSource(
                self._clock,
                sample_rate_hz=rate,
                channels=channels,
                mains_hz=section.get_float("mains_hz", 50.0),
                seed=section.get_int("seed", 20260730),
            )

        if driver == "replay":
            path = section.get_str("recording")
            notes.append(f"emg: replaying {path}")
            return ReplayEmgSource(
                path,
                self._clock,
                loop=section.get_bool("replay_loop", True),
                speed=section.get_float("replay_speed", 1.0),
            )

        if driver == "serial":
            transport = ReconnectingTransport(
                SerialTransport(
                    section.get_str("port", "/dev/ttyACM0"),
                    baudrate=section.get_int("baud", 460_800),
                ),
                self._clock,
            )
            source: EmgSource = SerialEmgSource(
                transport,
                channels,
                self._clock,
                sample_rate_hz=rate,
                adc_reference_v=section.get_float("adc_reference_v", 4.096),
                adc_bits=section.get_int("adc_bits", 16),
                amplifier_gain=section.get_float("amplifier_gain", 1000.0),
            )
            if section.get_bool("required", True):
                return source
            try:
                source.open()
                source.close()
            except DeviceError as exc:
                log.warning("EMG front end unavailable, using simulation", error=str(exc))
                notes.append(f"emg: fell back to simulation ({exc})")
                return SimulatedEmgSource(self._clock, sample_rate_hz=rate, channels=channels)
            return source

        raise ConfigurationError(f"unknown EMG driver '{driver}'")

    def _emg_channels(self, section: Config) -> tuple[EmgChannelSpec, ...]:
        raw: list[Any] = section.get_list("channels", [])
        if not raw:
            return DEFAULT_CHANNELS
        channels = []
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ConfigurationError("emg.channels entries must be tables")
            channels.append(
                EmgChannelSpec(
                    index=int(entry.get("index", index)),
                    name=str(entry.get("name", f"ch{index}")),
                    role=str(entry.get("role", "auxiliary")),
                    full_scale_v=float(entry.get("full_scale_v", 3.3)),
                    site=str(entry.get("site", "")),
                )
            )
        roles = {c.role for c in channels}
        if "flexor" not in roles or "extensor" not in roles:
            raise ConfigurationError(
                "emg.channels must include one 'flexor' and one 'extensor' channel",
                context={"roles": sorted(roles)},
            )
        return tuple(channels)

    # -- camera ---------------------------------------------------------------

    def _build_camera(self, simulate: bool, notes: list[str]) -> CameraSource | None:
        section = self._config.section("camera")
        if not section.get_bool("enabled", True):
            notes.append("camera: disabled by configuration")
            return None

        settings = CameraSettings(
            width=section.get_int("width", 640),
            height=section.get_int("height", 480),
            fps=section.get_float("fps", 30.0),
            pixel_format=PixelFormat(section.get_str("pixel_format", "rgb888")),
            exposure_us=section.get_int("exposure_us", 0) or None,
            autofocus=section.get_bool("autofocus", True),
        )
        driver = "simulated" if simulate else section.get_str("driver", "opencv")

        if driver == "simulated":
            scene_cfg = section.section("scene")
            notes.append("camera: synthetic scene")
            return SimulatedCamera(
                self._clock,
                settings=CameraSettings(
                    width=section.get_int("width", 160),
                    height=section.get_int("height", 120),
                    fps=settings.fps,
                ),
                scene=SceneObject(
                    label=scene_cfg.get_str("label", "bottle"),
                    distance_m=scene_cfg.get_float("distance_m", 0.35),
                    shape=scene_cfg.get_str("shape", "cylinder"),
                ),
            )

        if driver == "opencv":
            camera = OpenCvCamera(
                section.get("device", 0), self._clock, settings=settings,
                backend=section.get_str("backend", "auto"),
            )
            if section.get_bool("required", False):
                return camera
            try:
                camera.open()
                camera.close()
            except DeviceError as exc:
                log.warning("camera unavailable; AI assistance will degrade", error=str(exc))
                notes.append(f"camera: unavailable ({exc})")
                # Returning None is correct: the vision service reports "no
                # camera" and fusion falls back to EMG-only control.
                return None
            return camera

        raise ConfigurationError(f"unknown camera driver '{driver}'")
