"""User profiles and settings persistence.

The configuration loader already reads a per-user layer. Nothing ever wrote one,
so every adjustment a user made — theme, font size, reduce-motion, preferred
mode, grip speed — was lost on the next power cycle. For an assistive device
that is not a missing convenience: the settings that matter most here are
accessibility settings, and asking someone to re-apply them at every boot,
through an interface they configured *because* the default was hard for them to
use, is the wrong way round.

A profile is per *person*, not per device:

* **EMG calibration is personal.** Muscle signals differ between users and
  between sessions. Two people sharing a device need two calibrations, and
  switching users must not mean recalibrating from scratch.
* **Servo calibration is not.** Tendon slack belongs to the hand, so it lives
  with the hardware (:mod:`neurogrip.control.servo_calibration`) and is
  deliberately *not* stored here.
* **Preferences are personal.** Theme, accessibility, default mode.

Stored as JSON rather than TOML because the runtime core is standard-library
only: ``tomllib`` reads TOML but nothing in the standard library writes it, and
hand-rolling a serialiser to save a settings dictionary would be a poor trade.
Hand-edited configuration stays TOML; machine-written state is JSON, the same
split the calibration and training-statistics files already use.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import deep_merge
from .errors import ConfigurationError
from .logging import get_logger

__all__ = ["ProfileStore", "UserProfile"]

log = get_logger(__name__)

#: Profile names become file names, so they are restricted rather than escaped.
_VALID_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")

#: Settings a profile is allowed to carry. Anything else is refused on save.
#: Without this a UI bug could persist an arbitrary configuration override —
#: including a safety limit — into a file that is loaded on every boot.
ALLOWED_SETTING_PREFIXES: tuple[str, ...] = (
    "ui.theme",
    "ui.accessibility.",
    "modes.default",
    "emg.calibration_path",
    "training.",
)


def _is_allowed(path: str) -> bool:
    return any(
        path == prefix.rstrip(".") or path.startswith(prefix)
        for prefix in ALLOWED_SETTING_PREFIXES
    )


@dataclass(slots=True)
class UserProfile:
    """One person's saved settings."""

    name: str
    display_name: str = ""
    #: Dotted-path settings, merged over the file configuration at startup.
    settings: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    notes: str = ""
    version: int = 1

    def __post_init__(self) -> None:
        if not _VALID_NAME.match(self.name):
            raise ConfigurationError(
                f"invalid profile name {self.name!r}: use lower-case letters, "
                "digits, dash and underscore (max 32 characters)"
            )
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").replace("-", " ").title()

    def get(self, path: str, default: Any = None) -> Any:
        return self.settings.get(path, default)

    def set(self, path: str, value: Any) -> None:
        """Record one setting. Raises for a path a profile may not carry."""
        if not _is_allowed(path):
            raise ConfigurationError(
                f"{path} cannot be stored in a user profile; "
                f"allowed prefixes: {', '.join(ALLOWED_SETTING_PREFIXES)}"
            )
        self.settings[path] = value
        self.updated_at = time.time()

    def clear(self, path: str) -> None:
        self.settings.pop(path, None)
        self.updated_at = time.time()

    def as_overlay(self) -> dict[str, Any]:
        """Expand the dotted settings into a nested mapping for the config loader."""
        overlay: dict[str, Any] = {}
        for path, value in self.settings.items():
            node = overlay
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return overlay

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "name": self.name,
            "display_name": self.display_name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "notes": self.notes,
            "settings": dict(sorted(self.settings.items())),
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserProfile:
        return cls(
            name=str(data["name"]),
            display_name=str(data.get("display_name", "")),
            settings=dict(data.get("settings", {})),
            created_at=float(data.get("created_at", 0.0)),
            updated_at=float(data.get("updated_at", 0.0)),
            notes=str(data.get("notes", "")),
            version=int(data.get("version", 1)),
        )


class ProfileStore:
    """Directory of user profiles, with an active selection.

    Every write is atomic. A profile file is read on every boot, so a truncated
    one would leave the device unable to start — the failure mode that made
    atomic writes worth it for the calibration files applies here too.
    """

    def __init__(self, root: Path | str = "var/profiles") -> None:
        self._root = Path(root)
        self._active_marker = self._root / "active"

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, name: str) -> Path:
        if not _VALID_NAME.match(name):
            raise ConfigurationError(f"invalid profile name: {name!r}")
        return self._root / f"{name}.json"

    # -- enumeration ----------------------------------------------------------

    def names(self) -> tuple[str, ...]:
        if not self._root.is_dir():
            return ()
        return tuple(sorted(p.stem for p in self._root.glob("*.json")))

    def exists(self, name: str) -> bool:
        return self._path(name).exists()

    # -- read / write ---------------------------------------------------------

    def load(self, name: str) -> UserProfile:
        path = self._path(name)
        if not path.exists():
            raise ConfigurationError(f"no such profile: {name}")
        try:
            return UserProfile.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ConfigurationError(f"invalid profile file {path}: {exc}") from exc

    def save(self, profile: UserProfile) -> None:
        rejected = [p for p in profile.settings if not _is_allowed(p)]
        if rejected:
            raise ConfigurationError(
                f"profile {profile.name} carries settings it may not: {', '.join(rejected)}"
            )
        path = self._path(profile.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not profile.created_at:
            profile.created_at = time.time()
        profile.updated_at = time.time()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)
        log.info("profile saved", profile=profile.name, settings=len(profile.settings))

    def create(self, name: str, *, display_name: str = "") -> UserProfile:
        if self.exists(name):
            raise ConfigurationError(f"profile already exists: {name}")
        profile = UserProfile(name=name, display_name=display_name, created_at=time.time())
        self.save(profile)
        return profile

    def delete(self, name: str) -> None:
        """Remove a profile. The active selection falls back to the default."""
        path = self._path(name)
        if path.exists():
            path.unlink()
            log.info("profile deleted", profile=name)
        if self.active_name() == name:
            self._active_marker.unlink(missing_ok=True)

    # -- active selection -----------------------------------------------------

    def active_name(self) -> str | None:
        """Name of the selected profile, or ``None`` if none is selected."""
        if not self._active_marker.exists():
            return None
        name = self._active_marker.read_text(encoding="utf-8").strip()
        return name if name and self.exists(name) else None

    def set_active(self, name: str) -> None:
        if not self.exists(name):
            raise ConfigurationError(f"no such profile: {name}")
        self._root.mkdir(parents=True, exist_ok=True)
        temporary = self._active_marker.with_suffix(".tmp")
        temporary.write_text(name, encoding="utf-8")
        temporary.replace(self._active_marker)
        log.info("active profile changed", profile=name)

    def active(self, *, create_default: bool = True) -> UserProfile | None:
        """The selected profile.

        With ``create_default`` set — the normal case at startup — a device with
        no profiles gets one rather than running with nowhere to save settings.
        """
        name = self.active_name()
        if name is not None:
            try:
                return self.load(name)
            except ConfigurationError as exc:
                log.warning("active profile is unreadable; ignoring it", error=str(exc))

        # Try the remaining profiles in turn. A corrupt file must cost the user
        # their preferences, never their ability to start the hand.
        for candidate in self.names():
            if candidate == name:
                continue
            try:
                profile = self.load(candidate)
            except ConfigurationError as exc:
                log.warning("skipping unreadable profile", profile=candidate, error=str(exc))
                continue
            self.set_active(candidate)
            return profile

        if not create_default:
            return None
        # Overwrite rather than create: reaching here means every profile on disk
        # is unreadable, including any existing "default", and refusing to start
        # over a corrupt preferences file would be the wrong failure.
        profile = UserProfile(name="default", display_name="Default", created_at=time.time())
        self.save(profile)
        self.set_active(profile.name)
        return profile

    def overlay(self) -> dict[str, Any]:
        """Configuration overlay for the active profile, or an empty mapping."""
        profile = self.active(create_default=False)
        if profile is None:
            return {}
        return profile.as_overlay()

    def merge_into(self, base: dict[str, Any]) -> dict[str, Any]:
        """Merge the active profile's overlay over ``base``."""
        return deep_merge(base, self.overlay())
