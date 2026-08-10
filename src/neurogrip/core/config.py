"""Layered configuration.

Configuration is TOML, parsed with the standard library's :mod:`tomllib`. TOML was
chosen over YAML deliberately: it is in the standard library (no dependency on the
target device), it has an unambiguous type system, and it cannot express the
surprising implicit conversions that make YAML configs a source of field faults.

Layers are merged in increasing order of precedence:

1. ``config/default.toml``           — the baseline shipped with the software
2. a profile file, e.g. ``config/hardware.toml`` or ``config/simulation.toml``
3. a user profile from persistent storage (per-amputee tuning)
4. environment variables ``NEUROGRIP__SECTION__KEY=value``
5. command-line ``--set section.key=value`` overrides

The merged result is wrapped in a read-only :class:`Config` that fails loudly on a
missing required key, because silently defaulting a servo limit is a safety issue.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from .errors import ConfigurationError

__all__ = ["Config", "ConfigLoader", "bind_dataclass", "deep_merge"]

T = TypeVar("T")

ENV_PREFIX = "NEUROGRIP__"
_MISSING = object()


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``, returning a new dict.

    Nested tables merge key-by-key; every other type (including lists) is
    replaced wholesale. Replacing lists rather than concatenating them is the
    behaviour that makes "override the enabled sensor list" work as expected.
    """
    result: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _parse_scalar(raw: str) -> Any:
    """Parse an environment/CLI value using TOML scalar rules."""
    try:
        return tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw  # bare strings such as /dev/ttyUSB0 need no quoting


class Config:
    """Immutable view over the merged configuration tree.

    Access uses dotted paths::

        config.get("control.rate_hz", 200)
        config.require("servo.port")
        motion = config.section("control.motion")
    """

    __slots__ = ("_data", "_prefix", "_sources")

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        sources: Sequence[str] = (),
        prefix: str = "",
    ) -> None:
        self._data = dict(data)
        self._sources = tuple(sources)
        self._prefix = prefix

    # -- reading --------------------------------------------------------------

    def get(self, path: str, default: Any = None) -> Any:
        """Value at ``path``, or ``default`` when absent."""
        value = self._lookup(path)
        return default if value is _MISSING else value

    def require(self, path: str) -> Any:
        """Value at ``path``; raises :class:`ConfigurationError` when absent."""
        value = self._lookup(path)
        if value is _MISSING:
            raise ConfigurationError(
                f"missing required configuration key '{self._qualify(path)}'",
                context={"sources": list(self._sources)},
            )
        return value

    def get_float(self, path: str, default: float | None = None) -> float:
        return float(self._typed(path, default, (int, float), "number"))

    def get_int(self, path: str, default: int | None = None) -> int:
        value = self._typed(path, default, (int, float), "integer")
        if isinstance(value, float) and not value.is_integer():
            raise ConfigurationError(f"'{self._qualify(path)}' must be an integer, got {value}")
        return int(value)

    def get_bool(self, path: str, default: bool | None = None) -> bool:
        return bool(self._typed(path, default, (bool,), "boolean"))

    def get_str(self, path: str, default: str | None = None) -> str:
        return str(self._typed(path, default, (str,), "string"))

    def get_list(self, path: str, default: Sequence[Any] | None = None) -> list[Any]:
        value = self._typed(path, list(default) if default is not None else None, (list,), "array")
        return list(value)

    def _typed(self, path: str, default: Any, types: tuple[type, ...], label: str) -> Any:
        value = self._lookup(path)
        if value is _MISSING:
            if default is None:
                raise ConfigurationError(f"missing required {label} '{self._qualify(path)}'")
            return default
        if not isinstance(value, types) or (types != (bool,) and isinstance(value, bool)):
            raise ConfigurationError(
                f"'{self._qualify(path)}' must be a {label}, got {type(value).__name__}"
            )
        return value

    def section(self, path: str, *, required: bool = False) -> Config:
        """Sub-configuration rooted at ``path``; empty when absent unless required."""
        value = self._lookup(path)
        if value is _MISSING:
            if required:
                raise ConfigurationError(f"missing required section '{self._qualify(path)}'")
            value = {}
        if not isinstance(value, Mapping):
            raise ConfigurationError(f"'{self._qualify(path)}' is not a section")
        return Config(value, sources=self._sources, prefix=self._qualify(path))

    def sections(self, path: str) -> dict[str, Config]:
        """Every direct sub-table of ``path`` — used for per-device tables."""
        parent = self.section(path)
        return {
            key: parent.section(key) for key, value in parent.items() if isinstance(value, Mapping)
        }

    def _lookup(self, path: str) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return _MISSING
            node = node[part]
        return node

    def _qualify(self, path: str) -> str:
        return f"{self._prefix}.{path}" if self._prefix else path

    # -- mapping-ish protocol -------------------------------------------------

    def __contains__(self, path: str) -> bool:
        return self._lookup(path) is not _MISSING

    def __getitem__(self, path: str) -> Any:
        return self.require(path)

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def as_dict(self) -> dict[str, Any]:
        """Deep copy of the underlying tree (safe to mutate)."""
        import copy

        return copy.deepcopy(self._data)

    @property
    def sources(self) -> tuple[str, ...]:
        """Files and overlays that contributed, in application order."""
        return self._sources

    def with_overlay(self, overlay: Mapping[str, Any], *, source: str = "overlay") -> Config:
        """Return a new config with ``overlay`` merged on top."""
        return Config(
            deep_merge(self._data, overlay),
            sources=(*self._sources, source),
            prefix=self._prefix,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Config(prefix={self._prefix!r}, keys={sorted(self._data)})"


def bind_dataclass(cls: type[T], config: Config, *, path: str = "") -> T:
    """Instantiate dataclass ``cls`` from a config section.

    Only fields present in the configuration are overridden; everything else keeps
    its dataclass default. Unknown keys raise, which catches typos in hand-edited
    config files at startup instead of at 200 Hz in the control loop.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    section = config.section(path) if path else config
    known = {f.name for f in dataclass_fields(cls)}
    unknown = {k for k, v in section.items() if k not in known and not isinstance(v, Mapping)}
    if unknown:
        raise ConfigurationError(
            f"unknown configuration keys for {cls.__name__}: {sorted(unknown)}",
            context={"section": path or "<root>"},
        )
    kwargs = {name: section.get(name) for name in known if name in section}
    return cls(**kwargs)  # type: ignore[return-value]


class ConfigLoader:
    """Builds a :class:`Config` from files, environment and explicit overrides."""

    def __init__(self, root: Path | str | None = None) -> None:
        #: Directory that relative config paths are resolved against.
        self.root = Path(root) if root else Path.cwd()
        self._data: dict[str, Any] = {}
        self._sources: list[str] = []

    def add_file(self, path: Path | str, *, required: bool = True) -> ConfigLoader:
        """Merge a TOML file. Missing optional files are skipped silently."""
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self.root / resolved
        if not resolved.exists():
            if required:
                raise ConfigurationError(f"configuration file not found: {resolved}")
            return self
        try:
            with resolved.open("rb") as handle:
                parsed = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigurationError(f"invalid TOML in {resolved}: {exc}") from exc
        self._data = deep_merge(self._data, parsed)
        self._sources.append(str(resolved))
        return self

    def add_mapping(self, data: Mapping[str, Any], *, source: str = "mapping") -> ConfigLoader:
        """Merge an in-memory mapping (used by tests and the user-profile store)."""
        self._data = deep_merge(self._data, data)
        self._sources.append(source)
        return self

    def add_environment(self, environ: Mapping[str, str] | None = None) -> ConfigLoader:
        """Merge ``NEUROGRIP__SECTION__KEY`` variables from the environment."""
        env = os.environ if environ is None else environ
        overlay: dict[str, Any] = {}
        found = False
        for key, raw in env.items():
            if not key.startswith(ENV_PREFIX):
                continue
            found = True
            path = key[len(ENV_PREFIX) :].lower().split("__")
            _assign(overlay, path, _parse_scalar(raw))
        if found:
            self._data = deep_merge(self._data, overlay)
            self._sources.append("environment")
        return self

    def add_overrides(self, overrides: Sequence[str]) -> ConfigLoader:
        """Merge ``section.key=value`` strings from the command line."""
        if not overrides:
            return self
        overlay: dict[str, Any] = {}
        for item in overrides:
            if "=" not in item:
                raise ConfigurationError(f"override must be key=value, got '{item}'")
            key, _, raw = item.partition("=")
            _assign(overlay, key.strip().split("."), _parse_scalar(raw.strip()))
        self._data = deep_merge(self._data, overlay)
        self._sources.append("cli")
        return self

    def build(self) -> Config:
        """Produce the immutable merged configuration."""
        return Config(self._data, sources=tuple(self._sources))


def _assign(tree: dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Write ``value`` into ``tree`` at the nested ``path``, creating tables."""
    node = tree
    for part in path[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[path[-1]] = value
