"""Configuration bootstrap.

Resolves the configuration layers and installs logging before anything else runs,
so that even a failure during hardware construction is properly recorded.

Layer order (later wins)::

    config/default.toml  →  profile file  →  user profile  →  environment  →  --set
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..core.config import Config, ConfigLoader
from ..core.logging import configure_logging, get_logger

__all__ = ["DEFAULT_CONFIG", "find_project_root", "load_configuration"]

DEFAULT_CONFIG = "config/default.toml"

log = get_logger(__name__)


def find_project_root(start: Path | None = None) -> Path:
    """Locate the directory containing ``config/``.

    Walks upwards from ``start`` (or the installed package location) so the
    application works when run from a checkout, from an installed wheel with a
    sibling ``config`` directory, or from a systemd unit with an arbitrary
    working directory.
    """
    candidates = []
    if start is not None:
        candidates.append(Path(start).resolve())
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parents[3])

    for candidate in candidates:
        current = candidate
        for _ in range(5):
            if (current / "config" / "default.toml").exists():
                return current
            if current.parent == current:
                break
            current = current.parent
    return Path.cwd()


def load_configuration(
    *,
    config_path: str | Path | None = None,
    profile: str | None = None,
    overrides: Sequence[str] = (),
    root: Path | None = None,
    use_environment: bool = True,
) -> Config:
    """Build the merged configuration and configure logging from it."""
    project_root = root or find_project_root()
    loader = ConfigLoader(project_root)

    loader.add_file(DEFAULT_CONFIG, required=False)
    # Grip presets and object affordances are separate files because they are
    # tuning data a clinician may edit, not system configuration.
    loader.add_file("config/grasps.toml", required=False)
    loader.add_file("config/affordances.toml", required=False)

    if config_path:
        loader.add_file(config_path, required=True)
    if profile:
        loader.add_file(f"config/{profile}.toml", required=True)

    # A per-user profile lives outside the repository so it survives updates.
    loader.add_file("var/user.toml", required=False)

    if use_environment:
        loader.add_environment()
    loader.add_overrides(list(overrides))

    config = loader.build()

    logging_section = config.section("logging")
    configure_logging(
        level=logging_section.get_str("level", "INFO"),
        console=logging_section.get_bool("console", True),
        file_path=logging_section.get_str("file", "") or None,
        max_bytes=logging_section.get_int("max_bytes", 4_000_000),
        backups=logging_section.get_int("backups", 3),
        buffer_capacity=logging_section.get_int("buffer", 1000),
        quiet_loggers=tuple(logging_section.get_list("quiet", [])),
    )

    log.info(
        "configuration loaded",
        root=str(project_root),
        sources=list(config.sources) or ["built-in defaults"],
    )
    return config
