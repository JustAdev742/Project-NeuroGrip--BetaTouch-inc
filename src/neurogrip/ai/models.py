"""Model registry: locating, verifying and describing model files.

Weights are not in version control. This module is how the rest of the system
finds them, checks they are the ones expected, and reports honestly when they are
missing — rather than each backend inventing its own path handling.

A manifest (``models/manifest.toml``) records what should be present::

    [models.hggd_mcu]
    file = "hggd_mcu/hggd_mcu_int8.onnx"
    sha256 = "…"
    version = "1.0.0"
    input = "1x1x128x160"
    description = "Heatmap-guided grasp detector, int8"
    source = "https://…"

Checksums matter here beyond tidiness: silently loading a *different* model than
the one the affordance thresholds were tuned against would change the hand's
behaviour with no visible cause.
"""

from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..core.logging import get_logger

__all__ = ["ModelEntry", "ModelRegistry", "ModelStatus"]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Result of checking one model."""

    name: str
    path: Path
    present: bool
    size_bytes: int = 0
    checksum_ok: bool | None = None
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.present and self.checksum_ok is not False

    def __str__(self) -> str:  # pragma: no cover - display helper
        if not self.present:
            return f"{self.name}: MISSING ({self.path})"
        if self.checksum_ok is False:
            return f"{self.name}: CHECKSUM MISMATCH ({self.path})"
        return f"{self.name}: ok ({self.size_bytes / 1e6:.1f} MB)"


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """A declared model."""

    name: str
    file: str
    sha256: str = ""
    version: str = ""
    input_shape: str = ""
    description: str = ""
    source: str = ""
    #: When false, absence is expected and reported as informational.
    required: bool = False


class ModelRegistry:
    """Resolves and verifies model files against a manifest."""

    def __init__(self, root: Path | str = "models", manifest: Path | str | None = None) -> None:
        self._root = Path(root)
        self._manifest_path = Path(manifest) if manifest else self._root / "manifest.toml"
        self._entries: dict[str, ModelEntry] = {}
        self._load_manifest()

    def _load_manifest(self) -> None:
        if not self._manifest_path.exists():
            log.debug("no model manifest found", path=str(self._manifest_path))
            return
        try:
            with self._manifest_path.open("rb") as handle:
                data = tomllib.load(handle)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            log.warning("could not read model manifest", path=str(self._manifest_path), error=str(exc))
            return

        for name, entry in data.get("models", {}).items():
            if not isinstance(entry, dict):
                continue
            self._entries[name] = ModelEntry(
                name=name,
                file=str(entry.get("file", "")),
                sha256=str(entry.get("sha256", "")),
                version=str(entry.get("version", "")),
                input_shape=str(entry.get("input", "")),
                description=str(entry.get("description", "")),
                source=str(entry.get("source", "")),
                required=bool(entry.get("required", False)),
            )

    # -- lookup ---------------------------------------------------------------

    def path_for(self, name: str) -> Path | None:
        """Absolute path of a declared model, whether or not it exists."""
        entry = self._entries.get(name)
        if entry is None:
            return None
        return self._root / entry.file

    def entry(self, name: str) -> ModelEntry | None:
        return self._entries.get(name)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    # -- verification ---------------------------------------------------------

    def check(self, name: str, *, verify_checksum: bool = True) -> ModelStatus:
        """Check one model's presence and integrity."""
        entry = self._entries.get(name)
        if entry is None:
            return ModelStatus(
                name=name, path=self._root / name, present=False, detail="not declared in the manifest"
            )

        path = self._root / entry.file
        if not path.exists():
            return ModelStatus(
                name=name,
                path=path,
                present=False,
                detail="required model missing" if entry.required else "optional model not installed",
            )

        size = path.stat().st_size
        checksum_ok: bool | None = None
        if verify_checksum and entry.sha256:
            checksum_ok = _sha256(path) == entry.sha256.lower()
            if not checksum_ok:
                log.error("model checksum mismatch", model=name, path=str(path))

        return ModelStatus(
            name=name,
            path=path,
            present=True,
            size_bytes=size,
            checksum_ok=checksum_ok,
            detail=entry.description,
        )

    def check_all(self, *, verify_checksum: bool = False) -> list[ModelStatus]:
        """Check every declared model.

        Checksum verification is off by default because it reads whole files;
        the self-test enables it, startup does not.
        """
        return [self.check(name, verify_checksum=verify_checksum) for name in self.names]

    def missing_required(self) -> list[ModelStatus]:
        return [
            status
            for status in self.check_all()
            if not status.present and (self._entries[status.name].required)
        ]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()
