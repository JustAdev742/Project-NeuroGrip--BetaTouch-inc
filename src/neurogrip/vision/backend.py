"""Vision backend interface and registry.

The requirement is explicit: *do not tightly couple the software to one model.*
This module is how that is enforced. A backend is anything that turns a
:class:`~neurogrip.hal.camera.base.Frame` into a
:class:`~neurogrip.vision.types.VisionResult` and declares what it can do.

Consequences for the rest of the system:

* the pipeline never names a model — it asks the registry for whatever
  ``[vision] backend`` says;
* a backend that cannot produce grasps (a plain detector) simply omits the
  ``GRASP`` capability, and :mod:`neurogrip.ai.grasp` falls back to the
  heuristic planner;
* a backend that fails to load is replaced by the null backend, and the hand
  keeps working under direct EMG control.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..core.config import Config
from ..core.logging import get_logger
from ..hal.camera.base import Frame
from .types import VisionCapability, VisionResult

__all__ = [
    "BackendInfo",
    "VisionBackend",
    "available_backends",
    "create_backend",
    "register_backend",
]

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """Identity and runtime status of a loaded backend."""

    name: str
    version: str = ""
    capabilities: VisionCapability = VisionCapability.NONE
    #: Inference runtime actually in use (``onnxruntime``, ``tflite``, ``classical``).
    runtime: str = ""
    model_path: str = ""
    input_width: int = 0
    input_height: int = 0
    #: Set when the backend is running in a reduced mode (e.g. no weights found).
    degraded_reason: str = ""

    @property
    def is_degraded(self) -> bool:
        return bool(self.degraded_reason)

    def __str__(self) -> str:  # pragma: no cover - display helper
        suffix = f" (degraded: {self.degraded_reason})" if self.degraded_reason else ""
        return f"{self.name} [{self.runtime}] {self.capabilities.describe()}{suffix}"


@runtime_checkable
class VisionBackend(Protocol):
    """Frame → :class:`VisionResult`."""

    def initialize(self) -> None:
        """Load weights and warm up. May raise ``ModelLoadError``."""
        ...

    def shutdown(self) -> None:
        """Release model resources. Idempotent, must not raise."""
        ...

    def info(self) -> BackendInfo: ...

    @property
    def capabilities(self) -> VisionCapability: ...

    def process(self, frame: Frame) -> VisionResult:
        """Run inference on one frame.

        Must not raise for ordinary inference failures: return a
        :class:`VisionResult` with ``error`` set. Vision is an *assistive* input,
        and a model exception must never propagate into the control loop.
        """
        ...


#: name -> factory(config, **kwargs) -> VisionBackend
_REGISTRY: dict[str, Callable[..., VisionBackend]] = {}


def register_backend(name: str, factory: Callable[..., VisionBackend]) -> None:
    """Register a backend factory under ``name``."""
    _REGISTRY[name] = factory


def available_backends() -> tuple[str, ...]:
    """Names of all registered backends (shown in Settings ▸ Vision)."""
    _ensure_builtins()
    return tuple(sorted(_REGISTRY))


def create_backend(name: str, config: Config, **kwargs: object) -> VisionBackend:
    """Instantiate a backend by name.

    Unknown names fall back to the null backend with a warning rather than
    aborting startup: an unrecognised model string in a config file must not
    prevent someone from using their hand.
    """
    _ensure_builtins()
    factory = _REGISTRY.get(name)
    if factory is None:
        log.warning(
            "unknown vision backend; continuing without vision",
            requested=name,
            available=list(_REGISTRY),
        )
        factory = _REGISTRY["null"]
    return factory(config=config, **kwargs)


_builtins_loaded = False


def _ensure_builtins() -> None:
    """Import the bundled backends on first use.

    Deferred so that importing :mod:`neurogrip.vision` does not drag in
    ``onnxruntime`` on a machine that will only ever run the mock backend.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    _builtins_loaded = True
    from .backends import (  # noqa: F401  (import registers)
        anygrasp,
        hggd_mcu,
        mock,
        null,
        onnx_detector,
        replay,
    )
