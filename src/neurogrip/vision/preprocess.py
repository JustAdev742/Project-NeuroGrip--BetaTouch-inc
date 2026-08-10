"""Image preprocessing.

Two implementations of every operation: a NumPy fast path when it is installed,
and a pure-Python fallback. The fallback exists so the pipeline runs (slowly, on
small images) in CI and on a stripped target — not because it is a good idea to
resize a 720p frame in a Python loop.

Letterboxing (resize preserving aspect ratio, pad the remainder) is used rather
than a plain stretch: a distorted image changes an object's apparent aspect
ratio, which is one of the cues the grasp planner uses to tell an upright bottle
from a lying one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..hal.camera.base import Frame, PixelFormat

__all__ = ["LetterboxInfo", "letterbox", "normalise_pixels", "resize_nearest", "to_gray"]


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    """Geometry of a letterbox transform, needed to map coordinates back.

    A model predicts in *padded* image space; every coordinate it returns must be
    un-padded before it can be compared with anything from the original frame.
    :meth:`to_source` does that.
    """

    scale: float
    pad_x: int
    pad_y: int
    target_width: int
    target_height: int
    source_width: int
    source_height: int

    def to_source(self, x: float, y: float) -> tuple[float, float]:
        """Map normalised model-space coordinates back to normalised source space."""
        px = x * self.target_width - self.pad_x
        py = y * self.target_height - self.pad_y
        sx = px / max(1e-6, self.source_width * self.scale)
        sy = py / max(1e-6, self.source_height * self.scale)
        return (min(1.0, max(0.0, sx)), min(1.0, max(0.0, sy)))

    def scale_length(self, length: float) -> float:
        """Map a normalised length from model space to source space."""
        return length * self.target_width / max(1e-6, self.source_width * self.scale)

    def contains_model_point(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Whether a normalised model-space point lies on real image content.

        The letterbox border is a hard synthetic edge between the image and the
        pad value. Edge-based detectors respond strongly to it, so any consumer
        that looks for structure must exclude the padding — otherwise the
        strongest "graspable" region in the frame is the border of the frame.
        """
        px = x * self.target_width
        py = y * self.target_height
        content_w = self.source_width * self.scale
        content_h = self.source_height * self.scale
        pad_margin_x = margin * content_w
        pad_margin_y = margin * content_h
        return (
            self.pad_x + pad_margin_x <= px <= self.pad_x + content_w - pad_margin_x
            and self.pad_y + pad_margin_y <= py <= self.pad_y + content_h - pad_margin_y
        )


def _try_numpy():
    try:
        import numpy as np

        return np
    except ImportError:
        return None


def to_gray(frame: Frame) -> list[int]:
    """Convert a frame to an 8-bit greyscale row-major list.

    Uses ITU-R BT.601 luma weights, which is what OpenCV and most detectors
    assume, so the classical fallback and a trained model see the same image.
    """
    if frame.pixel_format is PixelFormat.GRAY8:
        return list(frame.data)
    if frame.pixel_format is PixelFormat.JPEG:
        raise ValueError("decode the JPEG before converting to greyscale")

    data = frame.data
    swap = frame.pixel_format is PixelFormat.BGR888
    np = _try_numpy()
    if np is not None:
        array = np.frombuffer(data, dtype=np.uint8).reshape((frame.height, frame.width, 3))
        if swap:
            array = array[:, :, ::-1]
        luma = (
            array[:, :, 0].astype(np.float32) * 0.299
            + array[:, :, 1].astype(np.float32) * 0.587
            + array[:, :, 2].astype(np.float32) * 0.114
        )
        return luma.astype(np.uint8).reshape(-1).tolist()

    out: list[int] = []
    for offset in range(0, len(data), 3):
        r, g, b = data[offset], data[offset + 1], data[offset + 2]
        if swap:
            r, b = b, r
        out.append(int(0.299 * r + 0.587 * g + 0.114 * b))
    return out


def resize_nearest(
    pixels: list[int], src_w: int, src_h: int, dst_w: int, dst_h: int
) -> list[int]:
    """Nearest-neighbour resize of a single-channel image.

    Nearest rather than bilinear: the fallback path is already the slow path, and
    for a graspability heatmap derived from edge density the interpolation
    quality is not the limiting factor.
    """
    if src_w == dst_w and src_h == dst_h:
        return pixels
    out = [0] * (dst_w * dst_h)
    x_ratio = src_w / dst_w
    y_ratio = src_h / dst_h
    for y in range(dst_h):
        sy = min(src_h - 1, int(y * y_ratio))
        row = sy * src_w
        out_row = y * dst_w
        for x in range(dst_w):
            sx = min(src_w - 1, int(x * x_ratio))
            out[out_row + x] = pixels[row + sx]
    return out


def letterbox(
    pixels: list[int], src_w: int, src_h: int, dst_w: int, dst_h: int, pad_value: int = 114
) -> tuple[list[int], LetterboxInfo]:
    """Resize preserving aspect ratio and pad to ``dst_w`` × ``dst_h``.

    ``pad_value`` of 114 matches the convention used by the YOLO family and by
    the HGGD reference implementation, so a model trained with that padding sees
    what it expects.
    """
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = resize_nearest(pixels, src_w, src_h, new_w, new_h)

    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2
    canvas = [pad_value] * (dst_w * dst_h)
    for y in range(new_h):
        dst_row = (y + pad_y) * dst_w + pad_x
        src_row = y * new_w
        canvas[dst_row : dst_row + new_w] = resized[src_row : src_row + new_w]

    return canvas, LetterboxInfo(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        target_width=dst_w,
        target_height=dst_h,
        source_width=src_w,
        source_height=src_h,
    )


def normalise_pixels(pixels: list[int], *, mean: float = 0.0, scale: float = 1.0 / 255.0) -> list[float]:
    """Scale 8-bit pixels into the float range a network expects."""
    return [(p - mean) * scale for p in pixels]


def frame_to_model_input(
    frame: Frame, width: int, height: int
) -> tuple[list[int], LetterboxInfo]:
    """Full preprocessing path: colour convert, letterbox, return with geometry."""
    gray = to_gray(frame)
    return letterbox(gray, frame.width, frame.height, width, height)
