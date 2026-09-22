"""I/O utilities for 16-bit grayscale medical images."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image preserving 16-bit depth when present.

    Supports PNG, TIFF, and other formats readable by OpenCV.
    Returns uint16 array for 16-bit sources, otherwise original dtype.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # IMREAD_UNCHANGED preserves bit depth
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image (unsupported or corrupted): {path}")

    if img.ndim == 3:
        # Convert multi-channel to grayscale if needed
        if img.shape[2] == 3 or img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = img[:, :, 0]

    return img


def validate_16bit(img: np.ndarray) -> Tuple[bool, str]:
    """Validate that the image is 16-bit grayscale (or convertible).

    Returns (is_valid, message).
    Accepts uint16 and also float32/64 in [0, 65535] range for intermediate use.
    """
    if img is None or img.size == 0:
        return False, "Empty or None image"
    if img.ndim != 2:
        return False, f"Expected 2D grayscale, got shape {img.shape}"
    if img.dtype == np.uint16:
        return True, "uint16 grayscale"
    if img.dtype == np.uint8:
        return True, "uint8 grayscale (accepted and promoted)"
    if img.dtype == np.int16:
        if int(img.min()) < 0:
            return False, "int16 image contains negative values"
        return True, "int16 grayscale (accepted and promoted)"
    if img.dtype in (np.float32, np.float64):
        if not np.isfinite(img).all():
            return False, "floating-point image contains NaN or infinity"
        if float(img.min()) < 0.0 or float(img.max()) > 65535.0:
            return False, "floating-point image must be in [0, 65535]"
        return True, f"Accepted dtype {img.dtype} (will be handled)"
    return False, f"Unsupported dtype {img.dtype}"


def to_float32(img: np.ndarray) -> np.ndarray:
    """Convert to float32 in original intensity range (no forced 0-1)."""
    if img.dtype == np.float32:
        return img
    return img.astype(np.float32)


def robust_normalize(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.0,
    out_range: Tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Percentile-based robust normalization to avoid outlier clipping.

    Preserves relative vessel contrast better than min-max on noisy data.
    """
    img = to_float32(img)
    # A bounded sample keeps percentile windowing predictable for live frames.
    flat = img.reshape(-1)
    stride = max(1, flat.size // 65536)
    sample = flat[::stride]
    lo = np.percentile(sample, p_low)
    hi = np.percentile(sample, p_high)
    if hi <= lo:
        hi = lo + 1.0
    out = (img - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if out_range != (0.0, 1.0):
        out = out * (out_range[1] - out_range[0]) + out_range[0]
    return out


def to_uint16(img: np.ndarray, scale_from_01: bool = False) -> np.ndarray:
    """Convert processed float image back to uint16 for storage/display."""
    img = to_float32(img)
    if scale_from_01:
        img = np.clip(img, 0.0, 1.0) * 65535.0
    else:
        # Assume already in roughly 0-65535 or stretch
        mn, mx = img.min(), img.max()
        if mx > mn:
            img = (img - mn) / (mx - mn) * 65535.0
        else:
            img = np.zeros_like(img)
    return np.clip(img, 0, 65535).astype(np.uint16)


def to_uint8_vis(img: np.ndarray) -> np.ndarray:
    """Convert any image to uint8 for visualization (windowed)."""
    img = to_float32(img)
    mn, mx = np.percentile(img, 1), np.percentile(img, 99)
    if mx <= mn:
        mx = mn + 1.0
    out = (img - mn) / (mx - mn) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def save_image(path: Union[str, Path], img: np.ndarray) -> None:
    """Save image; for uint16 uses PNG which supports 16-bit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.dtype == np.uint16:
        cv2.imwrite(str(path), img)
    else:
        # Convert float/other to uint16 or uint8
        if img.dtype in (np.float32, np.float64):
            img = to_uint16(img, scale_from_01=(img.max() <= 1.0 + 1e-3))
        cv2.imwrite(str(path), img)
