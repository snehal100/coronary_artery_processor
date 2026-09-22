"""Background estimation and anatomical structure suppression.

Targets low-frequency background, ribs, spine and large lung fields
while attempting to preserve thin high-frequency vessel structures.
"""
from __future__ import annotations

import numpy as np
import cv2
from .config import PipelineConfig, DEFAULT_CONFIG


def estimate_background(img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Estimate large-scale background via heavy Gaussian / box blur.

    Fast approximation of low-frequency illumination and soft-tissue.
    """
    k = cfg.bg_ksize
    if k % 2 == 0:
        k += 1
    # OpenCV GaussianBlur is highly optimized
    return cv2.GaussianBlur(img, (k, k), cfg.bg_sigma)


def subtract_background(
    img: np.ndarray, bg: np.ndarray | None = None, cfg: PipelineConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Subtract estimated background; result centered around zero then shifted."""
    if bg is None:
        bg = estimate_background(img, cfg)
    residual = img - bg
    # Shift to non-negative for later processing
    residual = residual - residual.min()
    return residual


def morphological_background_suppression(
    img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Black-hat / top-hat to suppress larger structures while keeping vessels.

    For X-ray angio, contrast-filled vessels often appear darker than
    surrounding tissue → white top-hat on inverted image or black-hat
    on original can isolate dark tubular structures.
    """
    k = cfg.morph_kernel_size
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # Work on float but OpenCV morph needs uint8 or similar; scale temporarily
    mn, mx = float(img.min()), float(img.max())
    if mx <= mn:
        return img.copy()
    scaled = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)

    # Black-hat: extracts dark structures smaller than kernel
    blackhat = cv2.morphologyEx(scaled, cv2.MORPH_BLACKHAT, kernel)
    # Top-hat: bright structures
    tophat = cv2.morphologyEx(scaled, cv2.MORPH_TOPHAT, kernel)

    # Combine: emphasize dark vessels (blackhat) and suppress large bright bg
    enhanced = cv2.add(scaled, tophat)
    enhanced = cv2.subtract(enhanced, blackhat)  # optional; tune

    # Prefer black-hat result for dark vessel isolation
    # Return normalized residual focused on dark structures
    result = blackhat.astype(np.float32) / 255.0 * (mx - mn) + mn
    return result


def suppress_large_structures(
    img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Combined background subtraction + morphological attenuation of ribs/spine.

    Fast path:
    1. Large Gaussian background estimate
    2. Residual
    3. Optional morphological cleanup of remaining large blobs
    """
    return suppress_anatomical_background(img, cfg)[0]


def suppress_anatomical_background(
    img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG
) -> tuple[np.ndarray, np.ndarray]:
    """Suppress broad anatomy and return the processed image plus a mask.

    The large Gaussian removes slowly varying lung/soft-tissue illumination.
    A large elliptical opening estimates bright rib/spine structures without
    erasing narrow vessels. The mask is diagnostic and is intentionally kept
    separate from vessel enhancement so the two operations remain inspectable.
    """
    img = np.asarray(img, dtype=np.float32)
    low_frequency = estimate_background(img, cfg)
    high_pass = img - low_frequency

    k = max(3, int(cfg.anatomy_open_ksize) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Box filtering is substantially faster than a morphology opening on
    # CPU while still estimating broad bright anatomy for attenuation.
    bright_anatomy = cv2.boxFilter(img, -1, (k, k), normalize=True)
    bright_anatomy = np.maximum(bright_anatomy - low_frequency, 0.0)
    anatomy_mask = np.clip(bright_anatomy * (1.0 + 2.0 * cfg.anatomy_suppression), 0.0, 1.0)

    # Keep dark vessel detail while attenuating bright broad structures.
    processed = high_pass - cfg.anatomy_suppression * bright_anatomy
    processed = cv2.GaussianBlur(processed, (3, 3), 0.6)
    processed -= processed.min()
    peak = float(processed.max())
    if peak > 0.0:
        processed /= peak
    return processed.astype(np.float32), anatomy_mask.astype(np.float32)
