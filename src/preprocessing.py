"""Preprocessing: validation, normalization, noise reduction."""
from __future__ import annotations

import numpy as np
import cv2
from .config import PipelineConfig, DEFAULT_CONFIG
from .io_utils import to_float32, robust_normalize, validate_16bit


def validate_and_prepare(img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Validate input and convert to float32 working representation."""
    ok, msg = validate_16bit(img)
    if not ok:
        raise ValueError(f"Invalid input image: {msg}")
    return to_float32(img)


def denoise(img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Lightweight noise reduction that preserves thin vessels.

    Uses small Gaussian blur by default (fast). Bilateral is optional
    but more expensive and disabled for real-time targets.
    """
    if cfg.use_bilateral:
        # Bilateral works on 8-bit; convert carefully
        vis = robust_normalize(img)
        u8 = (vis * 255).astype(np.uint8)
        den = cv2.bilateralFilter(
            u8, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space
        )
        # Map back approximately
        return den.astype(np.float32) / 255.0 * (img.max() - img.min()) + img.min()

    k = cfg.gaussian_ksize
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(img, (k, k), cfg.gaussian_sigma)


def normalize_for_processing(
    img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Robust percentile normalization to [0, 1] for subsequent stages."""
    return robust_normalize(img, cfg.percentile_low, cfg.percentile_high)
