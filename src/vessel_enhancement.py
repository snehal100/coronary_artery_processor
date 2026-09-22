"""Coronary vessel enhancement optimized for real-time CPU execution.

Uses a fast multi-scale morphological + Laplacian-of-Gaussian style
enhancement that approximates vesselness without expensive full
Hessian eigenvalue computation at every scale.
"""
from __future__ import annotations

import numpy as np
import cv2
from .config import PipelineConfig, DEFAULT_CONFIG


def multi_scale_vesselness(
    img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Fast multi-scale vessel enhancement.

    Strategy (speed-first):
    1. Normalize input.
    2. For each scale: Gaussian smooth → Laplacian (LoG-like) to highlight ridges.
    3. Take absolute value and max across scales.
    4. Light morphological close to reconnect thin vessels.
    This is substantially cheaper than full Frangi while still enhancing
    tubular structures of varying widths.
    """
    img_f = img.astype(np.float32)
    mn, mx = float(img_f.min()), float(img_f.max())
    if mx > mn:
        img_f = (img_f - mn) / (mx - mn)
    else:
        return np.zeros_like(img_f)

    vessel_map = np.zeros_like(img_f)
    for sigma in cfg.vessel_scales:
        ksize = max(3, int(4 * sigma + 1) | 1)
        blurred = cv2.GaussianBlur(img_f, (ksize, ksize), sigma)
        # Laplacian approximates second derivative (ridges / edges)
        lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        # Dark vessels produce positive response after sign flip of residual
        resp = np.abs(lap)
        vessel_map = np.maximum(vessel_map, resp)

    # Normalize response
    vmax = vessel_map.max()
    if vmax > 0:
        vessel_map /= vmax

    return vessel_map.astype(np.float32)


def enhance_vessels(
    img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Fast vessel enhancement combining multi-scale LoG + morphology."""
    vmap = multi_scale_vesselness(img, cfg)

    k = cfg.vessel_morph_size
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    vmap_u8 = (np.clip(vmap, 0, 1) * 255).astype(np.uint8)
    vmap_u8 = cv2.morphologyEx(vmap_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    return vmap_u8.astype(np.float32) / 255.0


def clahe_enhance(img: np.ndarray, cfg: PipelineConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Fast contrast enhancement, with optional CLAHE for offline use."""
    img_f = img.astype(np.float32)
    mn, mx = float(img_f.min()), float(img_f.max())
    if mx <= mn:
        return img_f
    u8 = ((img_f - mn) / (mx - mn) * 255).astype(np.uint8)
    if cfg.use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit, tileGridSize=cfg.clahe_tile_grid
        )
        enhanced = clahe.apply(u8)
    else:
        enhanced = cv2.equalizeHist(u8)
    return enhanced.astype(np.float32) / 255.0 * (mx - mn) + mn
