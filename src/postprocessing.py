"""Post-processing and output generation."""
from __future__ import annotations

import numpy as np
import cv2
from .config import PipelineConfig, DEFAULT_CONFIG
from .io_utils import to_uint16, to_uint8_vis


def fuse_enhancement(
    residual: np.ndarray,
    vessel_map: np.ndarray,
    original_norm: np.ndarray | None = None,
    alpha: float = 0.7,
) -> np.ndarray:
    """Fuse background-suppressed residual with vesselness map.

    Produces a final enhanced image emphasizing vessels.
    """
    # Normalize residual to [0,1]
    r = residual.astype(np.float32)
    rmin, rmax = r.min(), r.max()
    if rmax > rmin:
        r = (r - rmin) / (rmax - rmin)
    else:
        r = np.zeros_like(r)

    v = np.clip(vessel_map, 0.0, 1.0)

    # Weighted combination: residual provides anatomy context, vessel_map boosts vessels
    fused = (1.0 - alpha) * r + alpha * v
    # Optional slight original blend for continuity
    if original_norm is not None:
        fused = 0.85 * fused + 0.15 * original_norm
    return np.clip(fused, 0.0, 1.0)


def refine_mask(vessel_map: np.ndarray, min_area: int = 20) -> np.ndarray:
    """Simple connected-component cleanup of vessel map (optional)."""
    u8 = (np.clip(vessel_map, 0, 1) * 255).astype(np.uint8)
    _, binary = cv2.threshold(u8, 30, 255, cv2.THRESH_BINARY)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(u8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    return cleaned.astype(np.float32) / 255.0


def prepare_outputs(
    original: np.ndarray,
    processed: np.ndarray,
    enhanced: np.ndarray,
    cfg: PipelineConfig = DEFAULT_CONFIG,
) -> dict:
    """Generate three standard outputs: raw, processed, enhanced.

    Returns dict with uint16 and uint8 visualization versions.
    """
    return {
        "raw_uint16": to_uint16(original) if original.dtype != np.uint16 else original,
        "raw_vis": to_uint8_vis(original),
        "processed_uint16": to_uint16(processed, scale_from_01=(processed.max() <= 1.01)),
        "processed_vis": to_uint8_vis(processed),
        "enhanced_uint16": to_uint16(enhanced, scale_from_01=True),
        "enhanced_vis": to_uint8_vis(enhanced),
    }
