"""End-to-end coronary artery processing pipeline.

Designed for ≤36 ms/frame on CPU for typical 512×512 16-bit frames.
"""
from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

from .config import PipelineConfig, DEFAULT_CONFIG
from .io_utils import validate_16bit, to_float32, robust_normalize, to_uint8_vis
from .preprocessing import validate_and_prepare, denoise, normalize_for_processing
from .background_suppression import suppress_anatomical_background
from .vessel_enhancement import enhance_vessels, clahe_enhance, multi_scale_vesselness
from .postprocessing import fuse_enhancement


class CoronaryPipeline:
    """Stateful pipeline object; reuses kernels and config for repeated calls."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.cfg = config or DEFAULT_CONFIG
        self._warmed = False

    def warm_up(self, shape: Tuple[int, int] = (512, 512)) -> None:
        """Run a dummy pass to trigger OpenCV/NumPy initialization."""
        dummy = np.random.randint(0, 40000, shape, dtype=np.uint16)
        _ = self.process(dummy)
        self._warmed = True

    def process(
        self, img: np.ndarray, return_intermediates: bool = False
    ) -> Dict[str, np.ndarray]:
        """Process a single frame end-to-end.

        Returns dict with keys:
          - raw
          - processed   (after background / anatomical suppression)
          - enhanced    (final vessel-enhanced result)
          - latency_ms  (measured inside this call)
          and optional intermediates if requested.
        """
        t0 = time.perf_counter()

        # 1. Validation & float conversion
        work = validate_and_prepare(img, self.cfg)

        # Optional downscale for very large images (disabled by default)
        orig_shape = work.shape
        scale = 1.0
        if self.cfg.enable_downscale and max(orig_shape) > self.cfg.max_image_side:
            scale = self.cfg.max_image_side / max(orig_shape)
            new_w = int(orig_shape[1] * scale)
            new_h = int(orig_shape[0] * scale)
            work = cv2.resize(work, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 2. Noise reduction (lightweight)
        denoised = denoise(work, self.cfg)

        # 3. Robust normalization to [0,1]
        norm = normalize_for_processing(denoised, self.cfg)

        # 4. Background estimation + subtraction (ribs, spine, lungs, soft tissue)
        residual, anatomy_mask = suppress_anatomical_background(norm, self.cfg)

        # 5. Vessel enhancement (multi-scale vesselness)
        vessel_map = enhance_vessels(residual, self.cfg)

        # 6. Local contrast (CLAHE) on residual for additional detail
        residual_clahe = clahe_enhance(residual, self.cfg)

        # 7. Fuse
        enhanced = fuse_enhancement(residual_clahe, vessel_map, original_norm=norm, alpha=0.65)

        # Upscale back if needed
        if scale < 1.0:
            residual = cv2.resize(residual, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
            enhanced = cv2.resize(enhanced, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
            vessel_map = cv2.resize(vessel_map, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)
            anatomy_mask = cv2.resize(anatomy_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_LINEAR)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        result = {
            "raw": img if img.dtype == np.uint16 else to_float32(img),
            "processed": residual,       # background-suppressed
            "enhanced": enhanced,        # vessel-enhanced
            "vessel_map": vessel_map,
            "anatomy_mask": anatomy_mask,
            "working_shape": np.array(work.shape, dtype=np.int32),
            "latency_ms": latency_ms,
        }
        if return_intermediates:
            result["denoised"] = denoised
            result["normalized"] = norm
        return result

    def process_to_vis(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        """Convenience: returns uint8 visualization images + latency."""
        out = self.process(img)
        return {
            "raw_vis": to_uint8_vis(out["raw"]),
            "processed_vis": to_uint8_vis(out["processed"]),
            "enhanced_vis": to_uint8_vis(out["enhanced"]),
            "latency_ms": out["latency_ms"],
        }


def process_frame(
    img: np.ndarray, config: Optional[PipelineConfig] = None
) -> Dict[str, np.ndarray]:
    """Functional API for a single frame."""
    pipe = CoronaryPipeline(config)
    return pipe.process(img)
