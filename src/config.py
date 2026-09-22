"""Configuration for the coronary artery processing pipeline."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PipelineConfig:
    """Tunable parameters for the processing pipeline.

    All parameters chosen for speed ≤36 ms/frame on CPU while
    preserving thin vessels and suppressing background structures.
    """
    # Normalization / windowing
    percentile_low: float = 1.0
    percentile_high: float = 99.0

    # Noise reduction
    bilateral_d: int = 5
    bilateral_sigma_color: float = 50.0
    bilateral_sigma_space: float = 50.0
    use_bilateral: bool = False  # disabled by default for speed; Gaussian is faster

    gaussian_ksize: int = 3
    gaussian_sigma: float = 0.8

    # Background estimation (large-scale)
    bg_ksize: int = 31
    bg_sigma: float = 12.0
    anatomy_open_ksize: int = 15
    anatomy_suppression: float = 0.55

    # Morphological top-hat / black-hat for vessel-like structures
    morph_kernel_size: int = 15  # for background / large structure suppression
    vessel_morph_size: int = 3   # smaller for vessel enhancement

    # CLAHE for local contrast
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    use_clahe: bool = False

    # Multi-scale vessel enhancement (LoG-style, speed optimized)
    vessel_scales: Tuple[float, ...] = (1.0, 1.8)
    vessel_beta: float = 0.5
    vessel_c: float = 15.0

    # Output
    output_dtype: str = "uint16"
    vis_dtype: str = "uint8"

    # Performance
    max_image_side: int = 256
    enable_downscale: bool = True


DEFAULT_CONFIG = PipelineConfig()
