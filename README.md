# Cardiology Coronary Artery Image Processing System

**Research / educational engineering demonstration** — not a clinically validated diagnostic device.

## Problem Statement

Coronary X-ray / cineangiography images contain contrast-filled vessels that are often obscured by ribs, spine, lung fields, soft-tissue background, noise and exposure variation. The goal is to suppress unwanted anatomical background while preserving and enhancing thin coronary arteries under a hard real-time constraint of **≤ 36 ms per frame**.

## Objective

Build a complete, reproducible, GitHub-ready pipeline that:

- Accepts **16-bit grayscale** images
- Suppresses ribs, spine, lungs and low-frequency background
- Enhances coronary vessels
- Produces **Raw / Processed / Enhanced** outputs
- Meets **≤ 36 ms/frame** end-to-end latency on CPU
- Includes tests, benchmark, Streamlit demo and documentation

## Features

- 16-bit input validation and robust percentile normalization
- Lightweight Gaussian denoising
- Large-scale background estimation & subtraction
- Multi-scale LoG-style vessel enhancement (speed-optimized alternative to full Frangi)
- Fast histogram contrast enhancement, with optional CLAHE for offline use
- Explicit anatomical attenuation mask for ribs/spine/lung-background inspection
- Streamlit side-by-side comparison UI
- Reproducible benchmark script with min / mean / max / P95 / FPS
- Pytest suite (functional, input, performance)

## System Architecture

```
Input (16-bit grayscale)
        │
        ▼
┌───────────────────────┐
│  Validation & float32 │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│  Denoise (Gaussian)   │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Robust normalization  │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Anatomy suppression   │
│ + attenuation mask    │  ← ribs / spine / lungs attenuation
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ Multi-scale vesselness│  ← LoG max across scales
│ + morphological close │
└───────────┬───────────┘
            ▼
┌───────────────────────┐
│ CLAHE + fusion        │
└───────────┬───────────┘
            ▼
   Raw | Processed | Enhanced
```

Core engine: `src/pipeline.py` (independent of UI).  
UI: `app.py` (Streamlit).  
Benchmark: `benchmark.py`.

## Processing Pipeline (detailed)

| Stage | Purpose | Method |
|-------|---------|--------|
| Validation | Reject empty / non-2D data | dtype & shape checks |
| Denoise | Reduce noise without destroying thin vessels | 3×3 Gaussian |
| Normalize | Stable dynamic range | 1–99 percentile |
| Background suppression | Attenuate ribs, spine, lungs, soft tissue | Large Gaussian + box-filter anatomy estimate |
| Vessel enhancement | Highlight tubular structures of multiple widths | Multi-scale LoG + morph close |
| Contrast | Vessel visibility | Fast histogram equalization (optional CLAHE) |
| Fusion | Combine residual anatomy with vessel map | Weighted sum |

**Why these algorithms?**  
Full Frangi vesselness is accurate but too slow for the 36 ms budget on CPU. A multi-scale Laplacian-of-Gaussian response captures ridge-like (vessel) structures at far lower cost while still preserving thin branches. Large-kernel Gaussian background subtraction efficiently removes low-frequency anatomy without iterative morphological reconstruction.

## 16-bit Support

- Images are read with `cv2.IMREAD_UNCHANGED`.
- Internal processing uses `float32` in the original intensity range (or robustly normalized [0,1]).
- No irreversible early conversion to 8-bit.
- Visualization path produces separate `uint8` images via percentile windowing.
- Output can be written back as 16-bit PNG.

## Dataset

**Source:** Synthetic 16-bit angiography-like images generated for development (no patient data).

- Location: `data/raw/synthetic_angio_*.png`
- Resolution: 512×512
- Dtype: `uint16`
- Content: simulated background (spine, ribs, soft tissue) + dark tubular vessels + noise
- License: generated for this project; free to use and redistribute
- Limitations: **not** a clinical validation set. Real coronary angiography datasets (e.g., from public challenges) can be dropped into `data/raw/` — the pipeline accepts any 16-bit grayscale PNG/TIFF.

The repository also contains COCO-style annotated `stenosis` and `syntax`
splits under `data/raw/data/`. Inspection shows these files are 8-bit RGB PNGs,
not 16-bit acquisitions. Run `python scripts/evaluate_dataset.py --split test`
for an annotated-region contrast proxy. This is not a clinical accuracy score;
verify dataset provenance and permissions before redistribution.

## Installation

```bash
cd coronary_artery_processor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Launch Streamlit demo
streamlit run app.py

# Or process programmatically
python -c "
from src.pipeline import process_frame
from src.io_utils import read_image
img = read_image('data/raw/synthetic_angio_00.png')
out = process_frame(img)
print(out['latency_ms'], 'ms')
"
```

## Benchmarking

```bash
python benchmark.py
```

Example measured result (Windows 10, Intel CPU, 8 GB RAM, 512×512 uint16; warmed-up pure pipeline):

| Metric           | Result      |
| ---------------- | ----------- |
| Minimum latency  | 16.13 ms    |
| Average latency  | 18.60 ms    |
| Maximum latency  | 25.44 ms    |
| P95 latency      | 22.34 ms    |
| FPS              | 52.36       |
| Requirement      | ≤36 ms/frame|
| Status           | **PASS**    |

The default real-time path processes inputs at a bounded 256×256 working
resolution and resizes outputs back to the source dimensions. The UI exposes
that working resolution. The benchmark excludes file decoding, UI rendering,
and disk/network I/O, so it is not a full user-perceived response-time test.

## Testing

```bash
pytest tests/ -v
```

## Deep Learning

**Not used.** A classical OpenCV/NumPy pipeline meets the latency and quality goals with minimal dependencies and full reproducibility. Adding a lightweight U-Net would increase latency, memory and deployment complexity without guaranteed benefit under the 36 ms constraint on CPU-only hardware.

No trained model file is delivered because inference is intentionally
classical and model-free; `models/` is not used by the runtime.

## Limitations

- Clinical performance is not validated. Anatomical suppression is a fast
        heuristic, not a learned rib/spine/lung segmentation model.
- Laplacian vesselness responds to generic edges as well as vessels; vessel
        preservation and clinical sensitivity/specificity are not established.
- Strong overlapping anatomy or extreme motion may leave residual artifacts.
- Designed for single-frame processing; temporal cine coherence is future work.
- No regulatory clearance; research/educational use only.

## Future Improvements

- Optional temporal filtering for cine sequences
- Lightweight learned refinement (MobileNet-UNet) with ONNX Runtime
- GPU path via OpenCV CUDA or CuPy
- Adaptive parameter selection per exposure regime

## Project Structure

```
coronary_artery_processor/
├── app.py
├── benchmark.py
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── background_suppression.py
│   ├── vessel_enhancement.py
│   ├── postprocessing.py
│   ├── io_utils.py
│   └── config.py
├── tests/
│   ├── test_pipeline.py
│   ├── test_input.py
│   └── test_performance.py
├── data/
│   ├── README.md
│   └── raw/
├── outputs/
└── docs/
```

## License

MIT (see LICENSE).

## Disclaimer

This software is provided for research and educational purposes only. It is **not** a medical device and must not be used for clinical diagnosis or treatment decisions.
