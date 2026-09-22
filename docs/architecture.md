# Architecture

The system is deliberately lightweight and modular.

- `src/io_utils.py` — 16-bit I/O, validation, normalization helpers
- `src/preprocessing.py` — denoise + robust normalize
- `src/background_suppression.py` — large-scale background estimate & residual
- `src/vessel_enhancement.py` — multi-scale LoG vesselness + CLAHE
- `src/postprocessing.py` — fusion and output packing
- `src/pipeline.py` — orchestrates the stages; measures latency
- `src/config.py` — single dataclass of tunable parameters

UI (`app.py`) and benchmark (`benchmark.py`) both call the same `CoronaryPipeline` class, guaranteeing that measured latency matches the demo path.
