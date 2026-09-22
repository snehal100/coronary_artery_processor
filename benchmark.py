#!/usr/bin/env python3
"""End-to-end latency benchmark for the coronary processing pipeline.

Measures complete process() call from input frame to final enhanced output.
"""
from __future__ import annotations

import platform
import sys
import time
import importlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2

# Ensure src is importable
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import CoronaryPipeline
from src.io_utils import read_image
from src.config import DEFAULT_CONFIG


def get_hardware_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or "unknown",
        "opencv": cv2.__version__,
        "numpy": np.__version__,
    }
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    try:
        psutil = importlib.import_module("psutil")
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except ImportError:
        info["ram_gb"] = "unknown"
    return info


def load_frames(data_dir: Path, max_frames: int = 5) -> list:
    paths = sorted(data_dir.glob("*.png")) + sorted(data_dir.glob("*.tif")) + sorted(data_dir.glob("*.tiff"))
    frames = []
    for p in paths[:max_frames]:
        img = read_image(p)
        frames.append(img)
    if not frames:
        # Synthetic fallback
        print("No images found; generating synthetic 512x512 uint16 frames.")
        rng = np.random.default_rng(0)
        for _ in range(max_frames):
            frames.append(rng.integers(0, 45000, (512, 512), dtype=np.uint16))
    return frames


def run_benchmark(
    frames: list,
    warmup: int = 5,
    iterations: int = 50,
) -> dict:
    pipe = CoronaryPipeline(DEFAULT_CONFIG)
    # Warm-up
    for i in range(warmup):
        _ = pipe.process(frames[i % len(frames)])

    latencies = []
    for i in range(iterations):
        img = frames[i % len(frames)]
        t0 = time.perf_counter()
        out = pipe.process(img)
        t1 = time.perf_counter()
        # Prefer internal measurement if available, else wall time
        lat = out.get("latency_ms", (t1 - t0) * 1000.0)
        latencies.append(lat)

    arr = np.array(latencies)
    return {
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "std_ms": float(arr.std()),
        "fps": float(1000.0 / arr.mean()) if arr.mean() > 0 else 0.0,
        "n_frames": iterations,
        "warmup": warmup,
        "latencies": latencies,
        "threshold_ms": 36.0,
        "status": "PASS" if float(arr.max()) <= 36.0 else "FAIL",
    }


def main():
    data_dir = ROOT / "data" / "raw"
    frames = load_frames(data_dir, max_frames=5)
    print(f"Loaded {len(frames)} frames. Shape[0]={frames[0].shape}, dtype={frames[0].dtype}")

    hw = get_hardware_info()
    print("Hardware:", hw)

    results = run_benchmark(frames, warmup=8, iterations=60)

    status = results["status"]
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"| Metric           | Result                             |")
    print(f"| ---------------- | ---------------------------------- |")
    print(f"| Minimum latency  | {results['min_ms']:.3f} ms                       |")
    print(f"| Average latency  | {results['mean_ms']:.3f} ms                       |")
    print(f"| Median latency   | {results['median_ms']:.3f} ms                       |")
    print(f"| Maximum latency  | {results['max_ms']:.3f} ms                       |")
    print(f"| P95 latency      | {results['p95_ms']:.3f} ms                       |")
    print(f"| FPS              | {results['fps']:.2f}                             |")
    print(f"| Hardware         | {hw.get('cpu', 'N/A')[:40]} |")
    print(f"| Image resolution | {frames[0].shape[1]}x{frames[0].shape[0]}                          |")
    print(f"| Bit depth        | 16-bit                             |")
    print(f"| Iterations       | {results['n_frames']}                                |")
    print(f"| Warm-up          | {results['warmup']}                                 |")
    print(f"| Requirement      | ≤36 ms/frame                       |")
    print(f"| Status           | {status}                               |")
    print("=" * 60)

    # Write results for documentation
    out_path = ROOT / "outputs" / "benchmark_results.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"timestamp_utc={datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"mean_ms={results['mean_ms']:.4f}\n")
        f.write(f"min_ms={results['min_ms']:.4f}\n")
        f.write(f"max_ms={results['max_ms']:.4f}\n")
        f.write(f"fps={results['fps']:.2f}\n")
        f.write(f"median_ms={results['median_ms']:.4f}\n")
        f.write(f"p95_ms={results['p95_ms']:.4f}\n")
        f.write(f"status={status}\n")
        f.write(f"cpu={hw.get('cpu')}\n")
        f.write(f"shape={frames[0].shape}\n")
        f.write(f"dtype={frames[0].dtype}\n")
        f.write(f"iterations={results['n_frames']}\n")
        f.write(f"threshold_ms={results['threshold_ms']:.1f}\n")
    print(f"\nResults written to {out_path}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
