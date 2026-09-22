"""Performance regression tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import CoronaryPipeline


def test_latency_under_threshold():
    """Mean latency on 512x512 should stay well under 36 ms on typical CI/CPU."""
    pipe = CoronaryPipeline()
    img = np.random.randint(0, 40000, (512, 512), dtype=np.uint16)
    # warm-up
    for _ in range(3):
        pipe.process(img)
    times = []
    for _ in range(15):
        out = pipe.process(img)
        times.append(out["latency_ms"])
    mean = float(np.mean(times))
    maximum = float(np.max(times))
    p95 = float(np.percentile(times, 95))
    assert maximum <= 36.0, f"Maximum latency {maximum:.1f} ms exceeds 36 ms"
    assert p95 <= 36.0, f"P95 latency {p95:.1f} ms exceeds 36 ms"
    print(f"Mean latency: {mean:.2f} ms; P95: {p95:.2f} ms; max: {maximum:.2f} ms")
