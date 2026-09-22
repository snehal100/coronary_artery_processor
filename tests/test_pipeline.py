"""Functional and regression tests for the coronary pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import CoronaryPipeline, process_frame
from src.io_utils import read_image, validate_16bit, to_uint8_vis
from src.config import DEFAULT_CONFIG


@pytest.fixture
def synthetic_16bit():
    rng = np.random.default_rng(123)
    return rng.integers(1000, 40000, (256, 256), dtype=np.uint16)


@pytest.fixture
def real_sample():
    path = ROOT / "data" / "raw" / "synthetic_angio_00.png"
    if path.exists():
        return read_image(path)
    rng = np.random.default_rng(0)
    return rng.integers(0, 45000, (512, 512), dtype=np.uint16)


def test_validate_16bit(synthetic_16bit):
    ok, msg = validate_16bit(synthetic_16bit)
    assert ok
    assert "uint16" in msg


def test_process_returns_expected_keys(synthetic_16bit):
    out = process_frame(synthetic_16bit)
    assert "raw" in out
    assert "processed" in out
    assert "enhanced" in out
    assert "anatomy_mask" in out
    assert "latency_ms" in out
    assert out["processed"].shape == synthetic_16bit.shape
    assert out["enhanced"].shape == synthetic_16bit.shape
    assert out["anatomy_mask"].shape == synthetic_16bit.shape
    assert out["anatomy_mask"].dtype == np.float32
    assert tuple(out["working_shape"]) == (256, 256)


def test_output_not_empty(synthetic_16bit):
    out = process_frame(synthetic_16bit)
    assert out["processed"].size > 0
    assert out["enhanced"].max() > out["enhanced"].min()


def test_different_sizes():
    for h, w in [(128, 128), (256, 384), (512, 512)]:
        img = np.random.randint(0, 30000, (h, w), dtype=np.uint16)
        out = process_frame(img)
        assert out["enhanced"].shape == (h, w)


def test_low_contrast():
    img = np.full((200, 200), 20000, dtype=np.uint16)
    img[80:90, 50:150] = 19500  # faint vessel
    out = process_frame(img)
    assert out["enhanced"].shape == img.shape


def test_high_noise():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 50000, (256, 256), dtype=np.uint16)
    out = process_frame(img)
    assert out["latency_ms"] > 0


def test_pipeline_stateful(real_sample):
    pipe = CoronaryPipeline()
    out1 = pipe.process(real_sample)
    out2 = pipe.process(real_sample)
    assert out1["enhanced"].shape == out2["enhanced"].shape


def test_vis_conversion(synthetic_16bit):
    vis = to_uint8_vis(synthetic_16bit)
    assert vis.dtype == np.uint8
    assert vis.shape == synthetic_16bit.shape


def test_unsupported_raises():
    ok, msg = validate_16bit(np.zeros((5,), dtype=np.uint16))  # 1D
    assert not ok
    ok2, _ = validate_16bit(None)
    assert not ok2
