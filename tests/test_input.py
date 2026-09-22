"""Input validation and 16-bit handling tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import read_image, validate_16bit, robust_normalize, to_uint16, to_uint8_vis


def test_read_synthetic():
    path = ROOT / "data" / "raw" / "synthetic_angio_00.png"
    if not path.exists():
        pytest.skip("sample not present")
    img = read_image(path)
    assert img.ndim == 2
    assert img.dtype == np.uint16


def test_robust_normalize():
    img = np.array([[0, 100, 1000], [500, 600, 65535]], dtype=np.uint16)
    norm = robust_normalize(img)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0


def test_to_uint16_roundtrip():
    f = np.linspace(0, 1, 100).reshape(10, 10).astype(np.float32)
    u16 = to_uint16(f, scale_from_01=True)
    assert u16.dtype == np.uint16
    assert u16.max() == 65535


def test_float_input_range_and_finiteness():
    assert validate_16bit(np.array([[0.0, 65535.0]], dtype=np.float32))[0]
    assert not validate_16bit(np.array([[-1.0]], dtype=np.float32))[0]
    assert not validate_16bit(np.array([[np.nan]], dtype=np.float32))[0]
