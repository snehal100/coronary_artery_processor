#!/usr/bin/env python3
"""Evaluate vessel-region contrast on the bundled COCO-style dataset.

This is an engineering proxy metric, not clinical validation. COCO boxes and
polygons identify annotated regions, but they do not label ribs, spine, lungs,
or every coronary pixel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import read_image, to_uint8_vis
from src.pipeline import CoronaryPipeline


def annotation_mask(annotation: dict, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    segmentation = annotation.get("segmentation", [])
    if isinstance(segmentation, list):
        for polygon in segmentation:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            if len(points) >= 3:
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    if not mask.any():
        x, y, width, height = annotation["bbox"]
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1 = min(shape[1], int(x + width))
        y1 = min(shape[0], int(y + height))
        mask[y0:y1, x0:x1] = 1
    return mask.astype(bool)


def evaluate(split_dir: Path, limit: int) -> dict:
    annotation_path = split_dir / "annotations" / f"{split_dir.name}.json"
    image_dir = split_dir / "images"
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = {}
    for item in data.get("annotations", []):
        by_image.setdefault(item["image_id"], []).append(item)

    pipeline = CoronaryPipeline()
    pipeline.process(np.zeros((512, 512), dtype=np.uint8))
    rows = []
    for image_info in data.get("images", [])[:limit]:
        path = image_dir / image_info["file_name"]
        if not path.exists():
            continue
        image = read_image(path)
        if image.ndim != 2:
            raise ValueError(f"Expected grayscale image after reading {path}")
        result = pipeline.process(image)
        raw = to_uint8_vis(image).astype(np.float32) / 255.0
        enhanced = np.asarray(result["enhanced"], dtype=np.float32)
        positive = np.zeros(image.shape, dtype=bool)
        for annotation in by_image.get(image_info["id"], []):
            positive |= annotation_mask(annotation, image.shape)
        if not positive.any():
            continue
        negative = ~cv2.dilate(positive.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        if not negative.any():
            continue
        raw_contrast = float(raw[positive].mean() - raw[negative].mean())
        enhanced_contrast = float(enhanced[positive].mean() - enhanced[negative].mean())
        rows.append({
            "image": image_info["file_name"],
            "raw_contrast": raw_contrast,
            "enhanced_contrast": enhanced_contrast,
            "contrast_change": enhanced_contrast - raw_contrast,
            "latency_ms": float(result["latency_ms"]),
        })

    if not rows:
        raise RuntimeError("No annotated images were available for evaluation")
    return {
        "split": split_dir.name,
        "images_evaluated": len(rows),
        "mean_raw_contrast": float(np.mean([r["raw_contrast"] for r in rows])),
        "mean_enhanced_contrast": float(np.mean([r["enhanced_contrast"] for r in rows])),
        "mean_contrast_change": float(np.mean([r["contrast_change"] for r in rows])),
        "mean_latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
        "max_latency_ms": float(np.max([r["latency_ms"] for r in rows])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    split_dir = ROOT / "data" / "raw" / "data" / "stenosis" / args.split
    result = evaluate(split_dir, args.limit)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
