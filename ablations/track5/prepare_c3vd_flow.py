#!/usr/bin/env python3
"""Decode and rectify the official C3VD current-to-previous optical flow."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


K = np.array([[802.319, 0.0, 668.286], [0.0, 801.885, 547.733], [0.0, 0.0, 1.0]])
D = np.array([-0.42234, 0.10654, 0.0, 0.0])
RAW_SIZE = (1350, 1080)
OUTPUT_SIZE = (288, 288)
_LOW_MAPS = None


def low_resolution_maps():
    global _LOW_MAPS
    if _LOW_MAPS is None:
        map_x, map_y = cv2.initUndistortRectifyMap(K, D, None, K, RAW_SIZE, cv2.CV_32FC1)
        _LOW_MAPS = (
            cv2.resize(map_x, OUTPUT_SIZE, interpolation=cv2.INTER_AREA),
            cv2.resize(map_y, OUTPUT_SIZE, interpolation=cv2.INTER_AREA),
        )
    return _LOW_MAPS


def process(item: tuple[str, str]) -> dict:
    source_name, target_name = item
    source, target = Path(source_name), Path(target_name)
    encoded = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32)
    if encoded.shape != (RAW_SIZE[1], RAW_SIZE[0], 3):
        raise ValueError(f"unexpected C3VD flow shape {encoded.shape}: {source}")

    # Official encoding is linear in R/G over [-20, 20] pixels. This release is
    # uint8; using dtype max also supports a uint16 copy of the dataset.
    maximum = float(np.iinfo(np.asarray(Image.open(source)).dtype).max)
    flow = encoded[..., :2] / maximum * 40.0 - 20.0

    # cv2.undistort output coordinates use K again. For every rectified source
    # pixel, sample its distorted flow endpoint, rectify that endpoint, and take
    # the difference. This transforms vectors as vectors, not as RGB colours.
    map_x, map_y = low_resolution_maps()
    flow_x = cv2.remap(flow[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    flow_y = cv2.remap(flow[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    end_distorted = np.stack((map_x + flow_x, map_y + flow_y), axis=-1).reshape(-1, 1, 2)
    end_rectified = cv2.undistortPoints(end_distorted, K, D, P=K).reshape(OUTPUT_SIZE[1], OUTPUT_SIZE[0], 2)
    grid_x, grid_y = np.meshgrid(
        np.arange(OUTPUT_SIZE[0], dtype=np.float32) * RAW_SIZE[0] / OUTPUT_SIZE[0],
        np.arange(OUTPUT_SIZE[1], dtype=np.float32) * RAW_SIZE[1] / OUTPUT_SIZE[1])
    rectified = end_rectified - np.stack((grid_x, grid_y), axis=-1)
    rectified[..., 0] *= OUTPUT_SIZE[0] / RAW_SIZE[0]
    rectified[..., 1] *= OUTPUT_SIZE[1] / RAW_SIZE[1]
    magnitude = np.linalg.norm(rectified, axis=-1)
    finite = np.isfinite(magnitude)
    scale = float(np.percentile(magnitude[finite], 95.0)) if finite.any() else 0.0
    normalized = np.clip(magnitude / scale, 0.0, 1.0) if scale > 1e-6 else np.zeros_like(magnitude)
    normalized[~finite] = 0.0
    normalized = cv2.GaussianBlur(normalized.astype(np.float32), (0, 0), 1.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(normalized * 255.0).astype(np.uint8), "L").save(target)
    return {"source": str(source), "target": str(target), "p95_rectified_pixels": scale}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/raid/scratch_not_backed_up/public_datasets/C3VD"))
    parser.add_argument("--output", type=Path, default=Path("/raid/scratch_not_backed_up/xinwei/Datasets/C3VD_Undistorted/Motion"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    items = []
    for source in sorted(args.source.glob("*/*_flow.tiff")):
        target = args.output / "flow" / source.parent.name / source.name.replace("_flow.tiff", ".png")
        items.append((str(source), str(target)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(process, items, chunksize=4))
    manifest = {
        "source": str(args.source), "output": str(args.output), "count": len(records),
        "direction": "current_to_previous", "encoding": "R/G linear [-20,20] pixels",
        "camera_matrix": K.tolist(), "distortion": D.tolist(),
        "raw_size_wh": RAW_SIZE, "output_size_wh": OUTPUT_SIZE,
        "normalization": "per-frame p95 magnitude, clip [0,1], Gaussian sigma=1",
        "vector_rectification": "rectify source and flow endpoint, then subtract",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
