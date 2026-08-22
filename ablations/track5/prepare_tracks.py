#!/usr/bin/env python3
"""Precompute Track2Map/CoTracker heat maps and a video-level HK split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
PRISM_ROOT = ROOT.parent
T2M = PRISM_ROOT / "Track2Map-CLiMB" / "track2map-climb"
COTRACKER = T2M / "vendor" / "co-tracker"
sys.path.insert(0, str(ROOT.parent / "PRISM-TrackFusion-CLiMB"))
from track_fusion import TrackRasterizer


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--segment-length", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=18)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model(device):
    model = torch.hub.load(
        str(COTRACKER), "cotracker3_online", source="local", pretrained=False
    ).to(device).eval()
    state = torch.load(COTRACKER / "scaled_online.pth", map_location="cpu", weights_only=True)
    model.model.load_state_dict(state)
    return model


@torch.inference_mode()
def track_chunk(model, image_paths, device, grid_size):
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError("Failed to decode one or more frames")
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    original_length = len(rgb)
    while len(rgb) < model.step + 1:
        rgb.append(rgb[-1].copy())
    video = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2)[None].float().to(device)
    model(video_chunk=video[:, : model.step * 2], is_first_step=True, grid_size=grid_size)
    tracks = visibility = None
    for start in range(0, video.shape[1] - model.step, model.step):
        tracks, visibility = model(
            video_chunk=video[:, start : start + model.step * 2],
            is_first_step=False,
            grid_size=0,
        )
    if tracks is None:
        raise RuntimeError("CoTracker returned no tracks")
    return (
        frames,
        tracks[0, :original_length].cpu().numpy().astype(np.float32),
        visibility[0, :original_length].cpu().numpy().astype(bool),
    )


def main():
    opt = args()
    frames_root = Path(opt.frames)
    output_root = Path(opt.output) / "tracks"
    split_root = Path(opt.split_output)
    sequences = sorted(path for path in frames_root.iterdir() if path.is_dir())
    if not sequences:
        raise SystemExit("No frame directories found")
    shuffled = sequences[:]
    random.Random(opt.seed).shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * opt.val_fraction))
    val_names = {path.name for path in shuffled[:val_count]}
    train_lines, val_lines = [], []
    for sequence in sequences:
        paths = sorted(sequence.glob("*.png"))
        target = val_lines if sequence.name in val_names else train_lines
        target.extend(str(path) for path in paths[1:-1])
    split_root.mkdir(parents=True, exist_ok=True)
    (split_root / "train_files.txt").write_text("\n".join(train_lines) + "\n")
    (split_root / "val_files.txt").write_text("\n".join(val_lines) + "\n")

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    rasterizer = TrackRasterizer(radius=3, sigma=2.0)
    generated = 0
    for sequence_index, sequence in enumerate(sequences, 1):
        paths = sorted(sequence.glob("*.png"))
        sequence_output = output_root / sequence.name
        sequence_output.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(paths), opt.segment_length):
            chunk = paths[start : start + opt.segment_length]
            missing = [p for p in chunk if opt.overwrite or not (sequence_output / p.name).is_file()]
            if not missing:
                continue
            frames, tracks, visibility = track_chunk(model, chunk, device, opt.grid_size)
            for path, frame, points, visible in zip(chunk, frames, tracks, visibility):
                target = sequence_output / path.name
                if target.is_file() and not opt.overwrite:
                    continue
                heat = rasterizer(points, visible, frame.shape[:2])
                if not cv2.imwrite(str(target), np.rint(heat * 255.0).astype(np.uint8)):
                    raise RuntimeError("Failed to save {}".format(target))
                generated += 1
        print("[{}/{}] {} frames={}".format(sequence_index, len(sequences), sequence.name, len(paths)), flush=True)
    manifest = {
        "frames_root": str(frames_root),
        "tracks_root": str(output_root),
        "seed": opt.seed,
        "train_sequences": sorted(p.name for p in sequences if p.name not in val_names),
        "val_sequences": sorted(val_names),
        "train_centers": len(train_lines),
        "val_centers": len(val_lines),
        "generated_this_run": generated,
        "grid_size": opt.grid_size,
        "segment_length": opt.segment_length,
        "checkpoint": str(COTRACKER / "scaled_online.pth"),
    }
    (Path(opt.output) / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
