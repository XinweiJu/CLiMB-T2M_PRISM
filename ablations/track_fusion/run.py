#!/usr/bin/env python3
"""Run track-conditioned original DLPE depth or direct pose experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
PRISM_ROOT = ROOT.parent
T2M_APP = PRISM_ROOT / "Track2Map-CLiMB" / "track2map-climb"
POSE_APP = PRISM_ROOT / "PRISM-Pose-CLiMB"
for source_root in (PRISM_ROOT, T2M_APP, POSE_APP):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from climb_io import MapPoint, PoseRecord, write_points3d, write_runtime, write_trajectory
from track2map_frontend import (
    Track2MapFrontendConfig,
    Track2MapMonocularFrontend,
    load_track2map_models,
)
from track_fusion import FlowRasterizer, TrackFusedDepth, TrackFusedPose, TrackRasterizer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("depth", "pose"), required=True)
    parser.add_argument("--input", required=True, help="An mp4 or directory of mp4 files")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--processing-width", type=int, default=720)
    parser.add_argument("--segment-length", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=18)
    parser.add_argument("--fusion", choices=("add", "blend"), default="add")
    parser.add_argument(
        "--conditioning", choices=("tracks", "flow"), default="tracks",
        help="Map fused into the pretrained auxiliary channel",
    )
    parser.add_argument("--track-alpha", type=float, default=0.20)
    parser.add_argument("--track-radius", type=int, default=3)
    parser.add_argument("--track-sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2834)
    return parser.parse_args()


def videos(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.glob("*.mp4"))


def load_components(args: argparse.Namespace):
    vendor = T2M_APP / "vendor"
    bundle = load_track2map_models(
        cotracker_repo=vendor / "co-tracker",
        cotracker_checkpoint=vendor / "co-tracker" / "scaled_online.pth",
        prism_climb_root=vendor / "PRISM-CLiMB",
        device=args.device,
    )

    # Replace any packaged/submission depth with the explicitly recorded,
    # original FEAST DLPE checkpoint. This project never loads adapted weights.
    from prism_depth_climb import DepthGenerator

    checkpoint = json.loads((ROOT / "checkpoints.json").read_text())
    bundle.depth_model = DepthGenerator(
        checkpoint_dir=checkpoint["depth"]["checkpoint_dir"],
        device=bundle.device,
        manifest_path=PRISM_ROOT / "checkpoints.json",
    )
    return bundle


class FusedDepthFrontend(Track2MapMonocularFrontend):
    """Inject current CoTracker anchor points into PRISM anchor-depth inference."""

    def __init__(self, *args, rasterizer, fused_depth, conditioning: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.rasterizer = rasterizer
        self.fused_depth = fused_depth
        self.conditioning = conditioning
        self.current_track_map = None

    def _track_segment(self, frames_bgr):
        tracks, visibility = super()._track_segment(frames_bgr)
        if self.conditioning == "flow":
            if len(frames_bgr) < 2:
                self.current_track_map = np.zeros(frames_bgr[0].shape[:2], np.float32)
            else:
                self.current_track_map = self.rasterizer(frames_bgr[0], frames_bgr[1])
        else:
            self.current_track_map = self.rasterizer(
                tracks[0], visibility[0], frames_bgr[0].shape[:2]
            )
        return tracks, visibility

    def _estimate_anchor_depth(self, anchor_bgr, alignment_samples=None):
        if self.current_track_map is None:
            raise RuntimeError("CoTracker points must be generated before depth")
        original_model = self.depth_model
        self.depth_model = lambda image: self.fused_depth(image, self.current_track_map)
        try:
            return super()._estimate_anchor_depth(anchor_bgr, alignment_samples)
        finally:
            self.depth_model = original_model


def config(args: argparse.Namespace) -> Track2MapFrontendConfig:
    return Track2MapFrontendConfig(
        max_frames=args.max_frames,
        processing_width=args.processing_width,
        segment_length=args.segment_length,
        grid_size=args.grid_size,
        mixed_precision=True,
        motion_gate_enabled=False,
        propagate_on_failure=True,
        propagation_min_inlier_ratio=0.15,
    )


def depth_run(video: Path, args, bundle, rasterizer):
    fused = TrackFusedDepth(bundle.depth_model, mode=args.fusion, alpha=args.track_alpha)
    estimator = FusedDepthFrontend(
        video_path=video,
        seed=args.seed,
        config=config(args),
        cotracker_repo=T2M_APP / "vendor/co-tracker",
        cotracker_checkpoint=T2M_APP / "vendor/co-tracker/scaled_online.pth",
        prism_climb_root=T2M_APP / "vendor/PRISM-CLiMB",
        device=args.device,
        models=bundle,
        rasterizer=rasterizer,
        fused_depth=fused,
        conditioning=args.conditioning,
    )
    try:
        return estimator.process()
    finally:
        estimator.close()


def pose_run(video: Path, args, bundle, rasterizer):
    from prism_pose import PoseGenerator

    pose = PoseGenerator(
        variant="dlpe", device=bundle.device, manifest_path=ROOT / "pose_manifest.json"
    )
    fused_pose = TrackFusedPose(pose, mode=args.fusion, alpha=args.track_alpha)
    tracker = Track2MapMonocularFrontend(
        video_path=video,
        seed=args.seed,
        config=config(args),
        cotracker_repo=T2M_APP / "vendor/co-tracker",
        cotracker_checkpoint=T2M_APP / "vendor/co-tracker/scaled_online.pth",
        prism_climb_root=T2M_APP / "vendor/PRISM-CLiMB",
        device=args.device,
        models=bundle,
    )
    records = []
    camera_to_world = np.eye(4, dtype=np.float64)
    overlap = None
    frame_offset = 0
    tracker._decoded_frames = 0
    try:
        while True:
            frames, ended = tracker._read_segment(overlap)
            if not frames or (overlap is not None and len(frames) == 1):
                break
            tracks, visibility = tracker._track_segment(frames)
            if args.conditioning == "flow":
                maps = None
            else:
                maps = [rasterizer(tracks[i], visibility[i], frames[i].shape[:2]) for i in range(len(frames))]
            start = 0 if not records else 1
            for index in range(start, len(frames)):
                state = "anchor"
                if index > 0:
                    if args.conditioning == "flow":
                        pair_map = rasterizer(frames[index - 1], frames[index])
                        map0 = map1 = pair_map
                    else:
                        map0, map1 = maps[index - 1], maps[index]
                    relative = fused_pose(frames[index - 1], frames[index], map0, map1)
                    if relative.shape != (4, 4) or not np.isfinite(relative).all():
                        raise RuntimeError(f"Invalid DLPE pose at local frame {index}")
                    camera_to_world = camera_to_world @ np.linalg.inv(relative.astype(np.float64))
                    state = "dlpe_track_fused"
                frame_number = frame_offset + index
                records.append(PoseRecord(
                    frame_id=frame_number + 1,
                    timestamp=frame_number / tracker.fps,
                    camera_to_world=camera_to_world.copy(),
                    extras=[state, "1.000000", str(int(visibility[index].sum())), "0", "0"],
                ))
            if ended:
                break
            overlap = frames[-1]
            frame_offset += len(frames) - 1
    finally:
        tracker.close()
    return records, []


def main() -> None:
    args = arguments()
    if not 0.0 <= args.track_alpha <= 1.0:
        raise SystemExit("--track-alpha must be in [0,1]")
    input_path = Path(args.input)
    selected = videos(input_path)
    if not selected:
        raise SystemExit(f"No mp4 files found at {input_path}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    bundle = load_components(args)
    rasterizer = (
        FlowRasterizer()
        if args.conditioning == "flow"
        else TrackRasterizer(args.track_radius, args.track_sigma)
    )
    for video in selected:
        start = time.perf_counter()
        records, points = (
            depth_run(video, args, bundle, rasterizer)
            if args.mode == "depth"
            else pose_run(video, args, bundle, rasterizer)
        )
        elapsed = time.perf_counter() - start
        sequence_name = video.stem
        root = output / sequence_name / "1"
        write_trajectory(root / "camera_trajectory/cam_traj_map_000.txt", records)
        write_points3d(root / "3D_maps/0/points3D.txt", points)
        write_runtime(root / "runtime.txt", 0.0, elapsed)
        print(f"{video.stem}: frames={len(records)} points={len(points)} seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
