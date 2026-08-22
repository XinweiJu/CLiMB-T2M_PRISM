#!/usr/bin/env python3
"""Generate CLiMB trajectories with final Track5 depth or pose checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
PRISM_ROOT = ROOT.parent
T2M = PRISM_ROOT / "Track2Map-CLiMB" / "track2map-climb"
for path in (PRISM_ROOT, T2M, ROOT.parent / "PRISM-TrackFusion-CLiMB"):
    sys.path.insert(0, str(path))

from climb_io import PoseRecord, write_points3d, write_runtime, write_trajectory
from track2map_frontend import Track2MapFrontendConfig, Track2MapMonocularFrontend, load_track2map_models
from track_fusion import FlowRasterizer, TrackRasterizer
from inference5 import FlowAddDepth, FlowAddPose, Track5Depth, Track5Pose

WEIGHTS = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/PRISM_Track5_HK")
C3VD_WEIGHTS = Path("/raid/scratch_not_backed_up/xinwei/Datasets/C3VD_Undistorted/Weights_PRISM_Flow_C3VD")
MANIFEST = PRISM_ROOT / "checkpoints.json"


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("depth", "pose"), required=True)
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--checkpoint", default="weights_9")
    p.add_argument("--variant", choices=("track5", "c3vd-flow5", "c3vd-flow-add"), default="track5")
    p.add_argument("--conditioning", choices=("tracks", "flow"), default="tracks")
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=2834)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def cfg(max_frames=0):
    return Track2MapFrontendConfig(
        max_frames=max_frames,
        processing_width=720, segment_length=12, grid_size=18,
        mixed_precision=True, motion_gate_enabled=False,
        propagate_on_failure=True, propagation_min_inlier_ratio=0.15,
    )


def load_bundle(device):
    vendor = T2M / "vendor"
    return load_track2map_models(
        vendor / "co-tracker", vendor / "co-tracker/scaled_online.pth",
        vendor / "PRISM-CLiMB", device=device)


class DepthFrontend(Track2MapMonocularFrontend):
    def __init__(self, *args, track5, rasterizer, **kwargs):
        super().__init__(*args, **kwargs); self.track5 = track5; self.rasterizer = rasterizer; self.track_map = None
    def _track_segment(self, frames):
        tracks, visibility = super()._track_segment(frames)
        if isinstance(self.rasterizer, FlowRasterizer):
            self.track_map = self.rasterizer(frames[1], frames[0]) if len(frames) > 1 else np.zeros(frames[0].shape[:2], np.float32)
        else:
            self.track_map = self.rasterizer(tracks[0], visibility[0], frames[0].shape[:2])
        return tracks, visibility
    def _estimate_anchor_depth(self, image, alignment_samples=None):
        original = self.depth_model
        self.depth_model = lambda frame: self.track5(frame, self.track_map)
        try: return super()._estimate_anchor_depth(image, alignment_samples)
        finally: self.depth_model = original


def make_tracker(video, args, bundle):
    vendor = T2M / "vendor"
    return Track2MapMonocularFrontend(
        video, args.seed, cfg(args.max_frames), vendor / "co-tracker",
        vendor / "co-tracker/scaled_online.pth", vendor / "PRISM-CLiMB",
        device=args.device, models=bundle)


def run_depth(video, args, bundle, rasterizer):
    if args.variant == "track5":
        checkpoint = WEIGHTS / "hk_dlpe_track5_depth/models" / args.checkpoint
        model = Track5Depth(checkpoint, device=bundle.device, manifest_path=MANIFEST)
    else:
        stem = "c3vd_dlpe_flow5_depth" if args.variant == "c3vd-flow5" else "c3vd_dlpe_flow_add_depth"
        checkpoint = C3VD_WEIGHTS / stem / "models" / args.checkpoint
        cls = Track5Depth if args.variant == "c3vd-flow5" else FlowAddDepth
        model = cls(checkpoint, device=bundle.device, manifest_path=MANIFEST)
    vendor = T2M / "vendor"
    frontend = DepthFrontend(
        video, args.seed, cfg(args.max_frames), vendor / "co-tracker", vendor / "co-tracker/scaled_online.pth",
        vendor / "PRISM-CLiMB", device=args.device, models=bundle,
        track5=model, rasterizer=rasterizer)
    try: return frontend.process()
    finally: frontend.close()


def run_pose(video, args, bundle, rasterizer):
    if args.variant == "track5":
        checkpoint = WEIGHTS / "hk_dlpe_track5_pose/models" / args.checkpoint
        model = Track5Pose(checkpoint, device=bundle.device, manifest_path=MANIFEST)
    else:
        stem = "c3vd_dlpe_flow5_pose" if args.variant == "c3vd-flow5" else "c3vd_dlpe_flow_add_pose"
        checkpoint = C3VD_WEIGHTS / stem / "models" / args.checkpoint
        cls = Track5Pose if args.variant == "c3vd-flow5" else FlowAddPose
        model = cls(checkpoint, device=bundle.device, manifest_path=MANIFEST)
    tracker = make_tracker(video, args, bundle); tracker._decoded_frames = 0
    records=[]; pose=np.eye(4); overlap=None; offset=0
    try:
        while True:
            frames, ended = tracker._read_segment(overlap)
            if not frames or (overlap is not None and len(frames)==1): break
            tracks, visibility = tracker._track_segment(frames)
            if isinstance(rasterizer, FlowRasterizer):
                maps=[np.zeros(frames[0].shape[:2], np.float32)]
                maps += [rasterizer(frames[i], frames[i-1]) for i in range(1, len(frames))]
            else:
                maps=[rasterizer(tracks[i], visibility[i], frames[i].shape[:2]) for i in range(len(frames))]
            for i in range(0 if not records else 1, len(frames)):
                state="anchor"
                if i>0:
                    relative=model(frames[i-1], frames[i], maps[i-1], maps[i]).astype(np.float64)
                    pose=pose @ np.linalg.inv(relative); state="track5_pose"
                number=offset+i
                records.append(PoseRecord(number+1, number/tracker.fps, pose.copy(), [state,"1","0","0","0"]))
            if ended: break
            overlap=frames[-1]; offset += len(frames)-1
    finally: tracker.close()
    return records, []


def main():
    args=parse(); source=Path(args.input); videos=[source] if source.is_file() else sorted(source.glob("*.mp4"))
    output=Path(args.output); output.mkdir(parents=True, exist_ok=True)
    bundle=load_bundle(args.device)
    rasterizer=FlowRasterizer() if args.conditioning == "flow" else TrackRasterizer(3,2.0)
    for video in videos:
        start=time.perf_counter(); records,points=(run_depth if args.mode=="depth" else run_pose)(video,args,bundle,rasterizer)
        elapsed=time.perf_counter()-start; root=output/video.stem/"1"
        write_trajectory(root/"camera_trajectory/cam_traj_map_000.txt",records)
        write_points3d(root/"3D_maps/0/points3D.txt",points); write_runtime(root/"runtime.txt",0,elapsed)
        print(f"{video.stem}: frames={len(records)} points={len(points)} seconds={elapsed:.3f}",flush=True)


if __name__ == "__main__": main()
