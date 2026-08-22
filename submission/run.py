#!/usr/bin/env python3
"""Run the Track2Map-CLiMB bootstrap on every input video."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from climb_io import PoseRecord, write_points3d, write_runtime, write_trajectory
from monocular_vo import MonocularTrackVO, VOConfig
from trajectory_smoothing import smooth_pose_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track2Map-CLiMB monocular VO bootstrap")
    parser.add_argument("--profile", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--input", default="/input")
    parser.add_argument("--output", default="/output")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2834)
    parser.add_argument("--translation-step", type=float, default=0.01)
    parser.add_argument("--backend", choices=("lk", "track2map"), default="lk")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cotracker-repo", default="/opt/models/co-tracker")
    parser.add_argument("--cotracker-checkpoint", default="/opt/models/co-tracker/scaled_online.pth")
    parser.add_argument("--prism-climb-root", default="/opt/models/PRISM-CLiMB")
    parser.add_argument(
        "--prism-manifest",
        help="Optional checkpoints.json override; defaults to PRISM-CLiMB/checkpoints.json",
    )
    parser.add_argument(
        "--depth-checkpoint-dir",
        help="Optional PRISM encoder.pth/depth.pth directory override",
    )
    parser.add_argument("--processing-width", type=int, default=720)
    parser.add_argument("--segment-length", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--min-pnp-inlier-ratio", type=float, default=0.08)
    parser.add_argument("--disable-motion-gate", action="store_true")
    parser.add_argument("--disable-step-rejection", action="store_true")
    parser.add_argument("--temporal-depth-alignment", action="store_true")
    parser.add_argument("--propagate-on-failure", action="store_true")
    parser.add_argument("--propagation-max-frames", type=int, default=3)
    parser.add_argument("--propagation-decay", type=float, default=0.80)
    parser.add_argument("--propagation-min-inlier-ratio", type=float, default=0.25)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--generic-intrinsics", action="store_true")
    parser.add_argument("--translation-smoothing-window", type=int)
    parser.add_argument("--translation-smoothing-polyorder", type=int, default=2)
    parser.add_argument("--rotation-smoothing-window", type=int)
    parser.add_argument("--rotation-smoothing-polyorder", type=int, default=1)
    return parser.parse_args()


def apply_profile(args: argparse.Namespace) -> None:
    if args.profile not in ("v2", "v3"):
        return
    args.segment_length = 12
    args.grid_size = 18
    args.disable_motion_gate = True
    args.propagate_on_failure = True
    args.propagation_max_frames = 3
    args.propagation_decay = 0.80
    args.propagation_min_inlier_ratio = 0.15
    args.temporal_depth_alignment = False
    args.mixed_precision = True
    if args.profile == "v3":
        if args.translation_smoothing_window is None:
            args.translation_smoothing_window = 21
        if args.rotation_smoothing_window is None:
            args.rotation_smoothing_window = 9


def validate_smoothing_args(args: argparse.Namespace) -> None:
    for label, window_length, polyorder in (
        (
            "translation",
            args.translation_smoothing_window,
            args.translation_smoothing_polyorder,
        ),
        (
            "rotation",
            args.rotation_smoothing_window,
            args.rotation_smoothing_polyorder,
        ),
    ):
        if polyorder < 0:
            raise SystemExit(f"{label} smoothing polyorder must be non-negative")
        if window_length in (None, 0):
            continue
        if window_length < 0 or window_length % 2 == 0 or window_length <= polyorder:
            raise SystemExit(
                f"{label} smoothing window must be odd and greater than its polyorder"
            )


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.mp4") if path.is_file())


def fallback_records(video_path: Path, max_frames: int) -> list[PoseRecord]:
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        fps = 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    frame_count = max(1, frame_count)
    if max_frames > 0:
        frame_count = min(frame_count, max_frames)
    identity = np.eye(4, dtype=np.float64)
    return [
        PoseRecord(
            frame_id=frame_id,
            timestamp=(frame_id - 1) / fps,
            camera_to_world=identity.copy(),
            extras=["fallback", "0.000000", "0", "inf", "0.000000"],
        )
        for frame_id in range(1, frame_count + 1)
    ]


def process_run(
    video_path: Path,
    run_root: Path,
    seed: int,
    config,
    backend: str,
    args: argparse.Namespace,
    shared_models=None,
    extra_init_seconds: float = 0.0,
) -> tuple[int, int, float, float]:
    init_start = time.perf_counter()
    if backend == "track2map":
        from track2map_frontend import Track2MapMonocularFrontend

        estimator = Track2MapMonocularFrontend(
            video_path=video_path,
            seed=seed,
            config=config,
            cotracker_repo=Path(args.cotracker_repo),
            cotracker_checkpoint=Path(args.cotracker_checkpoint),
            prism_climb_root=Path(args.prism_climb_root),
            prism_manifest=Path(args.prism_manifest) if args.prism_manifest else None,
            depth_checkpoint_dir=(
                Path(args.depth_checkpoint_dir) if args.depth_checkpoint_dir else None
            ),
            device=args.device,
            models=shared_models,
        )
    else:
        estimator = MonocularTrackVO(video_path=video_path, seed=seed, config=config)
    init_seconds = time.perf_counter() - init_start + extra_init_seconds

    processing_start = time.perf_counter()
    try:
        records, map_points = estimator.process()
    except Exception as error:
        print(f"    warning: {video_path.name}: VO failed ({error}); writing fallback trajectory", flush=True)
        records = fallback_records(video_path, config.max_frames)
        map_points = []
    finally:
        estimator.close()
    smooth_pose_records(
        records,
        translation_window_length=args.translation_smoothing_window or 0,
        translation_polyorder=args.translation_smoothing_polyorder,
        rotation_window_length=args.rotation_smoothing_window or 0,
        rotation_polyorder=args.rotation_smoothing_polyorder,
    )
    processing_seconds = time.perf_counter() - processing_start

    write_trajectory(run_root / "camera_trajectory" / "cam_traj_map_000.txt", records)
    write_points3d(run_root / "3D_maps" / "0" / "points3D.txt", map_points)
    write_runtime(run_root / "runtime.txt", init_seconds, processing_seconds)
    return len(records), len(map_points), init_seconds, processing_seconds


def main() -> None:
    total_start = time.perf_counter()
    args = parse_args()
    apply_profile(args)
    validate_smoothing_args(args)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    videos = find_videos(input_dir)
    if not videos:
        raise SystemExit(f"No .mp4 videos found directly under: {input_dir}")

    shared_models = None
    shared_model_init_seconds = 0.0
    if args.backend == "track2map":
        from track2map_frontend import Track2MapFrontendConfig, load_track2map_models

        config = Track2MapFrontendConfig(
            max_frames=args.max_frames,
            processing_width=args.processing_width,
            segment_length=args.segment_length,
            grid_size=args.grid_size,
            min_pnp_inlier_ratio=args.min_pnp_inlier_ratio,
            motion_gate_enabled=not args.disable_motion_gate,
            step_rejection_enabled=not args.disable_step_rejection,
            temporal_depth_alignment_enabled=args.temporal_depth_alignment,
            propagate_on_failure=args.propagate_on_failure,
            propagation_max_frames=args.propagation_max_frames,
            propagation_decay=args.propagation_decay,
            propagation_min_inlier_ratio=args.propagation_min_inlier_ratio,
            mixed_precision=args.mixed_precision,
            use_endomapper_calibration=not args.generic_intrinsics,
        )
        model_init_start = time.perf_counter()
        shared_models = load_track2map_models(
            cotracker_repo=Path(args.cotracker_repo),
            cotracker_checkpoint=Path(args.cotracker_checkpoint),
            prism_climb_root=Path(args.prism_climb_root),
            prism_manifest=Path(args.prism_manifest) if args.prism_manifest else None,
            depth_checkpoint_dir=(
                Path(args.depth_checkpoint_dir) if args.depth_checkpoint_dir else None
            ),
            device=args.device,
        )
        shared_model_init_seconds = time.perf_counter() - model_init_start
        print(f"shared model init={shared_model_init_seconds:.3f}s", flush=True)
    else:
        config = VOConfig(max_frames=args.max_frames, translation_step=args.translation_step)
    first_run = True
    for video_path in videos:
        sequence_start = time.perf_counter()
        print(f"{video_path.stem}: starting {args.num_runs} run(s)", flush=True)
        for run_id in range(1, args.num_runs + 1):
            run_root = output_dir / video_path.stem / str(run_id)
            frames, points, init_seconds, processing_seconds = process_run(
                video_path=video_path,
                run_root=run_root,
                seed=args.seed + run_id,
                config=config,
                backend=args.backend,
                args=args,
                shared_models=shared_models,
                extra_init_seconds=shared_model_init_seconds if first_run else 0.0,
            )
            first_run = False
            print(
                f"  run={run_id} frames={frames} points={points} "
                f"init={init_seconds:.3f}s processing={processing_seconds:.3f}s",
                flush=True,
            )
        print(f"{video_path.stem}: total={time.perf_counter() - sequence_start:.2f}s", flush=True)
    print(f"all sequences: total={time.perf_counter() - total_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
