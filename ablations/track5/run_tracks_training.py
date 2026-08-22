#!/usr/bin/env python3
"""Launch depth-only or pose-only Track5 fine-tuning."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINING = ROOT.parent / "prism_depth_climb" / "training"
SOURCE = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/Weights_Feast_new/monodepth/hk_mono_finetuned_dlpe_edge_ssim_depth_fix_thorough2/models/weights_19")
FRAMES = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Hyper-Kvasir/BBPS-2-3Frames/Undistorted/Frames")
SHADING = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Hyper-Kvasir/BBPS-2-3Undistorted_Shading")
EDGE = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Hyper-Kvasir/BBPS-2-3Undistorted_Edge")
OUTPUT = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/PRISM_Track5_HK")
MOTION = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Hyper-Kvasir/BBPS-2-3Motion")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=("depth", "pose"), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--benchmark-batches", type=int, default=0)
    args = parser.parse_args()
    model_name = "hk_dlpe_track5_{}".format(args.experiment)
    command = [
        "python", str(TRAINING / "train_prism.py"), "--pipeline", "joint",
        "--method", "monodepth2", "--dataset", "hk", "--data", "hk",
        "--split", "hk_track5", "--data_path", str(FRAMES), "--png",
        "--height", "288", "--width", "288", "--frame_ids", "0", "-1", "1",
        "--depth_aux", "lum", "--pose_aux", "edge", "--motion_aux", "tracks",
        "--motion_root", str(MOTION), "--shading_root", str(SHADING),
        "--edge_root", str(EDGE), "--load_weights_folder", str(SOURCE),
        "--models_to_load", "encoder", "depth", "pose_encoder", "pose",
        "--log_dir", str(OUTPUT), "--model_name", model_name,
        "--num_epochs", str(args.epochs), "--batch_size", str(args.batch_size),
        "--num_workers", str(args.workers), "--save_frequency", "5",
        "--learning_rate", "0.00005", "--scheduler_step_size", "7",
    ]
    command.append("--freeze_pose" if args.experiment == "depth" else "--freeze_depth")
    if args.benchmark_batches:
        command.extend(["--benchmark_batches", str(args.benchmark_batches)])
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=TRAINING, check=True)


if __name__ == "__main__":
    main()
