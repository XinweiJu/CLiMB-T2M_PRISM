#!/usr/bin/env python3
"""Fine-tune DLPE with official rectified C3VD flow: flow-add or flow5."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINING = ROOT.parent / "prism_depth_climb" / "training"
SOURCE = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/Weights_Feast_new/monodepth/hk_mono_finetuned_dlpe_edge_ssim_depth_fix_thorough2/models/weights_19")
DATA_ROOT = Path("/raid/scratch_not_backed_up/xinwei/Datasets/C3VD_Undistorted")
OUTPUT = DATA_ROOT / "Weights_PRISM_Flow_C3VD"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("flow-add", "flow5"), required=True)
    parser.add_argument("--experiment", choices=("depth", "pose"), required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--benchmark-batches", type=int, default=0)
    args = parser.parse_args()
    name = f"c3vd_dlpe_{args.variant.replace('-', '_')}_{args.experiment}"
    command = [
        "python", str(TRAINING / "train_prism.py"), "--pipeline", "joint",
        "--method", "monodepth2", "--dataset", "hk", "--data", "c3vd",
        "--split", "c3vd_flow", "--data_path", str(DATA_ROOT / "Dataset"), "--png",
        "--height", "288", "--width", "288", "--frame_ids", "0", "-1", "1",
        "--depth_aux", "lum", "--pose_aux", "edge", "--motion_aux", "flow",
        "--motion_root", str(DATA_ROOT / "Motion"),
        "--shading_root", str(DATA_ROOT / "Shading"), "--edge_root", str(DATA_ROOT / "Edge"),
        "--load_weights_folder", str(SOURCE), "--models_to_load", "encoder", "depth", "pose_encoder", "pose",
        "--log_dir", str(OUTPUT), "--model_name", name, "--num_epochs", str(args.epochs),
        "--batch_size", str(args.batch_size), "--num_workers", str(args.workers),
        "--save_frequency", "5", "--learning_rate", "0.00005", "--scheduler_step_size", "7",
        "--motion_fusion", "add" if args.variant == "flow-add" else "channel",
    ]
    command.append("--freeze_pose" if args.experiment == "depth" else "--freeze_depth")
    if args.benchmark_batches:
        command.extend(["--benchmark_batches", str(args.benchmark_batches)])
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=TRAINING, check=True)


if __name__ == "__main__":
    main()
