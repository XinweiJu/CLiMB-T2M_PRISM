"""Fine-tune DLPE/DLPL on six COLMAP clips and validate on two held-out clips."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_three_stage import INITIAL_WEIGHTS, train_command


DATA_ROOT = Path(
    "/raid/scratch_not_backed_up/xinwei/CLiMB/EndoMapper-CLiMB/"
    "prism_depth_climb/colmap8_adaptation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["dlpe", "dlpl"], required=True)
    parser.add_argument("--joint-epochs", type=int, default=10)
    parser.add_argument("--pose-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = DATA_ROOT / "prepared_fullfps"
    run_root = DATA_ROOT / "training_runs" / args.variant
    env = os.environ.copy()
    env["PRISM_ENDOMAPPER_GENERATED"] = str(prepared / "generated")
    joint_name = "colmap8_{}_edge_ssim_joint".format(args.variant)
    pose_name = "colmap8_{}_edge_ssim_pose_finetune".format(args.variant)

    joint = train_command(
        args.variant,
        joint_name,
        run_root,
        prepared / "frames",
        "endomapper_colmap8",
        INITIAL_WEIGHTS[args.variant],
        args.joint_epochs,
        args.batch_size,
        args.workers,
        5,
        False,
    )
    subprocess.run(joint, cwd=Path(__file__).resolve().parent, env=env, check=True)
    joint_checkpoint = (
        run_root / joint_name / "models"
        / "weights_{}".format(args.joint_epochs - 1)
    )
    pose = train_command(
        args.variant,
        pose_name,
        run_root,
        prepared / "frames",
        "endomapper_colmap8",
        joint_checkpoint,
        args.pose_epochs,
        args.batch_size,
        args.workers,
        5,
        True,
    )
    subprocess.run(pose, cwd=Path(__file__).resolve().parent, env=env, check=True)


if __name__ == "__main__":
    main()
