"""Run CLiMB preprocessing, joint training, then frozen-depth pose tuning."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(
    "/raid/scratch_not_backed_up/xinwei/CLiMB/EndoMapper-CLiMB"
)
INITIAL_WEIGHTS = {
    "dlpe": Path(
        "/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/"
        "Weights_Feast_Ablation/monodepth0_joint_training/"
        "hk_mono_finetuned_dlpe_edge_ssim/models/weights_19"
    ),
    "dlpl": Path(
        "/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/"
        "Weights_Feast_Ablation/monodepth0_joint_training/"
        "hk_mono_finetuned_both_lum_edge_ssim/models/weights_19"
    ),
}


def execute(command: list[str], env: dict[str, str], dry_run: bool) -> None:
    print("+ {}".format(shlex.join(command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=TRAINING_ROOT, env=env, check=True)


def train_command(
    variant: str,
    model_name: str,
    log_dir: Path,
    frames_root: Path,
    split: str,
    load_weights: Path,
    epochs: int,
    batch_size: int,
    workers: int,
    save_frequency: int,
    pose_only: bool,
) -> list[str]:
    pose_aux = "edge" if variant == "dlpe" else "lum"
    command = [
        sys.executable,
        "train_prism.py",
        "--pipeline",
        "joint",
        "--model_name",
        model_name,
        "--dataset",
        "hk",
        "--data",
        "endomapper",
        "--data_path",
        str(frames_root),
        "--split",
        split,
        "--height",
        "288",
        "--width",
        "288",
        "--method",
        "monodepth2",
        "--training_mode",
        "both",
        "--depth_aux",
        "lum",
        "--pose_aux",
        pose_aux,
        "--edge_loss",
        "--load_weights_folder",
        str(load_weights),
        "--models_to_load",
        "encoder",
        "depth",
        "pose_encoder",
        "pose",
        "--log_dir",
        str(log_dir),
        "--num_epochs",
        str(epochs),
        "--scheduler_step_size",
        str(max(1, epochs // 2)),
        "--save_frequency",
        str(save_frequency),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(workers),
    ]
    if pose_only:
        command.append("--freeze_depth")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", nargs="+", choices=["dlpe", "dlpl"], default=["dlpe", "dlpl"]
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--joint-epochs", type=int, default=30)
    parser.add_argument("--pose-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-frequency", type=int, default=5)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.dataset_root / "prism_depth_climb"
    prepared_root = output_root / "prepared_{}fps".format(
        "{:g}".format(args.target_fps)
    )
    env = os.environ.copy()
    env["PRISM_ENDOMAPPER_GENERATED"] = str(prepared_root / "generated")

    if not args.skip_prepare:
        execute(
            [
                sys.executable,
                "prepare_endomapper.py",
                "--dataset-root",
                str(args.dataset_root),
                "--target-fps",
                str(args.target_fps),
                "--device",
                args.device,
            ],
            env,
            args.dry_run,
        )
    if args.prepare_only:
        return

    for variant in args.variants:
        initial = INITIAL_WEIGHTS[variant]
        if not args.dry_run and not initial.is_dir():
            raise FileNotFoundError(initial)
        variant_root = output_root / "training_runs" / variant
        joint_name = "endomapper_climb_{}_edge_ssim_joint".format(variant)
        pose_name = "endomapper_climb_{}_edge_ssim_pose_finetune".format(variant)
        execute(
            train_command(
                variant,
                joint_name,
                variant_root,
                prepared_root / "frames",
                "endomapper_climb",
                initial,
                args.joint_epochs,
                args.batch_size,
                args.workers,
                args.save_frequency,
                False,
            ),
            env,
            args.dry_run,
        )
        joint_checkpoint = (
            variant_root
            / joint_name
            / "models"
            / "weights_{}".format(args.joint_epochs - 1)
        )
        if not args.dry_run and not joint_checkpoint.is_dir():
            raise FileNotFoundError(joint_checkpoint)
        execute(
            train_command(
                variant,
                pose_name,
                variant_root,
                prepared_root / "frames",
                "endomapper_climb",
                joint_checkpoint,
                args.pose_epochs,
                args.batch_size,
                args.workers,
                args.save_frequency,
                True,
            ),
            env,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
