"""Benchmark representative DLPE joint-training batch sizes on one GPU."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATASET = Path("/raid/scratch_not_backed_up/xinwei/CLiMB/EndoMapper-CLiMB")
INITIAL = Path(
    "/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/"
    "Weights_Feast_Ablation/monodepth0_joint_training/"
    "hk_mono_finetuned_dlpe_edge_ssim/models/weights_19"
)
PATTERN = re.compile(r"^BENCHMARK .*$", re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[12, 24, 36, 48])
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared = DATASET / "prism_depth_climb" / "prepared_10fps"
    env = os.environ.copy()
    env["PRISM_ENDOMAPPER_GENERATED"] = str(prepared / "generated")
    summaries: list[str] = []

    for batch_size in args.batch_sizes:
        command = [
            sys.executable,
            "train_prism.py",
            "--pipeline", "joint",
            "--model_name", "benchmark_dlpe_edge_ssim_bs{}".format(batch_size),
            "--dataset", "hk",
            "--data", "endomapper",
            "--data_path", str(prepared / "frames"),
            "--split", "endomapper_climb",
            "--height", "288",
            "--width", "288",
            "--method", "monodepth2",
            "--training_mode", "both",
            "--depth_aux", "lum",
            "--pose_aux", "edge",
            "--edge_loss",
            "--load_weights_folder", str(INITIAL),
            "--log_dir", "/tmp/prism_depth_climb_benchmark",
            "--num_epochs", "1",
            "--save_frequency", "99",
            "--batch_size", str(batch_size),
            "--num_workers", str(args.workers),
            "--benchmark_batches", str(args.batches),
        ]
        print("Testing batch size {}...".format(batch_size), flush=True)
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        match = PATTERN.search(result.stdout)
        if result.returncode == 0 and match:
            summaries.append(match.group(0))
            print(match.group(0), flush=True)
        else:
            tail = "\n".join(result.stdout.splitlines()[-15:])
            summaries.append(
                "BENCHMARK batch_size={} FAILED returncode={}".format(
                    batch_size, result.returncode
                )
            )
            print(tail, flush=True)
            if "out of memory" in result.stdout.lower():
                print("Stopping after CUDA OOM.", flush=True)
                break

    print("\nSUMMARY")
    print("\n".join(summaries))


if __name__ == "__main__":
    main()
