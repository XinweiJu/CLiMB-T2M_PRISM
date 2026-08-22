#!/usr/bin/env bash
set -euo pipefail

gpu="${PRISM_GPU:-1}"
image="${PRISM_TRAIN_IMAGE:-prism-depth-climb:train}"
workspace="/home/xju/Workspace/PRISM-CLiMB"
raid="/raid/scratch_not_backed_up"

exec docker run --rm \
  --gpus "device=${gpu}" \
  --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "${workspace}:/workspace" \
  -v "${raid}:${raid}" \
  -w /workspace/prism_depth_climb/training \
  "${image}" "$@"
