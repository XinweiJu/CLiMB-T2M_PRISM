#!/usr/bin/env python3
import json
from pathlib import Path

import torch


root = Path("/raid/scratch_not_backed_up/xinwei/Datasets/C3VD_Undistorted/Weights_PRISM_Flow_C3VD")
source = Path("/raid/scratch_not_backed_up/xinwei/Datasets/Weights_FEAST/Weights_Feast_new/monodepth/hk_mono_finetuned_dlpe_edge_ssim_depth_fix_thorough2/models/weights_19")
result = {}
for name in (
    "c3vd_dlpe_flow_add_depth", "c3vd_dlpe_flow_add_pose",
    "c3vd_dlpe_flow5_depth", "c3vd_dlpe_flow5_pose",
):
    checkpoint = root / name / "models" / "weights_9"
    encoder = torch.load(checkpoint / "encoder.pth", map_location="cpu")
    pose_encoder = torch.load(checkpoint / "pose_encoder.pth", map_location="cpu")
    frozen_files = ("pose_encoder.pth", "pose.pth") if name.endswith("depth") else ("encoder.pth", "depth.pth")
    identical = {}
    for filename in frozen_files:
        current = torch.load(checkpoint / filename, map_location="cpu")
        original = torch.load(source / filename, map_location="cpu")
        comparable = [
            (key, value) for key, value in current.items()
            if torch.is_tensor(value) and key in original and value.shape == original[key].shape
        ]
        identical[filename] = all(
            torch.equal(value, original[key])
            for key, value in comparable
        )
    result[name] = {
        "encoder_conv1": list(encoder["encoder.conv1.weight"].shape),
        "pose_encoder_conv1": list(pose_encoder["encoder.conv1.weight"].shape),
        "frozen_identical": identical,
    }
print(json.dumps(result, indent=2))
