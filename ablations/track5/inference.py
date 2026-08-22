"""Inference interfaces for trained RGB+original-auxiliary+track models."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
PRISM_ROOT = ROOT.parent
sys.path.insert(0, str(PRISM_ROOT))
sys.path.insert(0, str(PRISM_ROOT / "PRISM-Pose-CLiMB"))

from prism_depth_climb.inference import ShadingGenerator, _aux_tensor, _load_state, _rgb_tensor
from prism_depth_climb.layers import disp_to_depth
from prism_depth_climb.networks import DepthDecoder, ResnetEncoder
from prism_pose.inference import EdgeGenerator
from prism_pose.layers import transformation_from_parameters
from prism_pose.networks import PoseDecoder


class Track5DepthGenerator:
    def __init__(self, checkpoint_dir, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        state = _load_state(checkpoint_dir / "encoder.pth", self.device)
        self.height = int(state.get("height", 288)); self.width = int(state.get("width", 288))
        self.encoder = ResnetEncoder(18, False)
        self.encoder.encoder.conv1 = torch.nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        self.encoder.load_state_dict({k: v for k, v in state.items() if k in self.encoder.state_dict()}, strict=True)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))
        self.decoder.load_state_dict(_load_state(checkpoint_dir / "depth.pth", self.device), strict=True)
        self.encoder.to(self.device).eval(); self.decoder.to(self.device).eval()
        self.shading = ShadingGenerator(device=self.device, manifest_path=PRISM_ROOT / "checkpoints.json")

    @torch.inference_mode()
    def __call__(self, image_bgr, track_map):
        original_hw = image_bgr.shape[:2]
        rgb = _rgb_tensor(image_bgr, (self.height, self.width), self.device)
        lum = _aux_tensor(self.shading(image_bgr), (self.height, self.width), self.device, "shading")
        tracks = _aux_tensor(track_map, (self.height, self.width), self.device, "tracks")
        raw = self.decoder(self.encoder(torch.cat([rgb, lum, tracks], 1)))[("disp", 0)]
        disp, _ = disp_to_depth(raw, 0.1, 100.0)
        return F.interpolate(disp, original_hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)


class Track5PoseGenerator:
    def __init__(self, checkpoint_dir, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint_dir = Path(checkpoint_dir)
        state = _load_state(checkpoint_dir / "pose_encoder.pth", self.device)
        self.height = int(state.get("height", 288)); self.width = int(state.get("width", 288))
        self.encoder = ResnetEncoder(18, False, num_input_images=2)
        self.encoder.encoder.conv1 = torch.nn.Conv2d(10, 64, 7, 2, 3, bias=False)
        self.encoder.load_state_dict(state, strict=True)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, 1, 2)
        self.decoder.load_state_dict(_load_state(checkpoint_dir / "pose.pth", self.device), strict=True)
        self.encoder.to(self.device).eval(); self.decoder.to(self.device).eval()
        config = {
            "height": 512, "width": 512,
            "mean_bgr": [103.939, 116.779, 123.68],
            "checkpoint_path": "/raid/scratch_not_backed_up/xinwei/Workspace_backup/Baselines/z_Other_models/DexiNed/checkpoints/16_model.pth",
        }
        self.edge = EdgeGenerator(config, self.device)

    def frame(self, image, edge, tracks):
        size = (self.height, self.width)
        return torch.cat([
            _rgb_tensor(image, size, self.device),
            _aux_tensor(edge, size, self.device, "edge"),
            _aux_tensor(tracks, size, self.device, "tracks"),
        ], 1)

    @torch.inference_mode()
    def __call__(self, image0, image1, tracks0, tracks1):
        pair = torch.cat([self.frame(image0, self.edge(image0), tracks0), self.frame(image1, self.edge(image1), tracks1)], 1)
        axisangle, translation = self.decoder([self.encoder(pair)])
        return transformation_from_parameters(axisangle[:, 0], translation[:, 0])[0].cpu().numpy().astype(np.float32)
