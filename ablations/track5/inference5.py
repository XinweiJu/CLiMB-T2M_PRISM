"""Inference interfaces for learned RGB+DLPE-auxiliary+track models."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from prism_depth_climb import EdgeGenerator, ShadingGenerator
from prism_depth_climb.layers import disp_to_depth, transformation_from_parameters
from prism_depth_climb.networks import DepthDecoder, PoseDecoder, ResnetEncoder


def _rgb(image_bgr, size, device):
    height, width = size
    resized = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(
        np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    ).div_(255.0).unsqueeze(0).to(device)


def _map(array, size, device, scale):
    height, width = size
    resized = cv2.resize(array.astype(np.float32), (width, height), cv2.INTER_LINEAR)
    return torch.from_numpy(np.ascontiguousarray(resized))[None, None].div_(scale).to(device)


def _state(path, device):
    value = torch.load(path, map_location=device)
    return {key: tensor for key, tensor in value.items() if isinstance(tensor, torch.Tensor)}


class Track5Depth:
    def __init__(self, checkpoint_dir, *, device="cuda", manifest_path=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = Path(checkpoint_dir)
        encoder_state = torch.load(checkpoint / "encoder.pth", map_location=self.device)
        self.height = int(encoder_state.get("height", 288))
        self.width = int(encoder_state.get("width", 288))
        self.encoder = ResnetEncoder(18, False)
        self.encoder.encoder.conv1 = torch.nn.Conv2d(5, 64, 7, 2, 3, bias=False)
        self.encoder.load_state_dict(_state(checkpoint / "encoder.pth", self.device), strict=True)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))
        self.decoder.load_state_dict(_state(checkpoint / "depth.pth", self.device), strict=True)
        self.encoder.to(self.device).eval(); self.decoder.to(self.device).eval()
        kwargs = {"device": self.device}
        if manifest_path is not None:
            kwargs["manifest_path"] = manifest_path
        self.shading = ShadingGenerator(**kwargs)

    @torch.inference_mode()
    def __call__(self, image_bgr, track_map):
        original_hw = image_bgr.shape[:2]
        size = (self.height, self.width)
        shading = self.shading(image_bgr)
        tensor = torch.cat([
            _rgb(image_bgr, size, self.device),
            _map(shading, size, self.device, 1.0),
            _map(track_map, size, self.device, 1.0),
        ], dim=1)
        raw = self.decoder(self.encoder(tensor))[("disp", 0)]
        disparity, _ = disp_to_depth(raw, 0.1, 100.0)
        disparity = F.interpolate(disparity, original_hw, mode="bilinear", align_corners=False)
        return disparity[0, 0].cpu().numpy().astype(np.float32)


class Track5Pose:
    def __init__(self, checkpoint_dir, *, device="cuda", manifest_path=None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = Path(checkpoint_dir)
        state = torch.load(checkpoint / "pose_encoder.pth", map_location=self.device)
        self.height = int(state.get("height", 288))
        self.width = int(state.get("width", 288))
        self.encoder = ResnetEncoder(18, False, num_input_images=2)
        self.encoder.encoder.conv1 = torch.nn.Conv2d(10, 64, 7, 2, 3, bias=False)
        self.encoder.load_state_dict(_state(checkpoint / "pose_encoder.pth", self.device), strict=True)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, 1, 2)
        self.decoder.load_state_dict(_state(checkpoint / "pose.pth", self.device), strict=True)
        self.encoder.to(self.device).eval(); self.decoder.to(self.device).eval()
        kwargs = {"device": self.device}
        if manifest_path is not None:
            kwargs["manifest_path"] = manifest_path
        self.edge = EdgeGenerator(**kwargs)

    def _frame(self, image, track):
        size = (self.height, self.width)
        edge = self.edge(image)
        return torch.cat([
            _rgb(image, size, self.device),
            _map(edge, size, self.device, 1.0),
            _map(track, size, self.device, 1.0),
        ], dim=1)

    @torch.inference_mode()
    def __call__(self, image0, image1, track0, track1):
        pair = torch.cat([self._frame(image0, track0), self._frame(image1, track1)], dim=1)
        axisangle, translation = self.decoder([self.encoder(pair)])
        matrix = transformation_from_parameters(axisangle[:, 0], translation[:, 0])
        return matrix[0].cpu().numpy().astype(np.float32)


class FlowAddDepth:
    """Four-channel DLPE depth trained with flow added to IID shading."""

    def __init__(self, checkpoint_dir, *, device="cuda", manifest_path=None, alpha=0.2):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alpha = alpha
        checkpoint = Path(checkpoint_dir)
        state = torch.load(checkpoint / "encoder.pth", map_location=self.device)
        self.height, self.width = int(state.get("height", 288)), int(state.get("width", 288))
        self.encoder = ResnetEncoder(18, False)
        self.encoder.encoder.conv1 = torch.nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        self.encoder.load_state_dict(_state(checkpoint / "encoder.pth", self.device), strict=True)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))
        self.decoder.load_state_dict(_state(checkpoint / "depth.pth", self.device), strict=True)
        self.encoder.to(self.device).eval(); self.decoder.to(self.device).eval()
        kwargs = {"device": self.device}
        if manifest_path is not None: kwargs["manifest_path"] = manifest_path
        self.shading = ShadingGenerator(**kwargs)

    @torch.inference_mode()
    def __call__(self, image_bgr, flow_map):
        original_hw = image_bgr.shape[:2]; size = (self.height, self.width)
        shading = self.shading(image_bgr)
        fused = np.clip(shading.astype(np.float32) + self.alpha * flow_map.astype(np.float32) * 255.0, 0, 255)
        tensor = torch.cat([_rgb(image_bgr, size, self.device), _map(fused, size, self.device, 1.0)], 1)
        raw = self.decoder(self.encoder(tensor))[("disp", 0)]
        disparity, _ = disp_to_depth(raw, 0.1, 100.0)
        return F.interpolate(disparity, original_hw, mode="bilinear", align_corners=False)[0, 0].cpu().numpy().astype(np.float32)


class FlowAddPose:
    """Eight-channel DLPE pose trained with flow added to DexiNed edge."""

    def __init__(self, checkpoint_dir, *, device="cuda", manifest_path=None, alpha=0.2):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.alpha = alpha; checkpoint = Path(checkpoint_dir)
        state = torch.load(checkpoint / "pose_encoder.pth", map_location=self.device)
        self.height, self.width = int(state.get("height", 288)), int(state.get("width", 288))
        self.encoder = ResnetEncoder(18, False, num_input_images=2)
        self.encoder.encoder.conv1 = torch.nn.Conv2d(8, 64, 7, 2, 3, bias=False)
        self.encoder.load_state_dict(_state(checkpoint / "pose_encoder.pth", self.device), strict=True)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, 1, 2)
        self.decoder.load_state_dict(_state(checkpoint / "pose.pth", self.device), strict=True)
        self.encoder.to(self.device).eval(); self.decoder.to(self.device).eval()
        kwargs = {"device": self.device}
        if manifest_path is not None: kwargs["manifest_path"] = manifest_path
        self.edge = EdgeGenerator(**kwargs)

    def _frame(self, image, flow):
        size = (self.height, self.width); edge = self.edge(image)
        fused = np.clip(edge.astype(np.float32) + self.alpha * flow.astype(np.float32), 0, 1)
        return torch.cat([_rgb(image, size, self.device), _map(fused, size, self.device, 1.0)], 1)

    @torch.inference_mode()
    def __call__(self, image0, image1, flow0, flow1):
        pair = torch.cat([self._frame(image0, flow0), self._frame(image1, flow1)], 1)
        axisangle, translation = self.decoder([self.encoder(pair)])
        return transformation_from_parameters(axisangle[:, 0], translation[:, 0])[0].cpu().numpy().astype(np.float32)
