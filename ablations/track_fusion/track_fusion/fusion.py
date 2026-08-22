"""Fuse sparse CoTracker points into the auxiliary input of pretrained PRISM models."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TrackRasterizer:
    """Turn `(N,2)` pixel coordinates into a dense `[0,1]` heat map."""

    radius: int = 3
    sigma: float = 2.0

    def __call__(
        self,
        points_xy: np.ndarray,
        visibility: np.ndarray,
        image_hw: tuple[int, int],
    ) -> np.ndarray:
        height, width = image_hw
        result = np.zeros((height, width), dtype=np.float32)
        points = np.asarray(points_xy, dtype=np.float32)
        visible = np.asarray(visibility, dtype=bool)
        valid = visible & np.isfinite(points).all(axis=1)
        points = np.rint(points[valid]).astype(np.int32)
        if len(points) == 0:
            return result
        inside = (
            (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height)
        )
        for x_coord, y_coord in points[inside]:
            cv2.circle(result, (int(x_coord), int(y_coord)), self.radius, 1.0, -1)
        if self.sigma > 0:
            result = cv2.GaussianBlur(result, (0, 0), self.sigma)
            maximum = float(result.max())
            if maximum > 0:
                result /= maximum
        return result


@dataclass
class FlowRasterizer:
    """Dense Farneback optical-flow magnitude, robustly mapped to `[0,1]`."""

    percentile: float = 95.0
    blur_sigma: float = 1.0

    def __call__(self, image0_bgr: np.ndarray, image1_bgr: np.ndarray) -> np.ndarray:
        if image0_bgr.shape != image1_bgr.shape:
            raise ValueError("Optical-flow image pair must have matching shapes")
        gray0 = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            gray0,
            gray1,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1]).astype(np.float32)
        finite = np.isfinite(magnitude)
        if not finite.any():
            return np.zeros(gray0.shape, dtype=np.float32)
        scale = float(np.percentile(magnitude[finite], self.percentile))
        if scale <= 1e-6:
            return np.zeros(gray0.shape, dtype=np.float32)
        result = np.clip(magnitude / scale, 0.0, 1.0)
        result[~finite] = 0.0
        if self.blur_sigma > 0:
            result = cv2.GaussianBlur(result, (0, 0), self.blur_sigma)
        return result.astype(np.float32, copy=False)


def fuse_auxiliary(
    auxiliary: np.ndarray,
    track_map: np.ndarray,
    *,
    scale: float,
    mode: str,
    alpha: float,
) -> np.ndarray:
    """Fuse maps while preserving the pretrained auxiliary channel's scale."""
    if auxiliary.shape != track_map.shape:
        track_map = cv2.resize(
            track_map, (auxiliary.shape[1], auxiliary.shape[0]), cv2.INTER_LINEAR
        )
    alpha = float(np.clip(alpha, 0.0, 1.0))
    track_scaled = np.clip(track_map, 0.0, 1.0) * scale
    base = auxiliary.astype(np.float32, copy=False)
    if mode == "blend":
        return ((1.0 - alpha) * base + alpha * track_scaled).astype(np.float32)
    if mode == "add":
        return np.clip(base + alpha * track_scaled, 0.0, scale).astype(np.float32)
    raise ValueError(f"Unsupported fusion mode: {mode}")


class TrackFusedDepth:
    """Original DLPE depth with IID shading/track fusion in channel four."""

    def __init__(self, depth_model, *, mode: str = "add", alpha: float = 0.20):
        self.model = depth_model
        self.mode = mode
        self.alpha = alpha

    def __call__(self, image_bgr: np.ndarray, track_map: np.ndarray) -> np.ndarray:
        shading = self.model.shading_generator(image_bgr)
        fused = fuse_auxiliary(
            shading, track_map, scale=255.0, mode=self.mode, alpha=self.alpha
        )
        return self.model.predict_with_shading(image_bgr, fused)


class TrackFusedPose:
    """Original DLPE pose with DexiNed edge/track fusion per frame."""

    def __init__(self, pose_model, *, mode: str = "add", alpha: float = 0.20):
        if getattr(pose_model, "auxiliary_type", None) != "edge":
            raise ValueError("TrackFusedPose requires the original DLPE edge pose model")
        self.model = pose_model
        self.mode = mode
        self.alpha = alpha

    def __call__(
        self,
        image0_bgr: np.ndarray,
        image1_bgr: np.ndarray,
        track_map0: np.ndarray,
        track_map1: np.ndarray,
    ) -> np.ndarray:
        edge0 = self.model.auxiliary_generator(image0_bgr)
        edge1 = self.model.auxiliary_generator(image1_bgr)
        fused0 = fuse_auxiliary(edge0, track_map0, scale=1.0, mode=self.mode, alpha=self.alpha)
        fused1 = fuse_auxiliary(edge1, track_map1, scale=1.0, mode=self.mode, alpha=self.alpha)
        return self.model.predict_with_auxiliary(image0_bgr, image1_bgr, fused0, fused1)
