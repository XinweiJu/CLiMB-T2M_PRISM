"""Track2Map-inspired monocular pose frontend for the CLiMB benchmark."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

from climb_io import MapPoint, PoseRecord


@dataclass
class Track2MapFrontendConfig:
    max_frames: int = 0
    processing_width: int = 720
    segment_length: int = 64
    grid_size: int = 20
    assumed_focal_scale: float = 0.505
    use_endomapper_calibration: bool = True
    min_pnp_points: int = 24
    min_pnp_inliers: int = 20
    min_pnp_inlier_ratio: float = 0.08
    pnp_reprojection_error: float = 3.0
    max_direction_dispersion: float = 1.35
    radial_alignment_threshold: float = 0.55
    strong_pnp_inlier_ratio: float = 0.70
    motion_gate_enabled: bool = True
    temporal_depth_alignment_enabled: bool = False
    depth_alignment_min_points: int = 20
    depth_alignment_max_scale_change: float = 2.0
    depth_alignment_momentum: float = 0.25
    depth_alignment_max_gap_frames: int = 3
    low_confidence_ratio: float = 0.25
    low_confidence_max_translation_step: float = 0.10
    low_confidence_max_rotation_step_deg: float = 3.0
    max_translation_step: float = 0.25
    max_rotation_step_deg: float = 8.0
    reject_translation_step: float = 1.0
    reject_rotation_step_deg: float = 30.0
    step_rejection_enabled: bool = True
    propagate_on_failure: bool = False
    propagation_max_frames: int = 3
    propagation_decay: float = 0.80
    propagation_min_inlier_ratio: float = 0.25
    mixed_precision: bool = False
    max_map_points: int = 12000


@dataclass
class DepthAlignmentSamples:
    image_points: np.ndarray
    camera_depths: np.ndarray


@dataclass
class PropagationState:
    relative_pose: np.ndarray | None = None
    age: int = 0


@dataclass
class Track2MapModelBundle:
    torch: Any
    device: Any
    cotracker: Any
    depth_model: Any


def load_track2map_models(
    cotracker_repo: Path,
    cotracker_checkpoint: Path,
    prism_climb_root: Path,
    prism_manifest: Path | None = None,
    depth_checkpoint_dir: Path | None = None,
    device: str = "cuda",
) -> Track2MapModelBundle:
    import torch

    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    cotracker_repo = cotracker_repo.resolve()
    cotracker_checkpoint = cotracker_checkpoint.resolve()
    prism_climb_root = prism_climb_root.resolve()
    if not (cotracker_repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"Invalid CoTracker repository: {cotracker_repo}")
    if not cotracker_checkpoint.is_file():
        raise FileNotFoundError(f"Missing CoTracker checkpoint: {cotracker_checkpoint}")
    if not (prism_climb_root / "prism_climb" / "inference.py").is_file():
        raise FileNotFoundError(f"Invalid PRISM-CLiMB repository: {prism_climb_root}")
    if str(prism_climb_root) not in sys.path:
        sys.path.insert(0, str(prism_climb_root))

    cotracker = torch.hub.load(
        str(cotracker_repo),
        "cotracker3_online",
        source="local",
        pretrained=False,
    ).to(resolved_device).eval()
    cotracker_state = torch.load(cotracker_checkpoint, map_location="cpu", weights_only=True)
    cotracker.model.load_state_dict(cotracker_state)

    from prism_climb import DepthGenerator

    depth_kwargs: dict[str, Any] = {"device": resolved_device}
    if prism_manifest is not None:
        depth_kwargs["manifest_path"] = prism_manifest.resolve()
    depth_model = DepthGenerator(
        checkpoint_dir=depth_checkpoint_dir.resolve()
        if depth_checkpoint_dir is not None
        else None,
        **depth_kwargs,
    )
    return Track2MapModelBundle(
        torch=torch,
        device=resolved_device,
        cotracker=cotracker,
        depth_model=depth_model,
    )


class Track2MapMonocularFrontend:
    """Bounded-window CoTracker3 tracking with PRISM anchor-depth PnP recovery."""

    def __init__(
        self,
        video_path: Path,
        seed: int,
        config: Track2MapFrontendConfig,
        cotracker_repo: Path,
        cotracker_checkpoint: Path,
        prism_climb_root: Path,
        prism_manifest: Path | None = None,
        depth_checkpoint_dir: Path | None = None,
        device: str = "cuda",
        models: Track2MapModelBundle | None = None,
    ):
        if models is None:
            models = load_track2map_models(
                cotracker_repo=cotracker_repo,
                cotracker_checkpoint=cotracker_checkpoint,
                prism_climb_root=prism_climb_root,
                prism_manifest=prism_manifest,
                depth_checkpoint_dir=depth_checkpoint_dir,
                device=device,
            )
        torch = models.torch

        self.video_path = video_path
        self.config = config
        self.random = np.random.default_rng(seed)
        cv2.setRNGSeed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.torch = torch
        self.device = models.device
        self.cotracker = models.cotracker
        self.depth_model = models.depth_model
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            self.fps = 30.0
        source_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_width <= 0 or source_height <= 0:
            raise RuntimeError(f"Invalid video dimensions: {source_width}x{source_height}")

        self.width = min(config.processing_width, source_width)
        self.height = int(round(source_height * self.width / source_width))
        if config.use_endomapper_calibration:
            scale_x = self.width / 720.0
            scale_y = self.height / 540.0
            self.intrinsics = np.array(
                [
                    [363.59255 * scale_x, 0.0, 369.09085 * scale_x],
                    [0.0, 364.29770 * scale_y, 268.70015 * scale_y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self.fisheye_distortion = np.array(
                [-0.1311029047, -0.0051492471, 0.0015123570, -0.0000699845],
                dtype=np.float64,
            )
        else:
            focal = config.assumed_focal_scale * self.width
            self.intrinsics = np.array(
                [
                    [focal, 0.0, self.width / 2.0],
                    [0.0, focal, self.height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            self.fisheye_distortion = None

        self.depth_multiplier: float | None = None
        self.last_depth_alignment_count = 0

    def close(self) -> None:
        self.capture.release()

    def _read_frame(self) -> np.ndarray | None:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame

    def _read_segment(self, overlap_frame: np.ndarray | None) -> tuple[list[np.ndarray], bool]:
        frames = [] if overlap_frame is None else [overlap_frame]
        reached_end = False
        while len(frames) < self.config.segment_length:
            if self.config.max_frames > 0 and self._decoded_frames >= self.config.max_frames:
                reached_end = True
                break
            frame = self._read_frame()
            if frame is None:
                reached_end = True
                break
            frames.append(frame)
            self._decoded_frames += 1
        return frames, reached_end

    def _track_segment(self, frames_bgr: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        original_length = len(frames_bgr)
        frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
        minimum_length = self.cotracker.step + 1
        while len(frames_rgb) < minimum_length:
            frames_rgb.append(frames_rgb[-1].copy())

        video = (
            torch.from_numpy(np.stack(frames_rgb, axis=0))
            .permute(0, 3, 1, 2)[None]
            .float()
            .to(self.device)
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.config.mixed_precision and self.device.type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            self.cotracker(
                video_chunk=video[:, : self.cotracker.step * 2],
                is_first_step=True,
                grid_size=self.config.grid_size,
            )
            tracks = visibility = None
            for start in range(0, video.shape[1] - self.cotracker.step, self.cotracker.step):
                tracks, visibility = self.cotracker(
                    video_chunk=video[:, start : start + self.cotracker.step * 2],
                    is_first_step=False,
                    grid_size=0,
                )
        if tracks is None or visibility is None:
            raise RuntimeError("CoTracker did not return tracks for the current segment")
        tracks_np = tracks[0, :original_length].detach().cpu().numpy().astype(np.float32)
        visibility_np = visibility[0, :original_length].detach().cpu().numpy().astype(bool)
        del video, tracks, visibility
        return tracks_np, visibility_np

    @staticmethod
    def _sample_depth_values(
        depth: np.ndarray,
        image_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_coords = np.rint(image_points[:, 0]).astype(np.int32)
        y_coords = np.rint(image_points[:, 1]).astype(np.int32)
        valid = (
            (x_coords >= 0)
            & (x_coords < depth.shape[1])
            & (y_coords >= 0)
            & (y_coords < depth.shape[0])
        )
        values = np.zeros(len(image_points), dtype=np.float32)
        values[valid] = depth[y_coords[valid], x_coords[valid]]
        valid &= np.isfinite(values) & (values > 1e-6)
        return values, valid

    def _align_depth_multiplier(
        self,
        unscaled_depth: np.ndarray,
        samples: DepthAlignmentSamples,
    ) -> tuple[float, int]:
        current_multiplier = float(self.depth_multiplier or 1.0)
        sampled_depth, sampled_valid = self._sample_depth_values(
            unscaled_depth,
            samples.image_points,
        )
        target_depth = np.asarray(samples.camera_depths, dtype=np.float64)
        valid = sampled_valid & np.isfinite(target_depth) & (target_depth > 1e-6)
        if int(np.sum(valid)) < self.config.depth_alignment_min_points:
            return current_multiplier, 0

        log_ratios = np.log(target_depth[valid] / sampled_depth[valid])
        median_log_ratio = float(np.median(log_ratios))
        absolute_deviation = np.abs(log_ratios - median_log_ratio)
        median_absolute_deviation = float(np.median(absolute_deviation))
        robust_threshold = max(3.0 * 1.4826 * median_absolute_deviation, 0.10)
        robust = absolute_deviation <= robust_threshold
        if int(np.sum(robust)) < self.config.depth_alignment_min_points:
            return current_multiplier, 0

        estimated_multiplier = float(np.exp(np.median(log_ratios[robust])))
        max_change = max(self.config.depth_alignment_max_scale_change, 1.0)
        estimated_multiplier = float(
            np.clip(
                estimated_multiplier,
                current_multiplier / max_change,
                current_multiplier * max_change,
            )
        )
        momentum = float(np.clip(self.config.depth_alignment_momentum, 0.0, 1.0))
        aligned_multiplier = float(
            np.exp(
                momentum * np.log(max(current_multiplier, 1e-8))
                + (1.0 - momentum) * np.log(max(estimated_multiplier, 1e-8))
            )
        )
        return aligned_multiplier, int(np.sum(robust))

    def _estimate_anchor_depth(
        self,
        anchor_bgr: np.ndarray,
        alignment_samples: DepthAlignmentSamples | None = None,
    ) -> np.ndarray:
        autocast = (
            self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
            if self.config.mixed_precision and self.device.type == "cuda"
            else nullcontext()
        )
        with autocast:
            # PRISM DepthGenerator accepts OpenCV BGR uint8 and returns inverse
            # depth/disparity as float32 with the original (H, W).
            depth_response = self.depth_model(anchor_bgr).astype(
                np.float32, copy=False
            )
        finite = np.isfinite(depth_response) & (depth_response > 1e-6)
        if not np.any(finite):
            raise RuntimeError("PRISM depth returned no finite inverse-depth values")
        low, high = np.quantile(depth_response[finite], [0.01, 0.99])
        inverse_depth = np.clip(depth_response, max(float(low), 1e-6), float(high))
        unscaled_depth = 1.0 / inverse_depth
        self.last_depth_alignment_count = 0
        if self.depth_multiplier is None:
            median_depth = float(np.median(unscaled_depth[finite]))
            self.depth_multiplier = 1.0 / max(median_depth, 1e-6)
        elif self.config.temporal_depth_alignment_enabled and alignment_samples is not None:
            self.depth_multiplier, self.last_depth_alignment_count = self._align_depth_multiplier(
                unscaled_depth,
                alignment_samples,
            )
        depth = unscaled_depth * self.depth_multiplier
        depth[~finite] = 0.0
        return depth

    def _geometry_tracks(self, tracks: np.ndarray) -> np.ndarray:
        if self.fisheye_distortion is None:
            return tracks
        geometry_tracks = np.empty_like(tracks, dtype=np.float32)
        for frame_index in range(tracks.shape[0]):
            geometry_tracks[frame_index] = cv2.fisheye.undistortPoints(
                tracks[frame_index].reshape(-1, 1, 2).astype(np.float64),
                self.intrinsics,
                self.fisheye_distortion,
                P=self.intrinsics,
            ).reshape(-1, 2).astype(np.float32)
        return geometry_tracks

    @staticmethod
    def _flow_statistics(
        previous_points: np.ndarray,
        current_points: np.ndarray,
        valid: np.ndarray,
        center: np.ndarray,
    ) -> tuple[float, float]:
        if int(np.sum(valid)) == 0:
            return float("inf"), 0.0
        previous = previous_points[valid]
        current = current_points[valid]
        flow = current - previous
        magnitude = np.linalg.norm(flow, axis=1)
        moving = magnitude > 1e-4
        if not np.any(moving):
            return float("inf"), 0.0
        previous = previous[moving]
        flow = flow[moving]
        magnitude = magnitude[moving]
        directions = np.arctan2(flow[:, 1], flow[:, 0])
        resultant = np.hypot(np.mean(np.cos(directions)), np.mean(np.sin(directions)))
        resultant = float(np.clip(resultant, 1e-8, 1.0))
        direction_dispersion = float(np.sqrt(-2.0 * np.log(resultant)))

        radial = previous - center.reshape(1, 2)
        radial_norm = np.linalg.norm(radial, axis=1)
        radial_valid = radial_norm > 1.0
        if not np.any(radial_valid):
            return direction_dispersion, 0.0
        cosine = np.sum(flow[radial_valid] * radial[radial_valid], axis=1) / (
            magnitude[radial_valid] * radial_norm[radial_valid] + 1e-8
        )
        return direction_dispersion, float(np.median(np.abs(cosine)))

    def _lift_anchor_points(
        self,
        anchor_points: np.ndarray,
        anchor_depth: np.ndarray,
        depth_sample_points: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_points = anchor_points if depth_sample_points is None else depth_sample_points
        x_coords = np.rint(sample_points[:, 0]).astype(np.int32)
        y_coords = np.rint(sample_points[:, 1]).astype(np.int32)
        in_bounds = (
            (x_coords >= 0)
            & (x_coords < self.width)
            & (y_coords >= 0)
            & (y_coords < self.height)
        )
        depths = np.zeros(len(anchor_points), dtype=np.float32)
        depths[in_bounds] = anchor_depth[y_coords[in_bounds], x_coords[in_bounds]]
        valid = in_bounds & np.isfinite(depths) & (depths > 1e-4)
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        points_3d = np.zeros((len(anchor_points), 3), dtype=np.float32)
        points_3d[:, 2] = depths
        points_3d[:, 0] = (anchor_points[:, 0] - cx) / fx * depths
        points_3d[:, 1] = (anchor_points[:, 1] - cy) / fy * depths
        return points_3d, valid

    def _estimate_pose_pnp(
        self,
        anchor_points_3d: np.ndarray,
        current_points: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray, float]:
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < self.config.min_pnp_points:
            return None, np.zeros(len(valid), dtype=bool), 0.0
        object_points = anchor_points_3d[valid_indices].astype(np.float32)
        image_points = current_points[valid_indices].astype(np.float32)
        try:
            ok, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
                objectPoints=object_points,
                imagePoints=image_points,
                cameraMatrix=self.intrinsics,
                distCoeffs=None,
                iterationsCount=300,
                reprojectionError=self.config.pnp_reprojection_error,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
        except cv2.error:
            return None, np.zeros(len(valid), dtype=bool), 0.0
        if not ok or rotation_vector is None or translation_vector is None:
            return None, np.zeros(len(valid), dtype=bool), 0.0
        inlier_local = np.arange(len(valid_indices)) if inliers is None else inliers.reshape(-1)
        if len(inlier_local) < self.config.min_pnp_inliers:
            return None, np.zeros(len(valid), dtype=bool), len(inlier_local) / max(len(valid_indices), 1)
        try:
            if hasattr(cv2, "solvePnPRefineLM") and len(inlier_local) >= 6:
                rotation_vector, translation_vector = cv2.solvePnPRefineLM(
                    object_points[inlier_local],
                    image_points[inlier_local],
                    self.intrinsics,
                    None,
                    rotation_vector,
                    translation_vector,
                )
            rotation, _ = cv2.Rodrigues(rotation_vector)
        except cv2.error:
            return None, np.zeros(len(valid), dtype=bool), 0.0

        transform_anchor_to_current = np.eye(4, dtype=np.float64)
        transform_anchor_to_current[:3, :3] = rotation
        transform_anchor_to_current[:3, 3] = translation_vector.reshape(3)
        inlier_mask = np.zeros(len(valid), dtype=bool)
        inlier_mask[valid_indices[inlier_local]] = True
        ratio = float(len(inlier_local) / max(len(valid_indices), 1))
        return transform_anchor_to_current, inlier_mask, ratio

    def _limit_pose_step(
        self,
        previous_pose: np.ndarray,
        candidate_pose: np.ndarray,
        inlier_ratio: float,
    ) -> tuple[np.ndarray | None, bool]:
        if not np.isfinite(candidate_pose).all():
            return None, False
        try:
            relative = np.linalg.inv(previous_pose) @ candidate_pose
        except np.linalg.LinAlgError:
            return None, False
        if not np.isfinite(relative).all():
            return None, False

        translation = relative[:3, 3].copy()
        translation_norm = float(np.linalg.norm(translation))
        trace_value = float(np.clip((np.trace(relative[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
        rotation_deg = float(np.degrees(np.arccos(trace_value)))
        if self.config.step_rejection_enabled and (
            translation_norm > self.config.reject_translation_step
            or rotation_deg > self.config.reject_rotation_step_deg
        ):
            return None, False

        if inlier_ratio < self.config.low_confidence_ratio:
            max_translation = self.config.low_confidence_max_translation_step
            max_rotation_deg = self.config.low_confidence_max_rotation_step_deg
        else:
            max_translation = self.config.max_translation_step
            max_rotation_deg = self.config.max_rotation_step_deg

        limited = False
        if max_translation > 0.0 and translation_norm > max_translation:
            relative[:3, 3] = translation * (max_translation / max(translation_norm, 1e-12))
            limited = True
        if max_rotation_deg > 0.0 and rotation_deg > max_rotation_deg:
            rotation_vector, _ = cv2.Rodrigues(relative[:3, :3])
            rotation_vector *= max_rotation_deg / max(rotation_deg, 1e-12)
            relative[:3, :3], _ = cv2.Rodrigues(rotation_vector)
            limited = True
        return previous_pose @ relative, limited

    @staticmethod
    def _scaled_relative_pose(relative_pose: np.ndarray, scale: float) -> np.ndarray:
        scaled = np.eye(4, dtype=np.float64)
        scaled[:3, 3] = relative_pose[:3, 3] * scale
        rotation_vector, _ = cv2.Rodrigues(relative_pose[:3, :3])
        scaled[:3, :3], _ = cv2.Rodrigues(rotation_vector * scale)
        return scaled

    def _segment_map_points(
        self,
        points_3d: np.ndarray,
        valid: np.ndarray,
        anchor_camera_to_world: np.ndarray,
        anchor_bgr: np.ndarray,
    ) -> list[MapPoint]:
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            return []
        remaining = self.config.max_map_points
        if len(valid_indices) > remaining:
            valid_indices = self.random.choice(valid_indices, size=remaining, replace=False)
        rotation_world = anchor_camera_to_world[:3, :3]
        translation_world = anchor_camera_to_world[:3, 3]
        points = []
        for index in valid_indices:
            xyz_world = rotation_world @ points_3d[index] + translation_world
            points.append(MapPoint(xyz=xyz_world, rgb=(180, 180, 180)))
        return points

    def _process_segment(
        self,
        frames_bgr: list[np.ndarray],
        global_start: int,
        anchor_camera_to_world: np.ndarray,
        include_anchor: bool,
        depth_alignment_samples: DepthAlignmentSamples | None,
        propagation_state: PropagationState,
    ) -> tuple[list[PoseRecord], list[MapPoint], np.ndarray, DepthAlignmentSamples | None]:
        tracks, visibility = self._track_segment(frames_bgr)
        geometry_tracks = self._geometry_tracks(tracks)
        anchor_depth = self._estimate_anchor_depth(
            frames_bgr[0],
            alignment_samples=depth_alignment_samples,
        )
        anchor_points = geometry_tracks[0]
        anchor_points_3d, depth_valid = self._lift_anchor_points(
            anchor_points,
            anchor_depth,
            depth_sample_points=tracks[0],
        )
        anchor_valid = visibility[0] & depth_valid
        center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)

        records = []
        previous_pose = anchor_camera_to_world.copy()
        next_depth_alignment_samples = None
        last_reliable_local_index: int | None = None
        for local_index in range(len(frames_bgr)):
            frame_number = global_start + local_index
            if local_index == 0:
                pose = anchor_camera_to_world.copy()
                state = "anchor"
                inlier_ratio = 1.0
                valid_count = int(np.sum(anchor_valid))
                direction_dispersion = float("inf")
                radial_alignment = 0.0
            else:
                pose_before_update = previous_pose.copy()
                pair_valid = visibility[local_index - 1] & visibility[local_index]
                direction_dispersion, radial_alignment = self._flow_statistics(
                    tracks[local_index - 1],
                    tracks[local_index],
                    pair_valid,
                    center,
                )
                pnp_valid = anchor_valid & visibility[local_index]
                transform, inlier_mask, inlier_ratio = self._estimate_pose_pnp(
                    anchor_points_3d,
                    geometry_tracks[local_index],
                    pnp_valid,
                )
                valid_count = int(np.sum(pnp_valid))
                gate_supported = (
                    direction_dispersion <= self.config.max_direction_dispersion
                    or radial_alignment >= self.config.radial_alignment_threshold
                    or inlier_ratio >= self.config.strong_pnp_inlier_ratio
                )
                update_pose = (
                    transform is not None
                    and inlier_ratio >= self.config.min_pnp_inlier_ratio
                    and (gate_supported or not self.config.motion_gate_enabled)
                )
                pose = None
                failure_state = "hold"
                if update_pose:
                    candidate_pose = anchor_camera_to_world @ np.linalg.inv(transform)
                    pose, step_limited = self._limit_pose_step(
                        previous_pose,
                        candidate_pose,
                        inlier_ratio,
                    )
                    if pose is None:
                        failure_state = "step_reject"
                    else:
                        state = "limited" if step_limited else "estimated"
                else:
                    gate_blocked = (
                        transform is not None
                        and inlier_ratio >= self.config.min_pnp_inlier_ratio
                        and self.config.motion_gate_enabled
                        and not gate_supported
                    )
                    failure_state = "gated" if gate_blocked else "hold"

                if pose is None:
                    can_propagate = (
                        self.config.propagate_on_failure
                        and failure_state != "gated"
                        and propagation_state.relative_pose is not None
                        and propagation_state.age < self.config.propagation_max_frames
                    )
                    if can_propagate:
                        decay = float(
                            np.clip(
                                self.config.propagation_decay,
                                0.0,
                                1.0,
                            )
                        )
                        propagation_scale = decay ** propagation_state.age
                        propagated_relative = self._scaled_relative_pose(
                            propagation_state.relative_pose,
                            propagation_scale,
                        )
                        pose = pose_before_update @ propagated_relative
                        state = "propagated"
                        propagation_state.age += 1
                    else:
                        pose = pose_before_update.copy()
                        state = failure_state

                if state == "estimated" and inlier_ratio >= self.config.propagation_min_inlier_ratio:
                    propagation_state.relative_pose = (
                        np.linalg.inv(pose_before_update) @ pose
                    )
                    propagation_state.age = 0
                elif state in ("estimated", "limited"):
                    propagation_state.relative_pose = None
                    propagation_state.age = 0

                if state in ("estimated", "limited", "propagated"):
                    last_reliable_local_index = local_index

            if (
                local_index == len(frames_bgr) - 1
                and self.config.temporal_depth_alignment_enabled
                and last_reliable_local_index is not None
                and local_index - last_reliable_local_index
                <= self.config.depth_alignment_max_gap_frames
            ):
                alignment_valid = (
                    anchor_valid
                    & visibility[local_index]
                    & np.isfinite(tracks[local_index]).all(axis=1)
                )
                if int(np.sum(alignment_valid)) >= self.config.depth_alignment_min_points:
                    try:
                        anchor_to_current = np.linalg.inv(pose) @ anchor_camera_to_world
                        current_points = (
                            anchor_to_current[:3, :3] @ anchor_points_3d[alignment_valid].T
                        ).T + anchor_to_current[:3, 3]
                        positive_depth = current_points[:, 2] > 1e-6
                        if int(np.sum(positive_depth)) >= self.config.depth_alignment_min_points:
                            next_depth_alignment_samples = DepthAlignmentSamples(
                                image_points=tracks[local_index][alignment_valid][positive_depth],
                                camera_depths=current_points[positive_depth, 2],
                            )
                    except np.linalg.LinAlgError:
                        next_depth_alignment_samples = None
            previous_pose = pose
            if include_anchor or local_index > 0:
                records.append(
                    PoseRecord(
                        frame_id=frame_number + 1,
                        timestamp=frame_number / self.fps,
                        camera_to_world=pose.copy(),
                        extras=[
                            state,
                            f"{inlier_ratio:.6f}",
                            str(valid_count),
                            f"{direction_dispersion:.6f}",
                            f"{radial_alignment:.6f}",
                            f"{self.depth_multiplier:.6f}",
                            str(self.last_depth_alignment_count),
                        ],
                    )
                )
        map_points = self._segment_map_points(
            anchor_points_3d,
            anchor_valid,
            anchor_camera_to_world,
            frames_bgr[0],
        )
        return records, map_points, previous_pose, next_depth_alignment_samples

    def process(self) -> tuple[list[PoseRecord], list[MapPoint]]:
        self._decoded_frames = 0
        records: list[PoseRecord] = []
        map_points: list[MapPoint] = []
        overlap_frame = None
        global_start = 0
        anchor_camera_to_world = np.eye(4, dtype=np.float64)
        depth_alignment_samples = None
        propagation_state = PropagationState()

        while True:
            frames_bgr, reached_end = self._read_segment(overlap_frame)
            if not frames_bgr or (overlap_frame is not None and len(frames_bgr) == 1):
                break
            (
                segment_records,
                segment_points,
                anchor_camera_to_world,
                depth_alignment_samples,
            ) = self._process_segment(
                frames_bgr=frames_bgr,
                global_start=global_start,
                anchor_camera_to_world=anchor_camera_to_world,
                include_anchor=len(records) == 0,
                depth_alignment_samples=depth_alignment_samples,
                propagation_state=propagation_state,
            )
            records.extend(segment_records)
            remaining_capacity = self.config.max_map_points - len(map_points)
            if remaining_capacity > 0:
                map_points.extend(segment_points[:remaining_capacity])
            if reached_end:
                break
            overlap_frame = frames_bgr[-1]
            global_start += len(frames_bgr) - 1

        if not records:
            raise RuntimeError(f"No frames were processed from: {self.video_path}")
        return records, map_points
