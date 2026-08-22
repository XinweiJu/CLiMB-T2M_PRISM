"""Lightweight Track2Map-inspired monocular visual odometry bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from climb_io import MapPoint, PoseRecord


@dataclass
class VOConfig:
    max_features: int = 1600
    min_tracks: int = 40
    min_pose_inliers: int = 30
    min_inlier_ratio: float = 0.45
    max_forward_backward_error: float = 1.5
    ransac_threshold_px: float = 1.5
    assumed_focal_scale: float = 0.9
    translation_step: float = 0.01
    radial_alignment_threshold: float = 0.55
    max_direction_dispersion: float = 1.35
    max_map_points: int = 12000
    map_points_per_frame: int = 80
    max_propagation_frames: int = 10
    max_frames: int = 0


class MonocularTrackVO:
    """Adjacent-frame VO with multi-cue gating and constant-motion fallback."""

    def __init__(self, video_path: Path, seed: int, config: VOConfig):
        self.video_path = video_path
        self.config = config
        self.random = np.random.default_rng(seed)
        cv2.setRNGSeed(seed)

        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            self.fps = 30.0
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.width <= 0 or self.height <= 0:
            raise RuntimeError(f"Invalid video dimensions: {self.width}x{self.height}")

        focal = config.assumed_focal_scale * max(self.width, self.height)
        self.intrinsics = np.array(
            [
                [focal, 0.0, self.width / 2.0],
                [0.0, focal, self.height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def close(self) -> None:
        self.capture.release()

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        features = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            useHarrisDetector=False,
        )
        if features is None:
            return np.empty((0, 2), dtype=np.float32)
        return features.reshape(-1, 2).astype(np.float32)

    def _track_features(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(previous_points) == 0:
            return previous_points, previous_points

        current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points.reshape(-1, 1, 2),
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if current_points is None or forward_status is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            previous_gray,
            current_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if backward_points is None or backward_status is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

        previous_flat = previous_points.reshape(-1, 2)
        current_flat = current_points.reshape(-1, 2)
        backward_flat = backward_points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(previous_flat - backward_flat, axis=1)
        valid = (
            forward_status.reshape(-1).astype(bool)
            & backward_status.reshape(-1).astype(bool)
            & np.isfinite(current_flat).all(axis=1)
            & (forward_backward_error <= self.config.max_forward_backward_error)
        )
        return previous_flat[valid], current_flat[valid]

    @staticmethod
    def _flow_statistics(previous_points: np.ndarray, current_points: np.ndarray, center: np.ndarray) -> tuple[float, float]:
        if len(previous_points) == 0:
            return float("inf"), 0.0

        flow = current_points - previous_points
        magnitude = np.linalg.norm(flow, axis=1)
        valid = magnitude > 1e-4
        if not np.any(valid):
            return float("inf"), 0.0

        flow = flow[valid]
        magnitude = magnitude[valid]
        directions = np.arctan2(flow[:, 1], flow[:, 0])
        resultant = np.hypot(np.mean(np.cos(directions)), np.mean(np.sin(directions)))
        resultant = float(np.clip(resultant, 1e-8, 1.0))
        direction_dispersion = float(np.sqrt(-2.0 * np.log(resultant)))

        radial = previous_points[valid] - center.reshape(1, 2)
        radial_norm = np.linalg.norm(radial, axis=1)
        radial_valid = radial_norm > 1.0
        if not np.any(radial_valid):
            return direction_dispersion, 0.0
        cosine = np.sum(flow[radial_valid] * radial[radial_valid], axis=1) / (
            magnitude[radial_valid] * radial_norm[radial_valid] + 1e-8
        )
        radial_alignment = float(np.median(np.abs(cosine)))
        return direction_dispersion, radial_alignment

    def _estimate_relative_pose(
        self,
        previous_points: np.ndarray,
        current_points: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, float]:
        if len(previous_points) < self.config.min_tracks:
            return None, None, np.zeros(len(previous_points), dtype=bool), 0.0

        essential, ransac_mask = cv2.findEssentialMat(
            previous_points,
            current_points,
            self.intrinsics,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=self.config.ransac_threshold_px,
        )
        if essential is None or ransac_mask is None:
            return None, None, np.zeros(len(previous_points), dtype=bool), 0.0

        ransac_inliers = ransac_mask.reshape(-1).astype(bool).copy()
        ransac_inlier_ratio = float(np.mean(ransac_inliers)) if len(ransac_inliers) else 0.0
        if int(np.sum(ransac_inliers)) < self.config.min_pose_inliers:
            return None, None, ransac_inliers, ransac_inlier_ratio

        essential_candidates = (
            [essential]
            if essential.shape == (3, 3)
            else [essential[row : row + 3] for row in range(0, essential.shape[0], 3)]
        )
        best_pose = None
        for candidate in essential_candidates:
            if candidate.shape != (3, 3):
                continue
            try:
                pose_count, rotation, translation, pose_mask = cv2.recoverPose(
                    candidate,
                    previous_points,
                    current_points,
                    self.intrinsics,
                    mask=ransac_mask.copy(),
                )
            except cv2.error:
                continue
            if best_pose is None or pose_count > best_pose[0]:
                best_pose = (pose_count, rotation, translation.reshape(3), pose_mask)
        if best_pose is None:
            return None, None, ransac_inliers, ransac_inlier_ratio

        _, rotation, translation, pose_mask = best_pose
        pose_inliers = pose_mask.reshape(-1).astype(bool)
        triangulation_inliers = (
            pose_inliers
            if int(np.sum(pose_inliers)) >= self.config.min_pose_inliers
            else ransac_inliers
        )
        return rotation, translation, triangulation_inliers, ransac_inlier_ratio

    def _triangulate_map_points(
        self,
        previous_points: np.ndarray,
        current_points: np.ndarray,
        inliers: np.ndarray,
        rotation: np.ndarray,
        translation: np.ndarray,
        previous_camera_to_world: np.ndarray,
        previous_bgr: np.ndarray,
    ) -> list[MapPoint]:
        selected_previous = previous_points[inliers]
        selected_current = current_points[inliers]
        if len(selected_previous) < 4:
            return []

        projection_previous = self.intrinsics @ np.hstack([np.eye(3), np.zeros((3, 1))])
        projection_current = self.intrinsics @ np.hstack([rotation, translation.reshape(3, 1)])
        homogeneous = cv2.triangulatePoints(
            projection_previous,
            projection_current,
            selected_previous.T,
            selected_current.T,
        )
        valid_w = np.abs(homogeneous[3]) > 1e-8
        points_previous = np.zeros((homogeneous.shape[1], 3), dtype=np.float64)
        points_previous[valid_w] = (homogeneous[:3, valid_w] / homogeneous[3, valid_w]).T
        points_current = (rotation @ points_previous.T + translation.reshape(3, 1)).T
        valid = (
            valid_w
            & np.isfinite(points_previous).all(axis=1)
            & (points_previous[:, 2] > 0.0)
            & (points_current[:, 2] > 0.0)
            & (np.linalg.norm(points_previous, axis=1) < 50.0)
        )
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) == 0:
            return []
        if len(valid_indices) > self.config.map_points_per_frame:
            valid_indices = self.random.choice(
                valid_indices,
                size=self.config.map_points_per_frame,
                replace=False,
            )

        rotation_world = previous_camera_to_world[:3, :3]
        translation_world = previous_camera_to_world[:3, 3]
        map_points = []
        for point_index in valid_indices:
            xyz_world = rotation_world @ points_previous[point_index] + translation_world
            pixel_x, pixel_y = np.rint(selected_previous[point_index]).astype(int)
            pixel_x = int(np.clip(pixel_x, 0, previous_bgr.shape[1] - 1))
            pixel_y = int(np.clip(pixel_y, 0, previous_bgr.shape[0] - 1))
            blue, green, red = (int(value) for value in previous_bgr[pixel_y, pixel_x])
            map_points.append(MapPoint(xyz=xyz_world, rgb=(red, green, blue)))
        return map_points

    def process(self) -> tuple[list[PoseRecord], list[MapPoint]]:
        ok, previous_bgr = self.capture.read()
        if not ok or previous_bgr is None:
            raise RuntimeError(f"Could not read first frame: {self.video_path}")

        previous_gray = cv2.cvtColor(previous_bgr, cv2.COLOR_BGR2GRAY)
        current_camera_to_world = np.eye(4, dtype=np.float64)
        last_camera_increment = np.eye(4, dtype=np.float64)
        records = [
            PoseRecord(
                frame_id=1,
                timestamp=0.0,
                camera_to_world=current_camera_to_world.copy(),
                extras=["init", "0.000000", "0", "inf", "0.000000"],
            )
        ]
        map_points: list[MapPoint] = []
        frame_id = 1
        consecutive_propagations = 0
        center = np.array([self.width / 2.0, self.height / 2.0], dtype=np.float32)

        while self.config.max_frames <= 0 or frame_id < self.config.max_frames:
            ok, current_bgr = self.capture.read()
            if not ok or current_bgr is None:
                break
            frame_id += 1
            current_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
            detected_points = self._detect_features(previous_gray)
            previous_points, current_points = self._track_features(
                previous_gray,
                current_gray,
                detected_points,
            )
            direction_dispersion, radial_alignment = self._flow_statistics(
                previous_points,
                current_points,
                center,
            )
            rotation, translation, inliers, inlier_ratio = self._estimate_relative_pose(
                previous_points,
                current_points,
            )

            direction_supported = direction_dispersion <= self.config.max_direction_dispersion
            radial_supported = radial_alignment >= self.config.radial_alignment_threshold
            strong_epipolar_support = inlier_ratio >= max(0.70, self.config.min_inlier_ratio)
            update_pose = (
                rotation is not None
                and translation is not None
                and inlier_ratio >= self.config.min_inlier_ratio
                and (direction_supported or radial_supported or strong_epipolar_support)
            )

            previous_camera_to_world = current_camera_to_world.copy()
            if update_pose:
                translation_norm = float(np.linalg.norm(translation))
                if translation_norm > 1e-8:
                    translation = translation / translation_norm * self.config.translation_step
                camera_previous_to_current = np.eye(4, dtype=np.float64)
                camera_previous_to_current[:3, :3] = rotation
                camera_previous_to_current[:3, 3] = translation
                last_camera_increment = np.linalg.inv(camera_previous_to_current)
                current_camera_to_world = previous_camera_to_world @ last_camera_increment
                state = "estimated"
                consecutive_propagations = 0
                new_points = self._triangulate_map_points(
                    previous_points,
                    current_points,
                    inliers,
                    rotation,
                    translation,
                    previous_camera_to_world,
                    previous_bgr,
                )
                remaining_capacity = self.config.max_map_points - len(map_points)
                if remaining_capacity > 0:
                    map_points.extend(new_points[:remaining_capacity])
            else:
                consecutive_propagations += 1
                if consecutive_propagations <= self.config.max_propagation_frames:
                    current_camera_to_world = previous_camera_to_world @ last_camera_increment
                    state = "propagated"
                else:
                    current_camera_to_world = previous_camera_to_world
                    state = "held"

            records.append(
                PoseRecord(
                    frame_id=frame_id,
                    timestamp=(frame_id - 1) / self.fps,
                    camera_to_world=current_camera_to_world.copy(),
                    extras=[
                        state,
                        f"{inlier_ratio:.6f}",
                        str(len(previous_points)),
                        f"{direction_dispersion:.6f}",
                        f"{radial_alignment:.6f}",
                    ],
                )
            )
            previous_bgr = current_bgr
            previous_gray = current_gray

        return records, map_points
