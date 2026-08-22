"""Dependency-free fixed-lag smoothing for CLiMB pose trajectories."""

from __future__ import annotations

import numpy as np

from climb_io import PoseRecord, rotation_matrix_to_quaternion


def effective_window_length(num_poses: int, requested: int, polyorder: int) -> int:
    if requested <= polyorder or requested % 2 == 0:
        raise ValueError("window length must be odd and greater than polyorder")
    available = num_poses if num_poses % 2 == 1 else num_poses - 1
    if available <= polyorder:
        return 0
    return min(requested, available)


def savgol_smooth(
    values: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    effective_window = effective_window_length(
        len(samples),
        window_length,
        polyorder,
    )
    if effective_window == 0:
        return samples.copy()

    half_window = effective_window // 2
    sample_positions = np.arange(effective_window, dtype=np.float64)
    weights_by_position = []
    for position in range(effective_window):
        offsets = sample_positions - float(position)
        design = np.vander(offsets, N=polyorder + 1, increasing=True)
        weights_by_position.append(np.linalg.pinv(design)[0])
    output = np.empty_like(samples)
    for index in range(len(samples)):
        start = min(max(index - half_window, 0), len(samples) - effective_window)
        position = index - start
        weights = weights_by_position[position]
        output[index] = weights @ samples[start : start + effective_window]
    return output


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_w = left[..., 0]
    left_xyz = left[..., 1:]
    right_w = right[..., 0]
    right_xyz = right[..., 1:]
    result = np.empty(np.broadcast_shapes(left.shape, right.shape), dtype=np.float64)
    result[..., 0] = left_w * right_w - np.sum(left_xyz * right_xyz, axis=-1)
    result[..., 1:] = (
        left_w[..., None] * right_xyz
        + right_w[..., None] * left_xyz
        + np.cross(left_xyz, right_xyz)
    )
    return result


def smooth_quaternions(
    quaternions: np.ndarray,
    window_length: int,
    polyorder: int,
) -> np.ndarray:
    continuous = np.asarray(quaternions, dtype=np.float64).copy()
    for index in range(1, len(continuous)):
        if float(np.dot(continuous[index - 1], continuous[index])) < 0.0:
            continuous[index] *= -1.0
    smoothed = savgol_smooth(continuous, window_length, polyorder)
    smoothed /= np.linalg.norm(smoothed, axis=1, keepdims=True).clip(min=1e-12)
    conjugate_first = smoothed[0] * np.array([1.0, -1.0, -1.0, -1.0])
    correction = quaternion_multiply(continuous[0], conjugate_first)
    smoothed = quaternion_multiply(correction, smoothed)
    smoothed /= np.linalg.norm(smoothed, axis=1, keepdims=True).clip(min=1e-12)
    return smoothed


def quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def smooth_pose_records(
    records: list[PoseRecord],
    translation_window_length: int = 0,
    translation_polyorder: int = 2,
    rotation_window_length: int = 0,
    rotation_polyorder: int = 1,
) -> None:
    if not records:
        return
    if translation_window_length > 0:
        translations = np.asarray(
            [record.camera_to_world[:3, 3] for record in records],
            dtype=np.float64,
        )
        first_translation = translations[0].copy()
        translations = savgol_smooth(
            translations,
            translation_window_length,
            translation_polyorder,
        )
        translations += first_translation - translations[0]
        for record, translation in zip(records, translations):
            record.camera_to_world[:3, 3] = translation

    if rotation_window_length > 0:
        quaternions = np.asarray(
            [rotation_matrix_to_quaternion(record.camera_to_world[:3, :3]) for record in records],
            dtype=np.float64,
        )
        quaternions = smooth_quaternions(
            quaternions,
            rotation_window_length,
            rotation_polyorder,
        )
        for record, quaternion in zip(records, quaternions):
            record.camera_to_world[:3, :3] = quaternion_to_rotation_matrix(quaternion)
