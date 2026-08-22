"""CLiMB trajectory, map, and runtime serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PoseRecord:
    frame_id: int
    timestamp: float
    camera_to_world: np.ndarray
    extras: list[str] = field(default_factory=list)


@dataclass
class MapPoint:
    xyz: np.ndarray
    rgb: tuple[int, int, int]
    reprojection_error: float = 0.0


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized (w, x, y, z) quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        return 1.0, 0.0, 0.0, 0.0
    quaternion /= norm
    return tuple(float(value) for value in quaternion)


def write_trajectory(path: Path, records: list[PoseRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write(
            "# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz, "
            "state, inlier_ratio, num_tracks, direction_dispersion, radial_alignment, "
            "depth_multiplier, depth_alignment_count\n"
        )
        for record in records:
            transform = np.asarray(record.camera_to_world, dtype=np.float64)
            translation = transform[:3, 3]
            quaternion = rotation_matrix_to_quaternion(transform[:3, :3])
            fields = [
                f"{record.timestamp:.9f}",
                f"{record.frame_id:06d}.png",
                *(f"{value:.9f}" for value in translation),
                *(f"{value:.9f}" for value in quaternion),
                *record.extras,
            ]
            output_file.write(",".join(fields) + "\n")


def write_points3d(path: Path, points: list[MapPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not points:
        points = [
            MapPoint(np.array([0.00, 0.00, 0.05]), (255, 0, 0)),
            MapPoint(np.array([0.01, 0.00, 0.05]), (0, 255, 0)),
            MapPoint(np.array([0.00, 0.01, 0.05]), (0, 0, 255)),
            MapPoint(np.array([0.01, 0.01, 0.05]), (255, 255, 255)),
        ]

    with path.open("w", encoding="utf-8") as output_file:
        output_file.write("# POINT3D_ID X Y Z R G B ERROR\n")
        for point_id, point in enumerate(points, start=1):
            x_coord, y_coord, z_coord = (float(value) for value in point.xyz)
            red, green, blue = point.rgb
            output_file.write(
                f"{point_id} {x_coord:.9f} {y_coord:.9f} {z_coord:.9f} "
                f"{red:d} {green:d} {blue:d} {point.reprojection_error:.6f}\n"
            )


def write_runtime(path: Path, init_seconds: float, processing_seconds: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        output_file.write(f"init_seconds={init_seconds:.6f}\n")
        output_file.write(f"processing_seconds={processing_seconds:.6f}\n")
