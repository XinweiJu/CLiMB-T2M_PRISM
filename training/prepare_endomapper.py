"""Extract CLiMB train/val videos and precompute frozen PRISM auxiliaries."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = Path(
    "/raid/scratch_not_backed_up/xinwei/CLiMB/EndoMapper-CLiMB"
)


def read_sequences(path: Path) -> list[str]:
    sequences = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not sequences:
        raise ValueError("No sequences found in {}".format(path))
    return sequences


def split_sequences(
    sequences: list[str], val_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    shuffled = list(sequences)
    random.Random(seed).shuffle(shuffled)
    count = max(1, round(len(shuffled) * val_fraction))
    val = sorted(shuffled[:count])
    train = sorted(shuffled[count:])
    return train, val


def generators(device: str):
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from prism_depth_climb.inference import EdgeGenerator, ShadingGenerator

    return EdgeGenerator(device=device), ShadingGenerator(device=device)


def write_gray(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.clip(array, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), image):
        raise OSError("Failed to write {}".format(path))


def process_sequence(
    sequence: str,
    raw_root: Path,
    prepared_root: Path,
    target_fps: float,
    size: int,
    edge_generator,
    shading_generator,
    max_sampled_frames: int | None,
) -> list[tuple[int, int, float]]:
    video = raw_root / sequence / "{}.mov".format(sequence)
    if not video.is_file():
        raise FileNotFoundError(video)

    frames_dir = prepared_root / "frames" / sequence
    edge_dir = prepared_root / "generated" / "edge" / sequence
    shading_dir = prepared_root / "generated" / "shading" / sequence
    for directory in [frames_dir, edge_dir, shading_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        capture.release()
        raise ValueError("Invalid FPS for {}".format(video))
    stride = max(1, round(source_fps / target_fps))
    records: list[tuple[int, int, float]] = []
    source_index = 0
    output_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if source_index % stride:
            source_index += 1
            continue

        frame_path = frames_dir / "{:06d}.jpg".format(output_index)
        edge_path = edge_dir / "{:06d}.png".format(output_index)
        shading_path = shading_dir / "{:06d}.png".format(output_index)
        resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)

        if not frame_path.exists():
            if not cv2.imwrite(
                str(frame_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]
            ):
                raise OSError("Failed to write {}".format(frame_path))
        if not edge_path.exists():
            write_gray(edge_path, edge_generator(resized) * 255.0)
        if not shading_path.exists():
            write_gray(shading_path, shading_generator(resized))

        records.append((output_index, source_index, source_index / source_fps))
        output_index += 1
        source_index += 1
        if max_sampled_frames and output_index >= max_sampled_frames:
            break

    capture.release()
    if len(records) < 3:
        raise ValueError("{} produced fewer than three sampled frames".format(sequence))
    return records


def write_split(
    path: Path, sequences: list[str], prepared_root: Path
) -> int:
    lines: list[str] = []
    for sequence in sequences:
        frames = sorted((prepared_root / "frames" / sequence).glob("*.jpg"))
        lines.extend(str(frame) for frame in frames[1:-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--sequence-list", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--size", type=int, default=288)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--max-sampled-frames", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root or args.dataset_root / "raw"
    sequence_list = args.sequence_list or args.dataset_root / "trainval.txt"
    output_root = args.output_root or args.dataset_root / "prism_depth_climb"
    prepared_root = output_root / "prepared_{}fps".format(
        "{:g}".format(args.target_fps)
    )
    sequences = read_sequences(sequence_list)
    if args.sequences:
        unknown = sorted(set(args.sequences) - set(sequences))
        if unknown:
            raise ValueError("Sequences not in trainval.txt: {}".format(unknown))
        sequences = args.sequences
    missing = [
        sequence
        for sequence in sequences
        if not (raw_root / sequence / "{}.mov".format(sequence)).is_file()
    ]
    if missing:
        print(
            "Skipping {} missing trainval videos under {}: {}".format(
                len(missing), raw_root, ", ".join(missing)
            ),
            flush=True,
        )
        sequences = [sequence for sequence in sequences if sequence not in missing]
    if len(sequences) < 2:
        raise ValueError("At least two available sequences are required")
    train_sequences, val_sequences = split_sequences(
        sequences, args.val_fraction, args.seed
    )
    edge_generator, shading_generator = generators(args.device)

    metadata_dir = prepared_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        records = process_sequence(
            sequence,
            raw_root,
            prepared_root,
            args.target_fps,
            args.size,
            edge_generator,
            shading_generator,
            args.max_sampled_frames,
        )
        with (metadata_dir / "{}.csv".format(sequence)).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_index", "source_frame", "time_seconds"])
            writer.writerows(records)
        print("{}: {} sampled frames".format(sequence, len(records)), flush=True)

    split_dir = args.split_dir or (
        Path(__file__).resolve().parent / "splits" / "endomapper_climb"
    )
    train_count = write_split(
        split_dir / "train_files.txt", train_sequences, prepared_root
    )
    val_count = write_split(
        split_dir / "val_files.txt", val_sequences, prepared_root
    )
    (metadata_dir / "train_sequences.txt").write_text(
        "\n".join(train_sequences) + "\n", encoding="utf-8"
    )
    (metadata_dir / "val_sequences.txt").write_text(
        "\n".join(val_sequences) + "\n", encoding="utf-8"
    )
    print(
        "Prepared {} train frames and {} validation frames under {}".format(
            train_count, val_count, prepared_root
        )
    )


if __name__ == "__main__":
    main()
