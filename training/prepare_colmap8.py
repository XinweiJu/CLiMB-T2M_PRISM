"""Prepare the eight public COLMAP-validation clips for adaptation experiments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from prepare_endomapper import (
    generators,
    split_sequences,
    write_gray,
    write_split,
)


DEFAULT_INPUT = Path(
    "/home/xju/Workspace/PRISM-CLiMB/Track2Map-CLiMB/"
    "dataset_EndoMapper-CLiMB/EndoMapper-CLiMB/colmap-validation-v1/input"
)
DEFAULT_OUTPUT = Path(
    "/raid/scratch_not_backed_up/xinwei/CLiMB/EndoMapper-CLiMB/"
    "prism_depth_climb/colmap8_adaptation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=288)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--val-count", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = sorted(args.input_root.glob("*.mp4"))
    if len(videos) != 8:
        raise ValueError("Expected 8 mp4 clips, found {}".format(len(videos)))
    prepared = args.output_root / "prepared_fullfps"
    edge_generator, shading_generator = generators(args.device)
    metadata = prepared / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    for video in videos:
        sequence = video.stem
        frames_dir = prepared / "frames" / sequence
        edge_dir = prepared / "generated" / "edge" / sequence
        shading_dir = prepared / "generated" / "shading" / sequence
        for directory in [frames_dir, edge_dir, shading_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(video))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        records = []
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            resized = cv2.resize(
                frame, (args.size, args.size), interpolation=cv2.INTER_AREA
            )
            frame_path = frames_dir / "{:06d}.jpg".format(index)
            edge_path = edge_dir / "{:06d}.png".format(index)
            shading_path = shading_dir / "{:06d}.png".format(index)
            if not frame_path.exists():
                cv2.imwrite(
                    str(frame_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]
                )
            if not edge_path.exists():
                write_gray(edge_path, edge_generator(resized) * 255.0)
            if not shading_path.exists():
                write_gray(shading_path, shading_generator(resized))
            records.append((index, index, index / fps))
            index += 1
        capture.release()
        if len(records) < 3:
            raise ValueError("{} has fewer than 3 frames".format(sequence))
        with (metadata / "{}.csv".format(sequence)).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(["sample_index", "source_frame", "time_seconds"])
            writer.writerows(records)
        print("{}: {} frames".format(sequence, len(records)), flush=True)

    sequences = [video.stem for video in videos]
    train, val = split_sequences(
        sequences, args.val_count / len(sequences), args.seed
    )
    split_dir = Path(__file__).resolve().parent / "splits" / "endomapper_colmap8"
    train_count = write_split(split_dir / "train_files.txt", train, prepared)
    val_count = write_split(split_dir / "val_files.txt", val, prepared)
    (metadata / "train_sequences.txt").write_text(
        "\n".join(train) + "\n", encoding="utf-8"
    )
    (metadata / "val_sequences.txt").write_text(
        "\n".join(val) + "\n", encoding="utf-8"
    )
    print(
        "Prepared train={} frames ({} clips), val={} frames ({} clips)".format(
            train_count, len(train), val_count, len(val)
        )
    )


if __name__ == "__main__":
    main()
