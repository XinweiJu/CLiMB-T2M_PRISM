#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path


SEQ_RE = re.compile(r"(?i)^seq[_-]?0*(\d+)(?:[_-]|$)")


def sequence_id(name: str) -> int:
    match = SEQ_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse sequence ID from {name!r}")
    return int(match.group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pairs_for_clip(clip: Path):
    images = sorted((clip / "imgs").glob("*.png"))
    masks = sorted(
        (clip / "segm_maps").glob("*_segm_map_*.png"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if len(images) != len(masks) or not images:
        raise ValueError(
            f"Unpaired clip {clip}: {len(images)} images, {len(masks)} masks"
        )
    for index, (image, mask) in enumerate(zip(images, masks)):
        yield {
            "sequence_id": sequence_id(clip.name),
            "clip": clip.name,
            "frame_index": index,
            "image": str(image.resolve()),
            "label": str(mask.resolve()),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segcol-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excluded-sequence", type=int, default=22)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"source": str(args.segcol_root.resolve()), "excluded_sequence": 22}
    for split in ("train", "valid"):
        clean, excluded = [], []
        for clip in sorted((args.segcol_root / split).iterdir()):
            if not clip.is_dir():
                continue
            target = excluded if sequence_id(clip.name) == args.excluded_sequence else clean
            target.extend(pairs_for_clip(clip))
        if any(row["sequence_id"] == args.excluded_sequence for row in clean):
            raise RuntimeError(f"Seq_{args.excluded_sequence:03d} leaked into {split}")
        path = args.output / f"segcol_{split}_minus_seq022.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in clean))
        excluded_path = args.output / f"segcol_{split}_excluded_seq022.jsonl"
        excluded_path.write_text("".join(json.dumps(row) + "\n" for row in excluded))
        summary[split] = {
            "clean_samples": len(clean),
            "excluded_samples": len(excluded),
            "clean_clips": len({row["clip"] for row in clean}),
            "excluded_clips": sorted({row["clip"] for row in excluded}),
            "manifest": path.name,
            "sha256": sha256(path),
        }
    (args.output / "segcol_minus_seq022_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
