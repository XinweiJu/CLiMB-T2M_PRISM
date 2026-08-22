#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Images(Dataset):
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.paths = sorted(self.root.glob("*/*.png")) + sorted(self.root.glob("*/*.jpg"))
        if not self.paths:
            raise RuntimeError(f"No Hyper-Kvasir images under {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        original = image.shape[:2]
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy((image.astype(np.float32) - MEAN_BGR).transpose(2, 0, 1))
        return tensor, str(path.relative_to(self.root)), torch.tensor(original)


def normalize_side(output, height, width):
    array = torch.sigmoid(output).float().cpu().numpy()[0]
    low, high = float(array.min()), float(array.max())
    array = (array - low) * 255.0 / max(high - low, 1e-8)
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexined-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    sys.path.insert(0, str(args.dexined_repo.resolve()))
    from model import DexiNed

    dataset = Images(args.image_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = DexiNed().cuda().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True), strict=True)
    written = 0
    with torch.inference_mode():
        for images, relative_paths, shapes in loader:
            outputs = model(images.cuda(non_blocking=True))
            for batch_index, relative in enumerate(relative_paths):
                height, width = map(int, shapes[batch_index].tolist())
                sides = [normalize_side(out[batch_index], height, width) for out in outputs]
                average = np.mean(sides, axis=0).astype(np.float32) / 255.0
                relative = Path(relative)
                target = args.output_root / relative.parent / "avg" / (relative.stem + ".png.npy")
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, average)
                written += 1
                if written % 500 == 0:
                    print(f"generated {written}/{len(dataset)}", flush=True)
    manifest = {
        "excluded_sequence": "Seq_022",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "source_root": str(args.image_root.resolve()),
        "output_root": str(args.output_root.resolve()),
        "images": written,
        "format": "seven independently min-max-normalized DexiNed sides, mean / 255 float32",
    }
    (args.output_root / "generation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
