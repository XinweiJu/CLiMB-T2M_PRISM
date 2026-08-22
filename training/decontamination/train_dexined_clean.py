#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sys
import time
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


class SegColEdges(Dataset):
    def __init__(self, manifest: Path, size: int, training: bool):
        self.rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        if not self.rows or any(int(row["sequence_id"]) == 22 for row in self.rows):
            raise RuntimeError(f"Unsafe or empty manifest: {manifest}")
        self.size = size
        self.training = training

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        label = cv2.imread(row["label"], cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            raise FileNotFoundError(row)
        image = image.astype(np.float32) - MEAN_BGR
        label = label.astype(np.float32) / 255.0
        height, width = label.shape
        if self.training and height > self.size and width > self.size:
            top = random.randint(0, height - self.size)
            left = random.randint(0, width - self.size)
            image = image[top:top + self.size, left:left + self.size]
            label = label[top:top + self.size, left:left + self.size]
        else:
            image = cv2.resize(image, (self.size, self.size))
            label = cv2.resize(label, (self.size, self.size))
        # Preserve the transform used by the archived DexiNed trainer.
        label[label > 0.1] += 0.2
        label = np.clip(label, 0.0, 1.0)
        image = image.transpose(2, 0, 1)
        label = label[None]
        return torch.from_numpy(image), torch.from_numpy(label)


SIDE_WEIGHTS = [0.7, 0.7, 1.1, 1.1, 0.3, 0.3, 1.3]


def bdcn_loss2(logits, targets, side_weight):
    """Exact loss formulation used by the archived DexiNed trainer."""
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float())
    num_negative = torch.sum((mask <= 0.0).float())
    denominator = (num_positive + num_negative).clamp_min(1.0)
    mask[mask > 0.0] = num_negative / denominator
    mask[mask <= 0.0] = 1.1 * num_positive / denominator
    probabilities = torch.sigmoid(logits)
    cost = torch.nn.functional.binary_cross_entropy(
        probabilities, targets.float(), weight=mask, reduction="none"
    )
    return side_weight * torch.sum(cost.float().mean((1, 2, 3)))


def side_loss(outputs, labels):
    return sum(
        bdcn_loss2(output, labels, weight)
        for output, weight in zip(outputs, SIDE_WEIGHTS)
    )


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            losses.append(float(side_loss(outputs, labels)))
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dexined-repo", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--size", type=int, default=352)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1021)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    sys.path.insert(0, str(args.dexined_repo.resolve()))
    from model import DexiNed

    device = torch.device("cuda")
    model = DexiNed().to(device)
    state = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    train_set = SegColEdges(args.train_manifest, args.size, training=True)
    val_set = SegColEdges(args.val_manifest, args.size, training=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, generator=generator,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-8)
    args.output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(args.epochs):
        # Match main.py: seed changes at epochs 0, 7 and 14.
        if epoch % 7 == 0:
            epoch_seed = args.seed + 1000 * (epoch // 7 + 1)
            random.seed(epoch_seed)
            np.random.seed(epoch_seed)
            torch.manual_seed(epoch_seed)
            torch.cuda.manual_seed_all(epoch_seed)
        if epoch in (10, 15):
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        model.train()
        losses, started = [], time.time()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            loss = side_loss(outputs, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_loss = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch + 1, "train_loss": float(np.mean(losses)),
            "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save(model.state_dict(), args.output / f"{epoch}_model.pth")
        torch.save(model.state_dict(), args.output / "last_model.pth")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), args.output / "best_model.pth")
        (args.output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    provenance = {
        "excluded_sequence": "Seq_022",
        "train_manifest_sha256": sha256(args.train_manifest),
        "val_manifest_sha256": sha256(args.val_manifest),
        "initial_checkpoint": str(args.initial_checkpoint),
        "initial_checkpoint_sha256": sha256(args.initial_checkpoint),
        "best_checkpoint_sha256": sha256(args.output / "best_model.pth"),
        "best_val_loss": best,
        "epochs": args.epochs,
        "final_checkpoint": str(args.output / f"{args.epochs - 1}_model.pth"),
        "final_checkpoint_sha256": sha256(args.output / f"{args.epochs - 1}_model.pth"),
        "side_weights": SIDE_WEIGHTS,
        "crop_size": args.size,
        "seed": args.seed,
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
