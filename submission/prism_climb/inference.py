"""Small NumPy/OpenCV inference interfaces extracted from PRISM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .layers import disp_to_depth
from .networks import (
    DepthDecoder,
    DexiNed,
    ResnetEncoder,
    decompose_decoder,
)


_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_MANIFEST = _ROOT / "checkpoints.json"


def _read_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _validate_bgr(image: np.ndarray, name: str = "image") -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if image.dtype != np.uint8:
        raise TypeError(f"{name} must have dtype uint8, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (H, W, 3), got {image.shape}")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty")


def _load_state(path: Path, device: torch.device) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dictionary in {path}")
    return state


def _rgb_tensor(
    image_bgr: np.ndarray,
    size_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    height, width = size_hw
    resized = cv2.resize(
        image_bgr, (width, height), interpolation=cv2.INTER_LANCZOS4
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32)
    return torch.from_numpy(array).div_(255.0).unsqueeze(0).to(device)


def _aux_tensor(
    auxiliary: np.ndarray,
    size_hw: tuple[int, int],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if not isinstance(auxiliary, np.ndarray) or auxiliary.ndim != 2:
        raise ValueError(f"{name} must be a NumPy array with shape (H, W)")
    if not np.issubdtype(auxiliary.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype")
    height, width = size_hw
    resized = cv2.resize(
        auxiliary.astype(np.float32, copy=False),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return torch.from_numpy(np.ascontiguousarray(resized))[None, None].to(device)


class DepthGenerator:
    """Generate inverse depth/disparity from BGR using internal shading inference."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        shading_generator: "ShadingGenerator | None" = None,
        device: str | torch.device | None = None,
        manifest_path: str | Path = _DEFAULT_MANIFEST,
    ) -> None:
        manifest = _read_manifest(manifest_path)["depth"]
        self.checkpoint_dir = Path(checkpoint_dir or manifest["checkpoint_dir"])
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        encoder_state = _load_state(self.checkpoint_dir / "encoder.pth", self.device)
        self.feed_height = int(encoder_state.get("height", manifest["height"]))
        self.feed_width = int(encoder_state.get("width", manifest["width"]))

        self.encoder = ResnetEncoder(18, False)
        if int(manifest["input_channels"]) == 4:
            self.encoder.encoder.conv1 = torch.nn.Conv2d(
                4, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        filtered = {
            key: value
            for key, value in encoder_state.items()
            if key in self.encoder.state_dict()
        }
        self.encoder.load_state_dict(filtered, strict=True)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales=range(4))
        self.decoder.load_state_dict(
            _load_state(self.checkpoint_dir / "depth.pth", self.device), strict=True
        )
        self.encoder.to(self.device).eval()
        self.decoder.to(self.device).eval()
        self.shading_generator = shading_generator or ShadingGenerator(
            device=self.device, manifest_path=manifest_path
        )

    @torch.inference_mode()
    def predict_with_shading(
        self, image_bgr: np.ndarray, shading: np.ndarray
    ) -> np.ndarray:
        """Return float32 inverse depth/disparity with shape ``image_bgr.shape[:2]``."""
        _validate_bgr(image_bgr)
        original_hw = image_bgr.shape[:2]
        image = _rgb_tensor(
            image_bgr, (self.feed_height, self.feed_width), self.device
        )
        image = torch.cat(
            [
                image,
                _aux_tensor(
                    shading,
                    (self.feed_height, self.feed_width),
                    self.device,
                    "shading",
                ),
            ],
            dim=1,
        )
        raw_disp = self.decoder(self.encoder(image))[("disp", 0)]
        disparity, _ = disp_to_depth(raw_disp, 0.1, 100.0)
        restored = F.interpolate(
            disparity, size=original_hw, mode="bilinear", align_corners=False
        )
        return restored[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)

    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """Generate shading internally and return `(H,W)` inverse depth."""
        _validate_bgr(image_bgr)
        return self.predict_with_shading(
            image_bgr, self.shading_generator(image_bgr)
        )

    predict = __call__


class EdgeGenerator:
    """Generate the DexiNed average edge probability used by PRISM."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
        manifest_path: str | Path = _DEFAULT_MANIFEST,
    ) -> None:
        manifest = _read_manifest(manifest_path)["edge"]
        self.checkpoint_path = Path(checkpoint_path or manifest["checkpoint_path"])
        self.height = int(manifest["height"])
        self.width = int(manifest["width"])
        self.mean_bgr = np.asarray(manifest["mean_bgr"], dtype=np.float32)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = DexiNed()
        self.model.load_state_dict(
            _load_state(self.checkpoint_path, self.device), strict=True
        )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return float32 edge probability with the same `(H,W)` as the input."""
        _validate_bgr(image_bgr)
        original_hw = image_bgr.shape[:2]
        resized = cv2.resize(
            image_bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        resized -= self.mean_bgr
        tensor = torch.from_numpy(
            np.ascontiguousarray(resized.transpose(2, 0, 1))
        )[None].to(self.device)
        side_outputs = self.model(tensor)
        # Reproduce DexiNed utils/image.py exactly: independently min-max each
        # sigmoid side output, quantize to uint8, cubic-resize, then average.
        predictions = []
        for output in side_outputs:
            array = torch.sigmoid(output)[0, 0].cpu().numpy()
            array = (array - array.min()) * 255.0 / (
                (array.max() - array.min()) + 1e-12
            )
            array = array.astype(np.uint8)
            array = cv2.resize(
                array,
                (original_hw[1], original_hw[0]),
                interpolation=cv2.INTER_CUBIC,
            )
            predictions.append(array)
        average = np.mean(np.asarray(predictions, dtype=np.float32), axis=0)
        return (average / 255.0).astype(np.float32, copy=False)

    predict = __call__


class ShadingGenerator:
    """Generate IID-SfM illumination in PRISM's original 0–255 convention."""

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
        manifest_path: str | Path = _DEFAULT_MANIFEST,
    ) -> None:
        manifest = _read_manifest(manifest_path)["shading"]
        self.checkpoint_dir = Path(checkpoint_dir or manifest["checkpoint_dir"])
        self.height = int(manifest["height"])
        self.width = int(manifest["width"])
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.encoder = ResnetEncoder(18, False)
        self.decoder = decompose_decoder(self.encoder.num_ch_enc, scales=range(4))
        self.encoder.load_state_dict(
            _load_state(
                self.checkpoint_dir / "decompose_encoder.pth", self.device
            ),
            strict=True,
        )
        self.decoder.load_state_dict(
            _load_state(self.checkpoint_dir / "decompose.pth", self.device),
            strict=True,
        )
        self.encoder.to(self.device).eval()
        self.decoder.to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, image_bgr: np.ndarray) -> np.ndarray:
        """Return float32 shading `(H,W)` scaled to `[0,255]` like saved PNGs."""
        _validate_bgr(image_bgr)
        original_hw = image_bgr.shape[:2]
        image = _rgb_tensor(image_bgr, (self.height, self.width), self.device)
        _, light = self.decoder(self.encoder(image))
        restored = F.interpolate(
            light, size=original_hw, mode="bilinear", align_corners=False
        )
        return (
            restored[0, 0].mul(255.0).cpu().numpy().astype(np.float32, copy=False)
        )

    predict = __call__
