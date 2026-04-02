from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

from padim_backend import IMAGENET_MEAN, IMAGENET_STD, ResNetFeatureExtractor, build_model


DEFAULT_IMAGE_SIZE = 256
DEFAULT_EPS = 1e-6


@dataclass(frozen=True)
class SyntheticSegArtifact:
    image_size: int
    threshold: float
    min_area: int
    open_kernel: int
    close_kernel: int
    final_dilate_kernel: int
    freeze_encoder: bool
    encoder_state_dict: dict[str, torch.Tensor]
    decoder_state_dict: dict[str, torch.Tensor]


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class SyntheticSegNet(nn.Module):
    def __init__(self, encoder: ResNetFeatureExtractor | None = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else build_model(weights_path=None)
        self.proj3 = nn.Conv2d(256, 128, kernel_size=1)
        self.proj2 = nn.Conv2d(128, 96, kernel_size=1)
        self.proj1 = nn.Conv2d(64, 64, kernel_size=1)
        self.dec2 = DecoderBlock(128 + 96, 128)
        self.dec1 = DecoderBlock(128 + 64, 96)
        self.dec0 = DecoderBlock(96 + 32, 64)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            DecoderBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        norm_inputs = (inputs - IMAGENET_MEAN.to(inputs.device, inputs.dtype)) / IMAGENET_STD.to(inputs.device, inputs.dtype)
        feat1, feat2, feat3 = self.encoder(norm_inputs)

        x3 = self.proj3(feat3)
        x3 = F.interpolate(x3, size=feat2.shape[-2:], mode="bilinear", align_corners=False)

        x2 = self.proj2(feat2)
        x = self.dec2(torch.cat([x3, x2], dim=1))
        x = F.interpolate(x, size=feat1.shape[-2:], mode="bilinear", align_corners=False)

        x1 = self.proj1(feat1)
        x = self.dec1(torch.cat([x, x1], dim=1))
        x = F.interpolate(x, size=inputs.shape[-2:], mode="bilinear", align_corners=False)

        stem = self.stem(inputs)
        x = self.dec0(torch.cat([x, stem], dim=1))
        return self.head(x)


def image_to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize((image_size, image_size), Image.BILINEAR),
        dtype=np.float32,
    )
    return torch.from_numpy(resized.transpose(2, 0, 1)).float() / 255.0


def estimate_cable_region(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    small_side = 160
    resized = np.asarray(
        Image.fromarray(image, mode="RGB").resize((small_side, small_side), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    luma = 0.299 * resized[..., 0] + 0.587 * resized[..., 1] + 0.114 * resized[..., 2]
    threshold = float(np.quantile(luma, 0.82))
    dark_mask = luma < threshold

    yy, xx = np.indices((small_side, small_side), dtype=np.float32)
    cy = (small_side - 1) * 0.5
    cx = (small_side - 1) * 0.5
    ry = small_side * 0.42
    rx = small_side * 0.42
    central_ellipse = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0

    mask_u8 = (np.logical_or(dark_mask, central_ellipse).astype(np.uint8) * 255)
    mask_u8 = np.asarray(Image.fromarray(mask_u8, mode="L").filter(ImageFilter.MaxFilter(11)), dtype=np.uint8)
    mask_u8 = np.asarray(
        Image.fromarray(mask_u8, mode="L").resize((width, height), Image.NEAREST),
        dtype=np.uint8,
    )
    return mask_u8 > 0


def _random_point_in_mask(rng: np.random.Generator, mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        height, width = mask.shape
        return int(rng.integers(0, height)), int(rng.integers(0, width))
    index = int(rng.integers(0, ys.size))
    return int(xs[index]), int(ys[index])


def generate_synthetic_mask(rng: np.random.Generator, candidate_mask: np.ndarray) -> np.ndarray:
    height, width = candidate_mask.shape
    canvas = Image.new("L", (width, height), color=0)
    draw = ImageDraw.Draw(canvas)
    num_regions = int(rng.integers(1, 4))

    for _ in range(num_regions):
        region_type = rng.choice(["ellipse", "wedge", "strip", "line"])
        cx, cy = _random_point_in_mask(rng, candidate_mask)
        if region_type == "ellipse":
            rx = int(rng.integers(max(8, width // 24), max(16, width // 7)))
            ry = int(rng.integers(max(8, height // 24), max(16, height // 7)))
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
        elif region_type == "wedge":
            radius = int(rng.integers(max(18, min(width, height) // 10), max(32, min(width, height) // 4)))
            start = float(rng.integers(0, 360))
            sweep = float(rng.integers(20, 100))
            draw.pieslice((cx - radius, cy - radius, cx + radius, cy + radius), start=start, end=start + sweep, fill=255)
        elif region_type == "strip":
            length = int(rng.integers(max(24, width // 8), max(48, width // 3)))
            thickness = int(rng.integers(6, 18))
            angle = float(rng.uniform(0.0, np.pi))
            dx = int(np.cos(angle) * length * 0.5)
            dy = int(np.sin(angle) * length * 0.5)
            draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=255, width=thickness)
        else:
            points = []
            for _ in range(4):
                px = int(np.clip(cx + rng.integers(-width // 8, width // 8 + 1), 0, width - 1))
                py = int(np.clip(cy + rng.integers(-height // 8, height // 8 + 1), 0, height - 1))
                points.append((px, py))
            draw.line(points, fill=255, width=int(rng.integers(6, 16)))

    mask = np.asarray(canvas, dtype=np.uint8) > 0
    mask &= candidate_mask
    if not np.any(mask):
        mask = candidate_mask.copy()

    mask_u8 = mask.astype(np.uint8) * 255
    if rng.random() < 0.7:
        mask_u8 = np.asarray(Image.fromarray(mask_u8, mode="L").filter(ImageFilter.MaxFilter(5)), dtype=np.uint8)
    if rng.random() < 0.5:
        mask_u8 = np.asarray(Image.fromarray(mask_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.uint8)
    return mask_u8 > 96


def apply_synthetic_anomaly(
    image: np.ndarray,
    source_image: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    image_f = image.astype(np.float32)
    source_f = source_image.astype(np.float32)
    output = image_f.copy()

    anomaly_type = rng.choice(["cutpaste", "erase", "color", "shift", "blur"])
    mask_f = mask[..., None].astype(np.float32)
    alpha = float(rng.uniform(0.75, 1.0))

    if anomaly_type == "cutpaste":
        source_patch = source_f
        blend = alpha * source_patch + (1.0 - alpha) * image_f
        output = output * (1.0 - mask_f) + blend * mask_f
    elif anomaly_type == "erase":
        fill = image_f.mean(axis=(0, 1), keepdims=True) * float(rng.uniform(0.2, 0.9))
        output = output * (1.0 - mask_f) + fill * mask_f
    elif anomaly_type == "color":
        shift = rng.uniform(-85.0, 85.0, size=(1, 1, 3)).astype(np.float32)
        scale = rng.uniform(0.5, 1.5, size=(1, 1, 3)).astype(np.float32)
        colored = np.clip(image_f * scale + shift, 0.0, 255.0)
        output = output * (1.0 - mask_f) + colored * mask_f
    elif anomaly_type == "shift":
        dx = int(rng.integers(-18, 19))
        dy = int(rng.integers(-18, 19))
        shifted = np.roll(image_f, shift=(dy, dx), axis=(0, 1))
        output = output * (1.0 - mask_f) + shifted * mask_f
    else:
        blurred = np.asarray(Image.fromarray(image.astype(np.uint8), mode="RGB").filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(2.0, 5.0)))), dtype=np.float32)
        sharpened = np.clip(image_f * 1.4 - blurred * 0.4, 0.0, 255.0)
        candidate = sharpened if rng.random() < 0.5 else blurred
        output = output * (1.0 - mask_f) + candidate * mask_f

    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    numerator = 2.0 * torch.sum(probs * targets, dim=(1, 2, 3)) + 1.0
    denominator = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(targets, dim=(1, 2, 3)) + 1.0
    dice = 1.0 - numerator / denominator
    return bce + dice.mean()


def save_artifact(path: Path, artifact: SyntheticSegArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "backend": "synthetic_seg_resnet18",
        "image_size": int(artifact.image_size),
        "threshold": float(artifact.threshold),
        "min_area": int(artifact.min_area),
        "open_kernel": int(artifact.open_kernel),
        "close_kernel": int(artifact.close_kernel),
        "final_dilate_kernel": int(artifact.final_dilate_kernel),
        "freeze_encoder": bool(artifact.freeze_encoder),
        "encoder_state_dict": {key: value.detach().cpu() for key, value in artifact.encoder_state_dict.items()},
        "decoder_state_dict": {key: value.detach().cpu() for key, value in artifact.decoder_state_dict.items()},
    }
    torch.save(payload, path)


def load_artifact(path: Path) -> SyntheticSegArtifact:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return SyntheticSegArtifact(
        image_size=int(payload["image_size"]),
        threshold=float(payload["threshold"]),
        min_area=int(payload["min_area"]),
        open_kernel=int(payload["open_kernel"]),
        close_kernel=int(payload["close_kernel"]),
        final_dilate_kernel=int(payload.get("final_dilate_kernel", 1)),
        freeze_encoder=bool(payload.get("freeze_encoder", True)),
        encoder_state_dict=payload["encoder_state_dict"],
        decoder_state_dict=payload["decoder_state_dict"],
    )


def build_model_from_artifact(artifact: SyntheticSegArtifact) -> SyntheticSegNet:
    encoder = build_model(weights_path=None)
    encoder.load_state_dict(artifact.encoder_state_dict)
    model = SyntheticSegNet(encoder=encoder)
    model.load_state_dict(artifact.decoder_state_dict, strict=False)
    model.eval()
    return model
